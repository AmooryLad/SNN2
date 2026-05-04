"""DFL decoding + anchor-point generation for YOLOv8-style head.

Three primitives:
  make_anchors(feats, strides)
    → anchor_points [N, 2]   per-cell centers in stride units (0.5, 1.5, ...)
    → stride_tensor [N]      stride of each anchor
    → level_lengths [num_levels]  N per level for splitting back

  dfl_decode(box_logits, reg_max)
    → distances [B, N, 4]    (left, top, right, bottom) in stride units

  distance_to_xyxy(anchor_points, distances, stride_tensor)
    → boxes [B, N, 4]        xyxy in pixel units (multiplied by stride)

Postprocess:
  decode_predictions(cls_logits, box_logits, score_thresh, nms_thresh, top_k)
    → list of {boxes, scores, labels} per image (matches torchvision interface)
"""

import torch
from torchvision.ops import batched_nms


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchor centers and stride mapping for each FPN level.

    Args:
      feats: list of feature tensors [B, C, Hi, Wi] (only spatial size used)
      strides: tuple of stride per level (e.g., (8, 16, 32))
      grid_cell_offset: 0.5 puts the anchor at the cell center

    Returns:
      anchor_points: [N_total, 2] (x, y) cell centers in stride units
      stride_tensor: [N_total]    stride per anchor
      level_lengths: list of int  number of anchors per level (sums to N_total)
    """
    assert len(feats) == len(strides)
    anchor_points, stride_tensor, level_lengths = [], [], []
    device, dtype = feats[0].device, feats[0].dtype

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy_grid, sx_grid = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack([sx_grid, sy_grid], dim=-1).reshape(-1, 2))
        stride_tensor.append(
            torch.full((h * w,), stride, device=device, dtype=dtype)
        )
        level_lengths.append(h * w)

    return torch.cat(anchor_points), torch.cat(stride_tensor), level_lengths


def dfl_decode(box_logits_per_level, reg_max=16):
    """Apply DFL: turn [B, 4*reg_max, Hi, Wi] logits into [B, Hi*Wi, 4] distances.

    Args:
      box_logits_per_level: list of [B, 4*reg_max, Hi, Wi]
      reg_max: number of bins per side (default 16 → max distance = 15 stride units)

    Returns:
      distances: [B, N_total, 4] (left, top, right, bottom) in stride units
    """
    B = box_logits_per_level[0].shape[0]
    device = box_logits_per_level[0].device
    dtype = box_logits_per_level[0].dtype
    proj = torch.arange(reg_max, device=device, dtype=dtype)  # [0, 1, ..., 15]

    out = []
    for box_logits in box_logits_per_level:
        # [B, 4*reg_max, H, W] → [B, 4, reg_max, H, W]
        _, _, H, W = box_logits.shape
        b = box_logits.view(B, 4, reg_max, H, W)
        # Softmax along reg_max axis, then weighted sum with [0..reg_max-1]
        d = (b.softmax(dim=2) * proj.view(1, 1, -1, 1, 1)).sum(dim=2)
        # [B, 4, H, W] → [B, H*W, 4]
        out.append(d.permute(0, 2, 3, 1).reshape(B, H * W, 4))

    return torch.cat(out, dim=1)  # [B, N_total, 4]


def distance_to_xyxy(anchor_points, distances, stride_tensor):
    """Convert (l, t, r, b) distances + anchor centers → xyxy in pixel units.

    Args:
      anchor_points: [N, 2] (x, y) cell centers in stride units
      distances:     [B, N, 4] (l, t, r, b) in stride units
      stride_tensor: [N] stride of each anchor

    Returns:
      boxes: [B, N, 4] xyxy in pixel units
    """
    # All in stride units first
    lt_offset = distances[..., :2]   # (l, t)
    rb_offset = distances[..., 2:]   # (r, b)
    cx_cy = anchor_points.unsqueeze(0)            # [1, N, 2]
    x1y1 = cx_cy - lt_offset                       # [B, N, 2]
    x2y2 = cx_cy + rb_offset                       # [B, N, 2]
    boxes_in_strides = torch.cat([x1y1, x2y2], dim=-1)
    # Multiply by per-anchor stride → pixel units
    return boxes_in_strides * stride_tensor.view(1, -1, 1)


def encode_to_dfl_target(gt_boxes_xyxy, anchor_points, stride_tensor, reg_max=16):
    """Encode GT boxes as discrete DFL distance targets.

    Used by the loss to compute the target distribution that DFL should match.

    Args:
      gt_boxes_xyxy: [N, 4] in pixel units
      anchor_points: [N, 2] cell centers in stride units (matched per-anchor)
      stride_tensor: [N] strides
      reg_max: number of bins

    Returns:
      target_distances: [N, 4] floating distances clipped to [0, reg_max - 1]
    """
    # Convert anchor center to pixel units, then compute (l, t, r, b)
    cx_cy_px = anchor_points * stride_tensor.unsqueeze(-1)   # [N, 2]
    lt = cx_cy_px - gt_boxes_xyxy[:, :2]
    rb = gt_boxes_xyxy[:, 2:] - cx_cy_px
    distances_px = torch.cat([lt, rb], dim=-1)               # [N, 4]
    distances_strides = distances_px / stride_tensor.unsqueeze(-1)
    return distances_strides.clamp(min=0.0, max=reg_max - 1 - 1e-3)


def decode_predictions(
    cls_logits_per_level,
    box_logits_per_level,
    strides,
    reg_max=16,
    score_thresh=0.05,
    nms_thresh=0.5,
    top_k=300,
    image_sizes=None,
):
    """Convert raw head outputs to per-image detection dicts.

    Mirrors torchvision RetinaNet's eval-mode output format so existing
    eval/inference code works with no changes:
        list of {'boxes': [N,4], 'scores': [N], 'labels': [N]}

    Args:
      cls_logits_per_level: list of [B, num_classes, Hi, Wi]
      box_logits_per_level: list of [B, 4*reg_max, Hi, Wi]
      strides: tuple of stride per level
      score_thresh: drop detections below this score
      nms_thresh: IoU threshold for class-agnostic batched NMS
      top_k: max detections per image after NMS
      image_sizes: list of (H, W) tuples to clamp boxes to image bounds (optional)
    """
    B = cls_logits_per_level[0].shape[0]
    num_classes = cls_logits_per_level[0].shape[1]

    anchor_points, stride_tensor, _ = make_anchors(
        cls_logits_per_level, strides, grid_cell_offset=0.5,
    )

    # Decode boxes
    distances = dfl_decode(box_logits_per_level, reg_max=reg_max)
    boxes_all = distance_to_xyxy(anchor_points, distances, stride_tensor)  # [B, N, 4]

    # Flatten cls logits per level → [B, N, C]
    cls_flat = []
    for c in cls_logits_per_level:
        B2, C, H, W = c.shape
        cls_flat.append(c.permute(0, 2, 3, 1).reshape(B2, H * W, C))
    cls_all = torch.cat(cls_flat, dim=1)
    scores_all = cls_all.sigmoid()                                          # [B, N, C]

    results = []
    for b in range(B):
        boxes = boxes_all[b]            # [N, 4]
        scores = scores_all[b]          # [N, C]
        # Threshold: keep (anchor, class) pairs above score_thresh
        scores_max, labels = scores.max(dim=1)  # [N], [N]
        keep = scores_max > score_thresh
        if not keep.any():
            results.append({
                "boxes": boxes.new_zeros((0, 4)),
                "scores": scores.new_zeros((0,)),
                "labels": labels.new_zeros((0,), dtype=torch.long),
            })
            continue

        boxes_k = boxes[keep]
        scores_k = scores_max[keep]
        labels_k = labels[keep]

        # Optional clamp to image bounds
        if image_sizes is not None:
            H, W = image_sizes[b]
            boxes_k[:, [0, 2]] = boxes_k[:, [0, 2]].clamp(0, W)
            boxes_k[:, [1, 3]] = boxes_k[:, [1, 3]].clamp(0, H)

        # Per-class NMS via batched_nms (much cleaner than manual loop)
        nms_keep = batched_nms(boxes_k, scores_k, labels_k, nms_thresh)
        if len(nms_keep) > top_k:
            nms_keep = nms_keep[:top_k]
        results.append({
            "boxes":  boxes_k[nms_keep],
            "scores": scores_k[nms_keep],
            "labels": labels_k[nms_keep],
        })

    return results


if __name__ == "__main__":
    # Smoke test: dummy head output → decoded boxes + sample postprocess
    B, C = 2, 80
    cls_logits = [
        torch.randn(B, C, 52, 52) * 0.5,
        torch.randn(B, C, 26, 26) * 0.5,
        torch.randn(B, C, 13, 13) * 0.5,
    ]
    box_logits = [
        torch.randn(B, 4 * 16, 52, 52) * 0.5,
        torch.randn(B, 4 * 16, 26, 26) * 0.5,
        torch.randn(B, 4 * 16, 13, 13) * 0.5,
    ]

    # Anchor points
    pts, strides_t, lengths = make_anchors(cls_logits, (8, 16, 32))
    print(f"anchor_points: {tuple(pts.shape)}  stride_tensor: {tuple(strides_t.shape)}")
    print(f"  per-level lengths: {lengths}  total: {sum(lengths)}")

    # DFL decode
    dist = dfl_decode(box_logits, reg_max=16)
    print(f"distances: {tuple(dist.shape)}  range: [{dist.min():.2f}, {dist.max():.2f}]")

    # xyxy
    boxes = distance_to_xyxy(pts, dist, strides_t)
    print(f"boxes: {tuple(boxes.shape)}  px range: [{boxes.min():.1f}, {boxes.max():.1f}]")

    # Full decode
    out = decode_predictions(
        cls_logits, box_logits, strides=(8, 16, 32),
        score_thresh=0.05, nms_thresh=0.5, top_k=100,
        image_sizes=[(416, 416), (416, 416)],
    )
    for i, o in enumerate(out):
        print(f"  image {i}: {len(o['boxes'])} dets  "
              f"boxes={tuple(o['boxes'].shape)}  scores={tuple(o['scores'].shape)}")
