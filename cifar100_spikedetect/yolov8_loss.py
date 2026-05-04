"""YOLOv8-style loss adapter for our SEW-ResNet detector.

Glue layer that takes our `YOLOv8Head` outputs + COCO-format targets,
and computes the full {cls, ciou, dfl} loss using ultralytics' battle-tested
utilities (TaskAlignedAssigner, BboxLoss).

We use ultralytics ONLY for these utilities — the head, backbone, training
loop, and dataloaders all stay ours. This avoids the AGPL implications of
using their training framework while still benefiting from the verified
loss/matcher implementation.

Inputs from the head:
  cls_logits_per_level: list of [B, num_classes, Hi, Wi]
  box_logits_per_level: list of [B, 4*reg_max,    Hi, Wi]

Inputs from the dataloader (per-image dict matching our existing format):
  targets: list of {'boxes': [N, 4] xyxy pixels, 'labels': [N]}

Output: dict of {'cls', 'ciou', 'dfl'} losses (sums, ready to backprop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.utils.loss import BboxLoss


class YOLOv8Loss(nn.Module):
    """Combined cls + CIoU + DFL loss using TAL matching.

    Hyperparameters follow YOLOv8 defaults:
      box_weight=7.5, cls_weight=0.5, dfl_weight=1.5
      tal_topk=10, tal_alpha=0.5, tal_beta=6.0
    """

    def __init__(
        self,
        num_classes=80,
        reg_max=16,
        strides=(8, 16, 32),
        tal_topk=10,
        box_weight=7.5,
        cls_weight=0.5,
        dfl_weight=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=num_classes,
            alpha=0.5,
            beta=6.0,
            stride=list(strides),
        )
        self.bbox_loss = BboxLoss(reg_max=reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # Projection vector for DFL softmax → expectation
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float))

    # ------------------------------------------------------------------ utils

    def _flatten_predictions(self, cls_logits_per_level, box_logits_per_level):
        """Flatten per-level head outputs into combined [B, N_total, *] tensors.

        Returns:
          pred_scores: [B, N_total, num_classes]    (raw logits, pre-sigmoid)
          pred_distri: [B, N_total, 4*reg_max]      (raw logits, pre-softmax)
        """
        cls_flat, box_flat = [], []
        for c, b in zip(cls_logits_per_level, box_logits_per_level):
            B, _, H, W = c.shape
            cls_flat.append(c.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes))
            box_flat.append(b.permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max))
        return torch.cat(cls_flat, dim=1), torch.cat(box_flat, dim=1)

    def _decode_bboxes(self, pred_distri, anchor_points):
        """Apply DFL: distri logits → decoded bboxes in stride units.

        pred_distri: [B, N, 4*reg_max]
        anchor_points: [N, 2] (cell centers in stride units)
        Returns: [B, N, 4] xyxy in stride units
        """
        B, N, _ = pred_distri.shape
        # [B, N, 4, reg_max] → softmax → expected distance per side
        d = pred_distri.view(B, N, 4, self.reg_max).softmax(dim=3)
        d = d.matmul(self.proj.to(d.dtype))  # [B, N, 4]
        return dist2bbox(d, anchor_points, xywh=False)  # xyxy in stride units

    def _build_target_tensor(self, targets, batch_size, device, dtype):
        """Convert list-of-dicts targets → padded [B, max_n, 5] tensor.

        Format: per-image padded tensor with [cls, x1, y1, x2, y2] in pixel units.
        Padding rows are zeros; mask_gt distinguishes real entries.
        """
        # Compute padding length
        max_n = max((len(t["labels"]) for t in targets), default=0)
        if max_n == 0:
            empty = torch.zeros(batch_size, 0, 5, device=device, dtype=dtype)
            mask_gt = torch.zeros(batch_size, 0, 1, device=device, dtype=torch.bool)
            return empty[..., 0:1], empty[..., 1:5], mask_gt

        gt = torch.zeros(batch_size, max_n, 5, device=device, dtype=dtype)
        for i, t in enumerate(targets):
            n = len(t["labels"])
            if n > 0:
                gt[i, :n, 0] = t["labels"].to(device, dtype=dtype)
                gt[i, :n, 1:5] = t["boxes"].to(device, dtype=dtype)

        gt_labels = gt[..., 0:1]
        gt_bboxes = gt[..., 1:5]
        # mask_gt: 1 where the row holds a real GT (boxes have nonzero area)
        mask_gt = (gt_bboxes.sum(dim=-1, keepdim=True) > 0)
        return gt_labels, gt_bboxes, mask_gt

    # ------------------------------------------------------------------ forward

    def forward(self, cls_logits_per_level, box_logits_per_level, targets):
        device = cls_logits_per_level[0].device
        dtype = cls_logits_per_level[0].dtype

        # Flatten per-level outputs
        pred_scores, pred_distri = self._flatten_predictions(
            cls_logits_per_level, box_logits_per_level
        )
        B, N, _ = pred_scores.shape

        # Anchor points and per-anchor strides
        anchor_points, stride_tensor = make_anchors(
            cls_logits_per_level, list(self.strides), grid_cell_offset=0.5,
        )
        # anchor_points: [N, 2] (stride units), stride_tensor: [N, 1]

        # Decode predicted boxes (in stride units)
        pred_bboxes_strides = self._decode_bboxes(pred_distri, anchor_points)
        # Convert to pixel units for TAL matching against pixel-unit GT boxes
        pred_bboxes_px = pred_bboxes_strides * stride_tensor

        # Build padded GT tensors
        gt_labels, gt_bboxes, mask_gt = self._build_target_tensor(
            targets, B, device, dtype,
        )

        # ---- TaskAlignedAssigner -------------------------------------------
        # TAL needs:
        #   pd_scores: [B, N, num_classes]  sigmoid (probability)
        #   pd_bboxes: [B, N, 4]            pixel-unit xyxy
        #   anc_points: [N, 2]              pixel-unit centers
        #   gt_labels: [B, max_n, 1]
        #   gt_bboxes: [B, max_n, 4]        pixel-unit xyxy
        #   mask_gt:   [B, max_n, 1]
        anchor_points_px = anchor_points * stride_tensor

        # TAL has internal float32 IoU ops that don't compose with autocast bf16.
        # Run it in float32 outside autocast — gradients aren't needed since
        # TAL sees only detached tensors anyway.
        with torch.amp.autocast(device_type=device.type, enabled=False):
            target_bboxes_px, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid().float(),
                pred_bboxes_px.detach().float(),
                anchor_points_px.float(),
                gt_labels.float(),
                gt_bboxes.float(),
                mask_gt,
            )[1:5]
        # Note: TaskAlignedAssigner returns
        #   (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)

        target_scores_sum = max(target_scores.sum(), 1.0)

        # ---- Cls loss (BCE with TAL soft labels) ---------------------------
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # ---- Box CIoU + DFL loss --------------------------------------------
        loss_iou = pred_scores.new_tensor(0.0)
        loss_dfl = pred_scores.new_tensor(0.0)
        if fg_mask.sum() > 0:
            # BboxLoss expects target_bboxes in *stride units* (it computes against
            # pred_bboxes in stride units), and uses anchor_points in stride units too.
            target_bboxes_strides = target_bboxes_px / stride_tensor
            imgsz = torch.tensor(
                cls_logits_per_level[0].shape[2:], device=device, dtype=dtype,
            ) * self.strides[0]

            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri,
                pred_bboxes_strides,
                anchor_points,
                target_bboxes_strides,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        return {
            "cls":  self.cls_weight * loss_cls,
            "ciou": self.box_weight * loss_iou,
            "dfl":  self.dfl_weight * loss_dfl,
        }


if __name__ == "__main__":
    # M2+M3 smoke test: dummy head output + dummy targets → loss
    from cifar100_spikedetect.yolov8_head import YOLOv8Head

    torch.manual_seed(0)
    head = YOLOv8Head(in_channels=256, num_classes=80, reg_max=16, num_levels=3)
    loss_fn = YOLOv8Loss(num_classes=80, reg_max=16, strides=(8, 16, 32))

    # Fake FPN features at P3/P4/P5 strides for 416×416 input
    feats = [
        torch.randn(2, 256, 52, 52),
        torch.randn(2, 256, 26, 26),
        torch.randn(2, 256, 13, 13),
    ]
    cls_logits, box_logits = head(feats)

    # Two images, varying GT counts
    targets = [
        {
            "boxes":  torch.tensor([[10.0, 10.0, 200.0, 200.0],
                                    [50.0, 100.0, 350.0, 300.0]]),
            "labels": torch.tensor([5, 12], dtype=torch.long),
        },
        {
            "boxes":  torch.tensor([[100.0, 50.0, 250.0, 200.0]]),
            "labels": torch.tensor([7], dtype=torch.long),
        },
    ]

    losses = loss_fn(cls_logits, box_logits, targets)
    total = sum(losses.values())
    print("YOLOv8Loss smoke test:")
    for k, v in losses.items():
        print(f"  {k:5s}: {v.item():.4f}")
    print(f"  total: {total.item():.4f}")

    # Backward check — gradients should flow into head
    total.backward()
    grads = [p.grad.abs().mean().item()
             for p in head.parameters() if p.grad is not None]
    print(f"  grad norms (mean abs): min={min(grads):.2e} max={max(grads):.2e}")
    print(f"  num head params with gradients: {len(grads)}")
