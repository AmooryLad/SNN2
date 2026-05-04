"""Run the SEW-ResNet RetinaNet detector on each frame of a video.

Reuses video I/O + put_text patterns from cifar100/video_classify.py.
RetinaNet handles NMS internally when in eval mode.

Example:
    python cifar100_detect/video_infer.py --video /tmp/bear.mp4 \
        --checkpoint cifar100_detect_V1 --out viz/bear_detected.mp4 \
        --score-thresh 0.3 --max-frames 600
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms

from cifar100_spikedetect.data import VOC_CLASSES
from cifar100_spikedetect.model import build_retinanet
from cifar100_spikedetect.yolov8_detector import build_yolov8_detector


# --- CLI ----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="path to local .mp4")
parser.add_argument("--out", default=None, help="output annotated mp4")
parser.add_argument("--checkpoint", default="cifar100_detect_V1",
                    help="checkpoint name under checkpoints/ (no suffix)")
parser.add_argument("--score-thresh", type=float, default=0.3)
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--frame-stride", type=int, default=1,
                    help="run detector every Nth frame; reuse boxes on others")
parser.add_argument("--img-size", type=int, default=416)
parser.add_argument("--device", default=None, help="cuda or cpu (default auto)")
parser.add_argument("--class-agnostic-nms", type=float, default=0.5,
                    help="Extra NMS pass ignoring class labels at this IoU (0=off)")
parser.add_argument("--top-k-per-frame", type=int, default=None,
                    help="Keep only top-K detections per frame after NMS")
args = parser.parse_args()


# --- Model --------------------------------------------------------------

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck_path = PROJECT_ROOT / "checkpoints" / f"{args.checkpoint}_best.pth"
if not ck_path.exists():
    print(f"Error: checkpoint not found: {ck_path}")
    sys.exit(1)

ckpt = torch.load(ck_path, map_location=device, weights_only=False)
classes = ckpt.get("classes", VOC_CLASSES)
img_size = ckpt.get("img_size", args.img_size)

state = ckpt["model_state_dict"]
is_yolov8 = any(k.startswith("head.cls_branches.") for k in state.keys())
print(f"Detector type: {'YOLOv8 (V7/V8)' if is_yolov8 else 'RetinaNet (V3-V5)'}")
if is_yolov8:
    model = build_yolov8_detector(
        num_classes=len(classes),
        num_steps=ckpt.get("num_steps", 8),
        backbone_source="none",
        trainable_backbone_layers=4,
    )
    model.score_thresh = args.score_thresh
else:
    model = build_retinanet(
        num_classes=len(classes),
        num_steps=ckpt.get("num_steps", 8),
        backbone_ckpt=None,
        trainable_backbone_layers=3,
        min_size=img_size,
        max_size=img_size,
    )
    model.score_thresh = args.score_thresh
model.load_state_dict(state, strict=False)
model.to(device).eval()

print(f"Loaded {args.checkpoint}  epoch {ckpt['epoch']}  mAP@50={ckpt.get('map', 0):.3f}")
print(f"Classes: {len(classes)} (incl. background)  img_size={img_size}")


# --- Video I/O ----------------------------------------------------------

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Failed to open video: {args.video}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W_src}x{H_src} @ {fps:.1f}fps, {total} frames")

out_path = Path(args.out) if args.out else (
    PROJECT_ROOT / "viz" / f"{Path(args.video).stem}_detected.mp4"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W_src, H_src))


# --- Helpers ------------------------------------------------------------

def put_text(img, text, pos, color=(255, 255, 255), scale=0.6, thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


# One color per class (excluding background at idx 0)
rng = np.random.RandomState(42)
CLASS_COLORS = [(0, 0, 0)] + [
    tuple(int(c) for c in rng.randint(50, 230, size=3)) for _ in classes[1:]
]


@torch.no_grad()
def detect(frame_bgr):
    """Run model on a BGR frame; return list of (box_src_coords, label, score)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Convert HxWx3 uint8 -> 3xHxW float [0,1]
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    # Resize to img_size — RetinaNet's internal transform also resizes, but
    # feeding a pre-resized tensor keeps the mapping simple.
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(img_size, img_size),
        mode="bilinear", align_corners=False,
    )[0].to(device)

    outs = model([tensor])
    out = outs[0]

    boxes = out["boxes"].cpu()
    labels = out["labels"].cpu()
    scores = out["scores"].cpu()

    # Filter by score
    keep = scores >= args.score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # Class-agnostic NMS: keep the highest-scoring box per region regardless
    # of class. Fixes the "multiple labels stacked on one object" problem
    # common with low-confidence models like ours (17.5% mAP).
    if args.class_agnostic_nms > 0 and len(boxes) > 0:
        keep_idx = tv_nms(boxes, scores, args.class_agnostic_nms)
        boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]

    if args.top_k_per_frame is not None and len(boxes) > args.top_k_per_frame:
        top_idx = scores.argsort(descending=True)[: args.top_k_per_frame]
        boxes, labels, scores = boxes[top_idx], labels[top_idx], scores[top_idx]

    sx = W_src / img_size
    sy = H_src / img_size
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        detections.append((x1, y1, x2, y2, int(label.item()), float(score.item())))
    return detections


def draw(frame_bgr, detections):
    for x1, y1, x2, y2, label, score in detections:
        color = CLASS_COLORS[label]
        cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        tag = f"{classes[label]} {score*100:.0f}%"
        put_text(frame_bgr, tag, (int(x1) + 4, int(y1) + 18),
                 color=color, scale=0.55, thick=2)
    put_text(frame_bgr, f"SEW-RetinaNet @ {img_size}px  score>{args.score_thresh}",
             (10, 24), scale=0.55)


# --- Main loop ----------------------------------------------------------

processed = 0
last_dets = []
t0 = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    if args.max_frames is not None and processed >= args.max_frames:
        break

    if processed % args.frame_stride == 0:
        last_dets = detect(frame_bgr)

    draw(frame_bgr, last_dets)
    writer.write(frame_bgr)
    processed += 1

    if processed % 50 == 0:
        elapsed = time.time() - t0
        fps_eff = processed / elapsed
        print(f"  {processed}/{total}  ({fps_eff:.1f} fps eff)  last n_dets={len(last_dets)}")

cap.release()
writer.release()
print(f"Saved: {out_path}")
