"""Benchmark CPU inference energy for the V5 KD detector on a video stream.

Energy estimation methodology:
  Direct hardware energy counters (Intel RAPL / AMD MSR) are not exposed on
  this VM, so we use the standard literature fallback:

      E_estimated = process_cpu_seconds × P_per_active_thread

  where P_per_active_thread = (CPU TDP) / (num_logical_threads).

  For AMD EPYC 7443: TDP=200W, 48 logical threads → 4.17 W/thread.
  This is conservative — peak per-thread power can be higher under turbo,
  lower at idle. Reported numbers are therefore *order of magnitude*
  estimates, not lab-grade measurements.

We measure separately:
  (a) wall-clock time end-to-end (includes I/O, video decode, etc.)
  (b) inference-only time (pure model forward, summed over frames)
  (c) process CPU-seconds (psutil, summed across all threads)

Outputs:
  - per-frame latency (mean/std)
  - effective FPS
  - total wall time
  - process CPU-seconds
  - estimated energy (joules)
  - estimated per-frame energy (millijoules)

Usage:
  python cifar100_spikedetect/benchmark_energy.py \
      --video /tmp/dashcam.webm \
      --checkpoint cifar100_spikedetect_V5_kd \
      --max-frames 600 \
      --frame-stride 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cifar100_spikedetect.data import VOC_CLASSES
from cifar100_spikedetect.model import build_retinanet
from cifar100_spikedetect.yolov8_detector import build_yolov8_detector


# --- CLI ----------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--checkpoint", default="cifar100_spikedetect_V5_kd")
parser.add_argument("--max-frames", type=int, default=None,
                    help="cap total frames processed (for speed)")
parser.add_argument("--frame-stride", type=int, default=1,
                    help="process every Nth frame")
parser.add_argument("--score-thresh", type=float, default=0.25)
parser.add_argument("--img-size", type=int, default=416)
parser.add_argument("--num-threads", type=int, default=None,
                    help="torch CPU threads (default = num_logical_threads)")
parser.add_argument("--cpu-tdp", type=float, default=200.0,
                    help="CPU TDP in Watts (AMD EPYC 7443 = 200W)")
parser.add_argument("--no-warmup", action="store_true")
args = parser.parse_args()


# --- Force CPU + thread settings ----------------------------------------

torch.set_num_threads(args.num_threads or os.cpu_count())
device = torch.device("cpu")
print(f"Device: CPU  Threads: {torch.get_num_threads()}")
print(f"Logical CPUs available: {os.cpu_count()}  ")
print(f"Estimated TDP/thread: {args.cpu_tdp / os.cpu_count():.2f} W "
      f"(TDP={args.cpu_tdp}W / {os.cpu_count()} threads)")


# --- Model --------------------------------------------------------------

ck_path = PROJECT_ROOT / "checkpoints" / f"{args.checkpoint}_best.pth"
ckpt = torch.load(str(ck_path), map_location=device, weights_only=False)
classes = ckpt.get("classes", VOC_CLASSES)
img_size = ckpt.get("img_size", args.img_size)

print(f"\nLoading {args.checkpoint} (epoch {ckpt['epoch']}, mAP@50={ckpt.get('map', 0):.3f})...")
state = ckpt["model_state_dict"]
is_yolov8 = any(k.startswith("head.cls_branches.") for k in state.keys())
print(f"  Detector type: {'YOLOv8 (V7/V8)' if is_yolov8 else 'RetinaNet (V3-V5)'}")
if is_yolov8:
    model = build_yolov8_detector(
        num_classes=len(classes),
        num_steps=ckpt.get("num_steps", 8),
        backbone_source="none",
        trainable_backbone_layers=4,
    )
else:
    model = build_retinanet(
        num_classes=len(classes),
        num_steps=ckpt.get("num_steps", 8),
        trainable_backbone_layers=3,
        min_size=img_size, max_size=img_size,
    )
model.load_state_dict(state, strict=False)
model.to(device).eval()
if hasattr(model, "score_thresh"):
    model.score_thresh = args.score_thresh

n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params:,}  ({n_params * 4 / 1024**2:.1f} MB fp32)")


# --- Open video ---------------------------------------------------------

cap = cv2.VideoCapture(args.video)
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
W_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nVideo: {args.video}")
print(f"  {W_src}x{H_src} @ {fps_in:.1f} fps, {total} frames "
      f"({total / fps_in:.1f}s duration)")
print(f"  Frame stride: {args.frame_stride} (will process every {args.frame_stride}th frame)")


# --- Warmup -------------------------------------------------------------

if not args.no_warmup:
    print("\nWarming up (5 dummy forward passes)...")
    dummy = torch.randn(3, img_size, img_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model([dummy])
    print("  Warmup done.")


# --- Inference loop with measurement ------------------------------------

print(f"\n=== Starting benchmark ===")
proc = psutil.Process()

# Snapshot CPU times before
cpu_times_start = proc.cpu_times()
mem_start = proc.memory_info().rss / 1024**2

wall_start = time.time()
inference_total = 0.0  # pure model forward time
preprocess_total = 0.0
total_dets = 0
frame_idx = 0
processed = 0
per_frame_latencies = []

with torch.no_grad():
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames is not None and processed >= args.max_frames:
            break
        if frame_idx % args.frame_stride != 0:
            frame_idx += 1
            continue

        # Preprocess
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )[0]
        t1 = time.perf_counter()
        preprocess_total += (t1 - t0)

        # Pure inference
        t2 = time.perf_counter()
        out = model([t])[0]
        t3 = time.perf_counter()
        inference_total += (t3 - t2)
        per_frame_latencies.append(t3 - t2)

        # Count valid detections (above threshold)
        valid = (out["scores"] >= args.score_thresh).sum().item()
        total_dets += valid

        processed += 1
        frame_idx += 1

        if processed % 25 == 0:
            elapsed = time.time() - wall_start
            fps_eff = processed / elapsed
            print(f"  {processed} frames done | {fps_eff:.2f} fps eff | "
                  f"avg latency {np.mean(per_frame_latencies)*1000:.1f} ms | "
                  f"dets last frame: {valid}")

cap.release()
wall_end = time.time()
cpu_times_end = proc.cpu_times()
mem_end = proc.memory_info().rss / 1024**2


# --- Compute results ----------------------------------------------------

wall_secs = wall_end - wall_start
process_cpu_secs = (
    (cpu_times_end.user - cpu_times_start.user) +
    (cpu_times_end.system - cpu_times_start.system)
)

avg_latency = float(np.mean(per_frame_latencies))
median_latency = float(np.median(per_frame_latencies))
p95_latency = float(np.percentile(per_frame_latencies, 95))
fps_inference_only = processed / inference_total
fps_wall = processed / wall_secs

# Energy estimate: CPU-seconds × (TDP / threads)
power_per_thread = args.cpu_tdp / os.cpu_count()
estimated_energy_joules = process_cpu_secs * power_per_thread
energy_per_frame_mj = (estimated_energy_joules / processed) * 1000  # millijoules

# Compare to ANN baseline (literature: ~200 mJ for ANN detector on equivalent input)
ann_baseline_mj = 200.0
efficiency_ratio = ann_baseline_mj / energy_per_frame_mj


# --- Report -------------------------------------------------------------

print("\n" + "=" * 60)
print("=== BENCHMARK RESULTS ===")
print("=" * 60)

print(f"\nFrames processed:         {processed}")
print(f"Total wall-clock time:    {wall_secs:.2f} s")
print(f"Pure inference time:      {inference_total:.2f} s")
print(f"Preprocessing time:       {preprocess_total:.2f} s")
print(f"Process CPU-seconds:      {process_cpu_secs:.2f} s "
      f"(over {os.cpu_count()} logical CPUs)")
print(f"Memory usage delta:       {mem_end - mem_start:+.1f} MB "
      f"(end: {mem_end:.0f} MB)")

print(f"\n--- Latency ---")
print(f"  Mean per-frame:         {avg_latency*1000:.1f} ms")
print(f"  Median per-frame:       {median_latency*1000:.1f} ms")
print(f"  P95 per-frame:          {p95_latency*1000:.1f} ms")
print(f"  Inference-only FPS:     {fps_inference_only:.2f}")
print(f"  End-to-end FPS (wall):  {fps_wall:.2f}")

print(f"\n--- Energy estimate ---")
print(f"  CPU TDP assumed:        {args.cpu_tdp:.0f} W "
      f"({power_per_thread:.2f} W/thread × {os.cpu_count()} threads)")
print(f"  Total CPU-seconds:      {process_cpu_secs:.2f}")
print(f"  Estimated total energy: {estimated_energy_joules:.2f} J")
print(f"  Energy per frame:       {energy_per_frame_mj:.2f} mJ")

print(f"\n--- Comparison ---")
print(f"  Equivalent ANN detector (literature): ~{ann_baseline_mj:.0f} mJ/frame")
print(f"  Estimated efficiency ratio:           {efficiency_ratio:.2f}× "
      f"({'better' if efficiency_ratio > 1 else 'worse'})")

print(f"\n--- Detection summary ---")
print(f"  Total detections (above {args.score_thresh}): {total_dets}")
print(f"  Average per frame:      {total_dets / processed:.2f}")

print("\n" + "=" * 60)
print("Note: energy is estimated from CPU-time × (TDP/threads).")
print("Direct RAPL/MSR measurement was not available on this VM.")
print("=" * 60)
