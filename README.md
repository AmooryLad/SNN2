# Neuromorphic Object Detection with Spiking Neural Networks

A hybrid SNN-ANN object detector built on a SEW-ResNet-18 spiking backbone with a YOLOv8-style detection head, trained on COCO 2017. Final-year B.Sc. (Hons) Computer Science thesis project at Asia Pacific University of Technology and Innovation (APU) / De Montfort University Leicester, aligned with **UN SDG 9 (Industry, Innovation and Infrastructure)** through energy-efficient AI.

**Author**: Ammar Sadik Shaker Attamish (TP072539)
**Supervisor**: Ms. Aziah Abdolla
**2nd Marker**: Dr. Fatin Izzati Ramli

---

## Headline results — V8 (final)

| Metric | Value |
|---|---:|
| **mAP@50 on COCO val** | **0.420** |
| **mAP@[.5:.95] on COCO val** | **0.258** |
| Total parameters | 17.4 M |
| Trainable (epoch 5+) | 17.3 M |
| Input resolution | 512 × 512 |
| SNN timesteps | T = 8 |
| Mean backbone firing rate | ~18 % |
| **Projected energy on neuromorphic silicon** | **62.2 mJ/frame** |
| **Energy ratio vs equivalent ANN** | **1.51 ×** |
| Wall-clock energy on Threadripper CPU | 35.7 J/frame (~7× worse than ANN) |
| Training wall time | ~38 h on RTX 5090 |

The V8 model is **competitive in accuracy with similarly-sized ANN detectors** (Faster R-CNN MobileNetV3-FPN at 19.4M params), and projected to be **1.51× more energy-efficient on neuromorphic silicon**. On commodity CPUs, the T=8 SEW backbone is ~7× slower than ANN — the well-known cost of simulating asynchronous spikes on synchronous hardware.

---

## Architecture

```
Input image  [B, 3, 512, 512]
         ↓
┌────────────────────────────────────────────────────────┐
│  SEW-ResNet-18 backbone        ←  SPIKING (T=8)        │
│  Conv 7×7 s2 → BNTT → LIF → MaxPool                    │
│  Layer 1: 2× SEWBlock  (64 → 64)                       │
│  Layer 2: 2× SEWBlock  (64 → 128, stride 2)  → P3      │
│  Layer 3: 2× SEWBlock  (128 → 256, stride 2) → P4      │
│  Layer 4: 2× SEWBlock  (256 → 512, stride 2) → P5      │
│  Each LIF emits binary {0,1}; SEW-ADD shortcut → {0,1,2}│
│  Time-mean over T=8 at output → continuous floats      │
└────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────┐
│  3-level FPN  (torchvision)    ←  ANN                  │
│  256ch, P3/P4/P5                                       │
└────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────┐
│  YOLOv8 decoupled head         ←  ANN                  │
│  cls / box branches per FPN level                      │
│  Conv-BN-SiLU + final 1×1 conv                         │
└────────────────────────────────────────────────────────┘
         ↓
TaskAlignedAssigner + CIoU + DFL  (ultralytics utils)
         ↓
NMS + Top-k decode → list of {boxes, scores, labels}
```

### Parameter breakdown

| Component | Params | Spiking? | Notes |
|---|---:|:---:|---|
| SEW-ResNet-18 backbone | 11.09 M | ✓ T=8 | Stem + 8 SEWBlocks, BNTT, learnable β & threshold |
| 3-level FPN | ~3.3 M | ✗ | torchvision implementation |
| YOLOv8 head | ~3.0 M | ✗ | Conv-BN-SiLU + final 1×1 |
| **Total** | **17.4 M** | mixed | |

### Spiking layer firing rates (V8 final, COCO val)

```
Backbone layer                 Mean firing rate
─────────────────────────────  ────────────────
blocks.0.conv1 (L1, first)            0.185
blocks.1.conv1 (L1)                   0.261
blocks.2.conv1 (L2, first)            0.319
blocks.3.conv1 (L2)                   0.248
blocks.4.conv1 (L3, first)            0.365  ← max
blocks.5.conv1 (L3)                   0.162
blocks.6.conv1 (L4, first)            0.267
blocks.7.conv1 (L4)                   0.069
─────────────────────────────  ────────────────
Mean across all spiking convs         0.180  (≈18%)
```

This is a healthy SNN sparsity profile, comparable to published spike-driven detectors. The first conv of each stage fires more (gathering features), the last conv fires less (consolidating). Layer 4 (deepest) is sparsest, which is normal.

---

## Version progression

The V3 → V8 history is preserved in the commit log. Each step added one architectural lever, with measured deltas on COCO val.

| Version | Date | Change | mAP@50 | Δ |
|---|---|---|---:|---:|
| V3_imagenet | Apr 24 | ImageNet pretrained backbone + V2 head/FPN warm-start, RetinaNet head | 0.298 | baseline |
| V4_mosaic | Apr 25 | + Mosaic + MixUp strong augmentation | 0.316 | +0.018 |
| V5_kd | Apr 27 | + Knowledge distillation from ANN RetinaNet teacher | 0.335 | +0.019 |
| V5b_atss | Apr 28 | + ATSS matcher (abandoned — regressed to 0.282) | — | (removed) |
| **V7_yolov8** | **Apr 30** | **Replaced RetinaNet head with YOLOv8 (TAL + CIoU + DFL)** | **0.384** | **+0.049** |
| **V8_512** | **May 4** | **Resolution 416 → 512, 30 epochs warm-starting from V7** | **0.420** | **+0.036** |

Total gain V3 → V8: **+0.122 mAP@50** (+41% relative). The largest single jump was V5 → V7 (the head architecture change). Resolution + longer training (V7 → V8) provided the second-largest gain.

---

## Repository structure

```
.
├── cifar100_spikedetect/       # Main detection code (V3 → V8)
│   ├── backbone.py             # SEWBackboneForDetection (SEW-ResNet-18 for detection)
│   ├── data_coco.py            # COCO 2017 dataloader (with mosaic aug)
│   ├── distill.py              # Feature-level KD from RetinaNet-R50-FPN-v2
│   ├── neurons.py              # I-LIF neuron (used by V6 spike-native FPN; not in V8)
│   ├── spiking_fpn.py          # Spike-native FPN (V6, not active in V8)
│   ├── yolov8_detector.py      # End-to-end YOLOv8-style detector wrapper
│   ├── yolov8_head.py          # Decoupled cls/box head with DFL
│   ├── yolov8_loss.py          # TAL matching + CIoU + DFL via ultralytics utils
│   ├── yolov8_decode.py        # DFL → bbox decode + NMS
│   ├── train.py                # Training loop with EMA, progressive unfreeze, AMP
│   ├── augs.py                 # Mosaic + MixUp
│   ├── infer.py                # Single-image inference
│   ├── video_infer.py          # Video inference (used for viz/v8_inference/)
│   ├── video_infer_spikes.py   # Video inference with spike-rate visualization
│   ├── benchmark_energy.py     # Wall-clock CPU energy benchmark (V8 SNN)
│   ├── benchmark_energy_ann.py # Wall-clock CPU energy benchmark (ANN baseline)
│   └── measure_energy.py       # SpikeYOLO-style FLOP × spike-rate × pJ projection
│
├── cifar100_sewresnet/         # CIFAR-100 SEW-ResNet classifier
│   ├── model.py                # SEWBlock, BNTT, LIF (imported by spikedetect/backbone.py)
│   ├── data.py                 # CIFAR-100 dataloader
│   ├── train.py                # CIFAR-100 training
│   └── video_classify.py       # Video classification demo
│
├── caltech101/                 # N-Caltech101 event-based classifier (IR objective)
│   ├── model.py                # SNNConvClassifier
│   ├── data.py                 # N-Caltech101 event-frame loader
│   ├── train.py                # Caltech training
│   ├── inference_gui.py        # Tkinter GUI for live inference
│   └── video_classify.py       # Event-camera-style video viz
│
├── checkpoints/                # Model weights (all LFS-tracked)
│   ├── cifar100_spikedetect_V5_kd_best.pth      ← LFS (311 MB)
│   ├── cifar100_spikedetect_V7_yolov8_best.pth  ← LFS (267 MB)
│   └── cifar100_spikedetect_V8_512_best.pth     ← LFS (267 MB)
│   (V3-V4 ckpts kept locally to stay within free LFS quota)
│
├── logs/                       # Training logs (full audit trail)
│   ├── cifar100_spikedetect_V3_imagenet.log   (V3 → V4 → V5 → V7 → V8)
│   ├── cifar100_spikedetect_V4_mosaic.log
│   ├── cifar100_spikedetect_V5_kd.log
│   ├── cifar100_spikedetect_V7_yolov8.log
│   ├── cifar100_spikedetect_V8_512.log        ← latest run
│   └── ...
│
├── vids/                       # Test videos for inference & benchmarks
│   ├── face-demographics-walking-and-pause.mp4
│   └── Video of object detection test - tech 4 fun.mp4
│
├── viz/                        # Visualization assets + inference outputs
│   ├── v8_inference/           # Annotated inference videos from V8
│   │   ├── face-demographics_v8.mp4
│   │   ├── tech4fun_v8.mp4
│   │   ├── pedestrians_v8.mp4
│   │   └── objdet_test_v8.mp4
│   ├── user_vids/              # Source video clips
│   ├── real_frames/            # Sampled real-world frames
│   └── bbbunny_frames/         # Demo frame set
│
├── data/                       # NOT IN REPO — download separately (see below)
│   ├── coco/                   # COCO 2017 (annotations + train2017 + val2017)
│   └── ncaltech101/            # N-Caltech101 event recordings
│
├── CODE_REVIEW.md              # Early code review of the FC baseline (historical)
├── README.md                   # This file
├── .gitignore
├── .gitattributes              # LFS config for V8 checkpoint
└── .venv-backups/              # Pip freeze for venv reproducibility
    └── 35967209/venv-main-latest.txt
```

---

## Energy efficiency analysis

### Projected energy on neuromorphic silicon (45nm CMOS, 32-bit)

Constants from Horowitz 2014: `E_MAC = 4.6 pJ`, `E_AC = 0.9 pJ`. Spike rates measured on COCO val.

| Component | GFLOPs | V8 energy | Per-component % |
|---|---:|---:|---:|
| Stem (MAC, real-image input) | 0.62 | 2.84 mJ | 5% |
| **Spiking backbone (AC)** | **70.87** | **9.32 mJ** | **15%** |
| FPN (MAC) | 3.41 | 15.67 mJ | 25% |
| Head (MAC) | 7.47 | 34.34 mJ | 56% |
| **Total** | **82.36** | **62.17 mJ** | |
| Equivalent pure ANN | — | **93.60 mJ** | |
| **Energy ratio (ANN / SNN)** | — | **1.51 ×** | |

The spiking backbone, which does 86% of the FLOPs, consumes only 15% of system energy thanks to AC operations and 18% mean firing rate. The FPN + head (still ANN) dominate the energy budget. **V6 path** (spike-ifying FPN + head with I-LIF K=4) projects to ~5.15× — within 10% of SpikeYOLO's published 5.7×.

### Wall-clock energy on commodity hardware

Measured on AMD Threadripper PRO 7965WX (16 threads, TDP 350W, 50-frame video benchmark):

| System | Latency | Wall energy | Ratio |
|---|---:|---:|---:|
| V8 (T=8 SEW backbone, simulated spikes) | 717 ms/frame | **35.7 J/frame** | **0.14× (7× worse)** |
| ANN baseline (Faster R-CNN MobileNetV3) | 99 ms/frame | 5.13 J/frame | 1.0× |

This is the inherent overhead of simulating asynchronous spikes on synchronous CPUs — every spike requires a full float multiplication. The neuromorphic projection (62.2 mJ/frame) and commodity wall-clock (35.7 J/frame) differ by ~580×, which quantifies the value of dedicated neuromorphic silicon for SNN deployment.

---

## Setup

### Requirements

- Python 3.12
- CUDA 12.8 GPU (RTX 30/40/50-series). CPU-only inference works but is ~17× slower.
- ~60 GB disk for COCO 2017 + venv + checkpoints

### Environment recreation

```bash
# Clone the repo
git clone git@github.com:AmooryLad/SNN2.git
cd SNN2

# Pull V8 checkpoint via LFS (auto on clone if Git LFS is installed)
git lfs install
git lfs pull

# Create venv from the pinned freeze
python3.12 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r .venv-backups/35967209/venv-main-latest.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
# Plus packages not in the freeze:
./venv/bin/pip install torchmetrics ultralytics pycocotools \
    opencv-python-headless psutil snntorch
```

### Dataset download

**COCO 2017** (~25 GB) into `data/coco/`:
```bash
mkdir -p data/coco/{annotations,train2017,val2017}
cd data/coco
# Download from https://cocodataset.org/#download
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip val2017.zip annotations_trainval2017.zip
```

**N-Caltech101** (optional, for the Caltech classifier):
```bash
# Download from https://www.garrickorchard.com/datasets/n-caltech101
# Extract into data/ncaltech101/
```

---

## Running inference

### On video (with annotations)

```bash
./venv/bin/python -m cifar100_spikedetect.video_infer \
    --video vids/face-demographics-walking-and-pause.mp4 \
    --checkpoint cifar100_spikedetect_V8_512 \
    --out viz/v8_inference/output.mp4 \
    --score-thresh 0.25 \
    --max-frames 400 \
    --device cuda
```

Pre-computed inference outputs are at [`viz/v8_inference/`](viz/v8_inference/).

### On a single image

```bash
./venv/bin/python -m cifar100_spikedetect.infer \
    --image path/to/image.jpg \
    --checkpoint cifar100_spikedetect_V8_512 \
    --score-thresh 0.25
```

### On COCO val (full mAP eval)

The training script [`train.py`](cifar100_spikedetect/train.py) runs eval per epoch. To run eval-only on the V8 checkpoint:

```python
# Quick standalone eval — see train.py:512-538 for reference logic
from cifar100_spikedetect.yolov8_detector import build_yolov8_detector
from cifar100_spikedetect.data_coco import build_dataloaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

device = torch.device("cuda")
ckpt = torch.load("checkpoints/cifar100_spikedetect_V8_512_best.pth",
                  map_location=device, weights_only=False)
model = build_yolov8_detector(num_classes=81, num_steps=8,
                              backbone_source="none",
                              trainable_backbone_layers=4).to(device).eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

_, val_loader, _ = build_dataloaders(root="data/coco",
                                     img_size=512, batch_size=8,
                                     num_workers=8, aug_level="basic")
metric = MeanAveragePrecision(iou_type="bbox")
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(imgs)
        metric.update([{k: v.cpu() for k, v in p.items()} for p in preds],
                      [{k: v.cpu() for k, v in t.items()} for t in targets])
print(metric.compute())
```

---

## Running energy benchmarks

### Projected energy on neuromorphic silicon (recommended)

```bash
./venv/bin/python -m cifar100_spikedetect.measure_energy \
    --checkpoint cifar100_spikedetect_V8_512 \
    --num-images 100 \
    --device cuda
```

Reports per-layer FLOPs × measured spike rates × E_AC vs E_MAC. ~30 seconds on GPU.

### Wall-clock CPU benchmark

```bash
# SNN
./venv/bin/python -m cifar100_spikedetect.benchmark_energy \
    --video vids/face-demographics-walking-and-pause.mp4 \
    --checkpoint cifar100_spikedetect_V8_512 \
    --max-frames 50 --num-threads 16 --cpu-tdp 350

# ANN baseline (Faster R-CNN MobileNetV3) for apples-to-apples
./venv/bin/python -m cifar100_spikedetect.benchmark_energy_ann \
    --video vids/face-demographics-walking-and-pause.mp4 \
    --model fasterrcnn_mobilenet_v3_large_fpn \
    --max-frames 50 --num-threads 16 --cpu-tdp 350 --img-size 512
```

`--cpu-tdp` should be set to your CPU's TDP (Threadripper PRO 7965WX = 350W; AMD EPYC 7443 = 200W; etc.).

---

## Running training

### V8 reproduction (≈38 hours on RTX 5090)

The current `EXPERIMENT = "V8_512"` setting in [`train.py`](cifar100_spikedetect/train.py) reproduces V8:
- Warm-starts from `cifar100_spikedetect_V7_yolov8_best.pth` (you'd need to provide this — only V8 is in this repo)
- 30 epochs, batch=8, img=512, T=8, KD enabled
- ~75 min/epoch on RTX 5090

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./venv/bin/python -m cifar100_spikedetect.train
```

### Earlier versions

V5_kd, V7_yolov8, and V8_512 checkpoints all ship with the repo via Git LFS. To reproduce V8 from scratch, the full chain is:

- **V8** ← warm-start backbone+head from V7_yolov8 → 30 epochs at 512×512 with KD
- **V7_yolov8** ← warm-start backbone from V5_kd, fresh YOLOv8 head → 20 epochs at 416×416 with KD
- **V5_kd** ← warm-start from V4 (not in repo), KD enabled → 20 epochs at 416×416

To recreate V7 (warm-starting from V5):
```bash
# Edit cifar100_spikedetect/train.py and set EXPERIMENT = "V7_yolov8"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ./venv/bin/python -m cifar100_spikedetect.train
```

V3 and V4 checkpoints are not in the repo (would exceed free LFS quota). To reproduce them you would need to start from `cifar100_detect_V2_coco_best.pth`, which is also not shipped. The full `EXPERIMENTS` configuration dict is at [`train.py:87`](cifar100_spikedetect/train.py#L87).

### Memory canary

Before kicking off a long training run at a new resolution / batch, flip `MEM_CANARY = True` in [`train.py`](cifar100_spikedetect/train.py) (line ~384) to do one forward+backward and report peak VRAM. Helps avoid OOM mid-training.

---

## Notable files for the thesis defense

| File | What it shows |
|---|---|
| [`logs/cifar100_spikedetect_V8_512.log`](logs/cifar100_spikedetect_V8_512.log) | Full V8 training trace — 30 epochs, monotone progression |
| [`logs/cifar100_spikedetect_V7_yolov8.log`](logs/cifar100_spikedetect_V7_yolov8.log) | V7 baseline trace |
| [`logs/cifar100_spikedetect_V3_imagenet.log`](logs/cifar100_spikedetect_V3_imagenet.log) | V3 starting point |
| [`viz/v8_inference/`](viz/v8_inference/) | 4 annotated inference videos |
| [`cifar100_spikedetect/measure_energy.py`](cifar100_spikedetect/measure_energy.py) | The energy projection methodology |

---

## Limitations & honest caveats

1. **Energy claim is for neuromorphic silicon** (45nm CMOS), not the RTX 5090 / Raspberry Pi 5 deployment target mentioned in the original IR. On commodity hardware, the T=8 SEW backbone is ~7× slower and uses ~7× more energy than equivalent ANN. The 1.51× advantage is a hardware-projected number, not a measured wall-clock number.

2. **The V6 spike-native pathway** (spiking FPN + head) is set up in code but not trained. The 5.15× projection assumes a successful V6 training run.

3. **The architecture chosen for academic rigor** (real binary spikes, T=8 LIF) is **not optimal for commodity-hardware speed**. SpikeYOLO's I-LIF design (single-pass integer activations) is more commodity-friendly, but loses the strict-binary-spike property.

4. **Parameter count (17.4M) is competitive** with ANN baselines like Faster R-CNN MobileNetV3 (19.4M), but the V8 mAP@[.5:.95] of 0.258 is below FRCNN-MNv3's ~0.328. Pareto position: V8 is at higher mAP@50 with lower mAP@[.5:.95] — better at detection, worse at tight localization, which is consistent with the reduced precision of spike-based features.

5. **No deployment on Raspberry Pi 5** — the original IR objective. Realistic latency at 17M params + T=8 on a Pi 5 CPU would be several seconds per frame, which is incompatible with the 10-20 ms latency requirements stated in the user-requirements analysis.

---

## Acknowledgements

This work uses:
- **PyTorch** + **torchvision** (Meta AI)
- **snnTorch** (Eshraghian et al.) for LIF neurons and surrogate gradients
- **Ultralytics YOLOv8** utilities (TaskAlignedAssigner, BboxLoss) — used under their license terms
- **MS COCO 2017** (Lin et al.) for detection training
- **N-Caltech101** (Orchard et al.) for event-based classification
- The **SEW-ResNet** architecture (Fang et al., NeurIPS 2021)
- **SpikeYOLO** (Luo et al., ECCV 2024) for the I-LIF neuron design (V6 path)

---

## License

Code in this repository is for academic / educational use. Pretrained weights derived from torchvision's ImageNet ResNet-18 (BSD license) and torchvision's RetinaNet-R50-FPN-v2 (BSD license, used as KD teacher only — not redistributed). YOLOv8 utility code from ultralytics is used under the AGPL-3.0 license — note that this means code that imports `ultralytics.utils.tal` / `ultralytics.utils.loss` inherits AGPL obligations if redistributed.
