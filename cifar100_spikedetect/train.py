"""Training loop for SEW-RetinaNet on Pascal VOC or COCO.

Path A progression (V3 → V6):
  V3_imagenet : swap backbone to ImageNet-pretrained SEW-ResNet-18 weights.
  V4_mosaic   : V3 + Mosaic/MixUp strong augmentation.
  V5_kd       : V4 + KD from frozen ANN teacher (torchvision RetinaNet-R50).
  V6_spiking  : V5 + spike-native I-LIF FPN/head.

Single-file configuration at top, Tee logger, best-mAP checkpointing,
AdamW with warmup + cosine schedule, optional AMP.
"""

import copy
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from cifar100_spikedetect.model import build_retinanet


# --- ModelEMA: exponential moving average of weights for cleaner eval ---

class ModelEMA:
    """Maintain a shadow copy of `model` with weights = decay * ema + (1-decay) * current.
    Eval the EMA model rather than the raw training weights -- typically +0.5-1.5 mAP.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.ema.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        # Sync buffers (BN running stats, snn.Leaky lazy state)
        for e, b in zip(self.ema.buffers(), model.buffers()):
            if e.shape == b.shape and e.dtype == b.dtype:
                e.copy_(b)
            else:
                e.data = b.data.clone()


# --- Progressive backbone unfreezing schedule ---------------------------

def set_trainable_backbone_layers(model, n):
    """Set how many top backbone layer-groups are trainable.
       n=0 freezes all backbone; n=4 unfreezes everything.
       layer1=blocks[0..1], layer2=blocks[2..3], layer3=blocks[4..5], layer4=blocks[6..7].
       Stem is always trainable (random init).
    """
    block_cutoff = (4 - n) * 2  # how many blocks (from layer1 forward) to freeze
    for i, block in enumerate(model.backbone.body.blocks):
        for p in block.parameters():
            p.requires_grad = (i >= block_cutoff)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def progressive_unfreeze_schedule(epoch):
    """Return how many backbone layer-groups should be unfrozen at this epoch.
    Schedule: 1→L4 only, 2→+L3, 3→+L2 (matches our prior trainable_backbone_layers=3).
    L1 stays frozen always — early features are generic + L1 has biggest activations
    (full unfreeze tipped batch=10+KD over 24 GB memory limit at epoch 8).
    """
    if epoch < 2:
        return 1   # head + L4 only
    if epoch < 5:
        return 2   # + L3
    return 3       # + L2 (cap here — L1 stays frozen for memory + irrelevant features)


# --- Experiment config ---------------------------------------------------

EXPERIMENT = "V8_512"

# schema: dataset, epochs, warmup, max_iters, do_eval,
#         backbone_source, warm_start_ckpt, warm_start_scope, aug_level, label
EXPERIMENTS = {
    "V3_debug": ("coco", 1, 0, 50, False,
                 'imagenet_ann', None, 'none', 'basic',
                 "V3 smoke test — 50 iters, no eval"),
    "V3_imagenet": ("coco", 20, 2, None, True,
                    'imagenet_ann',
                    "cifar100_detect_V2_coco_best.pth", 'head_fpn',
                    'basic',
                    "V3 ImageNet backbone + V2 head/FPN warm-start"),
    "V4_mosaic": ("coco", 20, 2, None, True,
                  'imagenet_ann',
                  "cifar100_spikedetect_V3_imagenet_best.pth", 'all',
                  'strong',
                  "V4 = V3 + Mosaic/MixUp"),
    "V5_kd": ("coco", 20, 2, None, True,
              'imagenet_ann',
              "cifar100_spikedetect_V4_mosaic_best.pth", 'all',
              'strong',
              "V5 = V4 + KD from ANN teacher (enable via USE_KD)"),
    "V6_spiking": ("coco", 15, 2, None, True,
                   'imagenet_ann',
                   "cifar100_spikedetect_V5_kd_best.pth", 'all',
                   'strong',
                   "V6 = V5 + spike-native FPN/head (I-LIF K=4)"),
    "V7_yolov8": ("coco", 20, 2, None, True,
                  'imagenet_ann',
                  "cifar100_spikedetect_V5_kd_best.pth", 'backbone_only',
                  'strong',
                  "V7 = V5 backbone + YOLOv8 head (TAL+CIoU+DFL via ultralytics)"),
    "V7_debug":  ("coco", 1, 0, 50, False,
                  'imagenet_ann',
                  "cifar100_spikedetect_V5_kd_best.pth", 'backbone_only',
                  'basic',
                  "V7 50-iter smoke test"),
    "V7_sanity": ("coco", 1, 0, None, True,
                  'imagenet_ann',
                  "cifar100_spikedetect_V5_kd_best.pth", 'backbone_only',
                  'strong',
                  "V7 1-epoch sanity check (full train + eval)"),
    "V8_512": ("coco", 30, 2, None, True,
               'imagenet_ann',
               "cifar100_spikedetect_V7_yolov8_best.pth", 'all',
               'strong',
               "V8 = V7 weights at 512x512, 30 epochs, batch=8, num_workers=14"),
}

(dataset_name, num_epochs, warmup_epochs, max_iters, do_eval,
 backbone_source, warm_start_ckpt, warm_start_scope, aug_level,
 exp_desc) = EXPERIMENTS[EXPERIMENT]
EXPERIMENT_NAME = f"cifar100_spikedetect_{EXPERIMENT}"

# --- Feature flags (enable stage-specific code paths) ------------------
USE_KD = EXPERIMENT in ("V5_kd", "V7_yolov8", "V7_debug", "V7_sanity", "V8_512")
USE_SPIKING_HEAD = EXPERIMENT in ("V6_spiking",)
USE_YOLO = EXPERIMENT in ("V7_yolov8", "V7_debug", "V7_sanity", "V8_512")


# --- Hyperparameters -----------------------------------------------------

SEED = 42
img_size = 416
num_steps = 8

if dataset_name == "coco":
    batch_size = 20
    num_workers = 10
    lr_head = 3e-4
    lr_backbone = 3e-5
    use_amp = True
else:
    batch_size = 8
    num_workers = 4
    lr_head = 5e-4
    lr_backbone = 5e-5
    use_amp = False

# KD adds an ANN teacher (~37M params) + its own forward activations.
# With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True the allocator
# avoids fragmentation, letting us push batch=16 (was OOM at 12 before).
if USE_KD:
    batch_size = 10
    num_workers = 8

# For V6 (spiking head) use lower LR since we're fine-tuning warmed-up convs
if USE_SPIKING_HEAD:
    lr_head *= 0.33

# V8: 512x512 input — measured peak 26.7 GB / 33.7 GB on RTX 5090 with
# batch=8 (79% used, ~7 GB headroom for fragmentation drift). num_workers=14
# uses ~30% of the 48-thread Threadripper without contending with the main
# process. Going b9 is feasible but eats headroom; b10 OOMs.
if EXPERIMENT == "V8_512":
    img_size = 512
    batch_size = 8
    num_workers = 14

weight_decay = 1e-4
grad_clip_norm = 1.0
trainable_backbone_layers = 3

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# --- Log tee -------------------------------------------------------------

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"{EXPERIMENT_NAME}.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


log_file = open(log_path, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"=== Experiment {EXPERIMENT}: {exp_desc} ===")
print(f"Log: {log_path}")
print(f"dataset={dataset_name} batch={batch_size} img={img_size} "
      f"epochs={num_epochs} warmup={warmup_epochs} amp={use_amp}")
print(f"backbone_source={backbone_source}  warm_start={warm_start_ckpt} "
      f"scope={warm_start_scope}  aug={aug_level}")
print(f"USE_KD={USE_KD}  USE_SPIKING_HEAD={USE_SPIKING_HEAD}")


# --- Data ----------------------------------------------------------------

if dataset_name == "coco":
    from cifar100_spikedetect.data_coco import build_dataloaders as build_coco_loaders
    train_loader, val_loader, class_names = build_coco_loaders(
        root=str(PROJECT_ROOT / "data" / "coco"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        aug_level=aug_level,
    )
    num_classes = len(class_names)
else:
    from cifar100_spikedetect.data import build_dataloaders as build_voc_loaders, VOC_CLASSES
    train_loader, val_loader = build_voc_loaders(
        root=str(PROJECT_ROOT / "data" / "voc"),
        img_size=img_size, batch_size=batch_size, num_workers=num_workers,
        download=False,
    )
    class_names = VOC_CLASSES
    num_classes = len(class_names)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
print(f"num_classes={num_classes} (incl. background)")


# --- Model ---------------------------------------------------------------

base_cifar_ckpt = PROJECT_ROOT / "checkpoints" / "cifar100_sew_T_best.pth"
if USE_YOLO:
    from cifar100_spikedetect.yolov8_detector import build_yolov8_detector
    model = build_yolov8_detector(
        num_classes=num_classes,
        num_steps=num_steps,
        backbone_ckpt=str(base_cifar_ckpt) if base_cifar_ckpt.exists() else None,
        backbone_source=backbone_source,
        trainable_backbone_layers=trainable_backbone_layers,
    ).to(device)
else:
    model = build_retinanet(
        num_classes=num_classes,
        num_steps=num_steps,
        backbone_ckpt=str(base_cifar_ckpt) if base_cifar_ckpt.exists() else None,
        backbone_source=backbone_source,
        trainable_backbone_layers=trainable_backbone_layers,
        spiking_fpn=USE_SPIKING_HEAD,
        K=4,
    ).to(device)


# --- Warm-start from previous experiment checkpoint ---------------------

def warm_start_from(ckpt_path, scope='all'):
    """scope:
         'all'           — load every matching tensor
         'head_fpn'      — load only non-backbone tensors (FPN + heads)
         'backbone_only' — load only backbone weights (V7: head changed, can't reuse)
    """
    ws = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    src_state = ws["model_state_dict"]
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in src_state.items():
        if k not in own_state or own_state[k].shape != v.shape:
            skipped += 1
            continue
        if scope == 'head_fpn' and k.startswith('backbone.body.'):
            continue   # preserve the freshly-loaded ImageNet backbone
        if scope == 'backbone_only' and not k.startswith('backbone.body.'):
            continue   # only transfer the SEW backbone, head/FPN start fresh
        own_state[k] = v
        loaded += 1
    model.load_state_dict(own_state)
    print(f"Warm-started from {ckpt_path.name} (scope={scope}): "
          f"loaded {loaded} keys, skipped {skipped}")


if warm_start_ckpt is not None:
    ws_path = PROJECT_ROOT / "checkpoints" / warm_start_ckpt
    if ws_path.exists():
        warm_start_from(ws_path, scope=warm_start_scope)
    else:
        print(f"(warning) warm-start ckpt not found: {ws_path}")

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_total:,}  Trainable: {n_train:,}")


# --- KD teacher (V5+) ---------------------------------------------------

teacher = None
KD_WEIGHT = 1.0  # coefficient on KD loss
if USE_KD:
    from cifar100_spikedetect.distill import (
        load_teacher, teacher_fpn_features,
        get_student_fpn_features, feature_kd_loss,
    )
    print("Loading ANN teacher: torchvision RetinaNet-R50-FPN-v2 @ 416...")
    teacher = load_teacher(device, min_size=img_size, max_size=img_size)


# --- Optimizer ----------------------------------------------------------

def split_params(model):
    bb, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (bb if name.startswith("backbone.body.") else head).append(p)
    return bb, head


bb_params, head_params = split_params(model)
print(f"Backbone trainable: {sum(p.numel() for p in bb_params):,}")
print(f"Head/FPN trainable: {sum(p.numel() for p in head_params):,}")

optimizer = torch.optim.AdamW([
    {"params": bb_params, "lr": lr_backbone},
    {"params": head_params, "lr": lr_head},
], weight_decay=weight_decay)


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * t))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# bf16 autocast: wider exponent range than fp16, prevents focal-loss underflow.
# No GradScaler needed (bf16 gradients don't underflow like fp16 does).
amp_dtype = torch.bfloat16 if use_amp else None
nan_skip_counter = 0

# EMA shadow model — track an exponential moving average of weights for eval.
USE_EMA = True
ema = ModelEMA(model, decay=0.999) if USE_EMA else None
print(f"USE_EMA={USE_EMA}  USE_PROGRESSIVE_UNFREEZE=True")


# --- Checkpointing ------------------------------------------------------

checkpoint_dir = PROJECT_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / f"{EXPERIMENT_NAME}_best.pth"
start_epoch = 0
best_map = 0.0

if checkpoint_path.exists():
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Prefer raw model weights for training continuation; fall back to EMA
    raw_state = ckpt.get("raw_model_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(raw_state)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_map = ckpt.get("map", 0.0)
    # Re-init EMA from the freshly loaded raw weights
    if ema is not None:
        ema.ema.load_state_dict(ckpt["model_state_dict"])  # primary_state == EMA weights
    print(f"Resumed from epoch {ckpt['epoch']} (mAP: {best_map:.3f})")
else:
    print("No resume checkpoint — fresh start for this experiment.")


# --- Training -----------------------------------------------------------

def move_targets(targets, device):
    return [{"boxes": t["boxes"].to(device, non_blocking=True),
             "labels": t["labels"].to(device, non_blocking=True)} for t in targets]


# --- Memory canary ------------------------------------------------------
# Flip MEM_CANARY=True to run a single forward+backward at the current
# (img_size, batch_size, USE_KD) config and print peak VRAM, then exit.
# Use this before committing to a long run at a new resolution.
MEM_CANARY = False
if MEM_CANARY:
    print(f"[canary] one forward+backward at img={img_size} batch={batch_size} USE_KD={USE_KD}...")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    imgs, targets = next(iter(train_loader))
    imgs = [img.to(device, non_blocking=True) for img in imgs]
    targets = move_targets(targets, device)
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if USE_KD and teacher is not None:
                t_feats = teacher_fpn_features(teacher, imgs)
                s_feats = (model.extract_fpn_features(imgs) if USE_YOLO
                           else get_student_fpn_features(model, imgs))
                loss = loss + KD_WEIGHT * feature_kd_loss(s_feats, t_feats)
    else:
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        if USE_KD and teacher is not None:
            t_feats = teacher_fpn_features(teacher, imgs)
            s_feats = (model.extract_fpn_features(imgs) if USE_YOLO
                       else get_student_fpn_features(model, imgs))
            loss = loss + KD_WEIGHT * feature_kd_loss(s_feats, t_feats)
    loss.backward()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[canary] img={img_size} batch={batch_size} "
          f"peak={peak_gb:.2f} GB / {total_gb:.1f} GB "
          f"({100*peak_gb/total_gb:.0f}% used)")
    sys.exit(0)


for epoch in range(start_epoch, start_epoch + num_epochs):
    # Progressive unfreezing — uses ABSOLUTE epoch so resume preserves the
    # already-unfrozen state (otherwise resume would refreeze L2/L3).
    n_unfrozen = progressive_unfreeze_schedule(epoch)
    n_train = set_trainable_backbone_layers(model, n_unfrozen)
    print(f"--- Epoch {epoch}: unfreezing {n_unfrozen}/4 backbone layer-groups, "
          f"trainable params {n_train:,} ---")

    model.train()
    if USE_YOLO:
        running = {"cls": 0.0, "ciou": 0.0, "dfl": 0.0, "total": 0.0}
    else:
        running = {"classification": 0.0, "bbox_regression": 0.0, "total": 0.0}
    n_batches = 0
    t0 = time.time()

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        if max_iters is not None and batch_idx >= max_iters:
            break
        imgs = [img.to(device, non_blocking=True) for img in imgs]
        targets = move_targets(targets, device)

        # V4+: apply MixUp batch-wise with 50% probability
        if aug_level == "strong":
            from cifar100_spikedetect.augs import mixup_batch
            imgs, targets = mixup_batch(imgs, targets, prob=0.5, alpha=8.0)

        if sum(t["boxes"].shape[0] for t in targets) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                if USE_KD and teacher is not None:
                    t_feats = teacher_fpn_features(teacher, imgs)
                    if USE_YOLO:
                        s_feats = model.extract_fpn_features(imgs)
                    else:
                        s_feats = get_student_fpn_features(model, imgs)
                    kd_loss = feature_kd_loss(s_feats, t_feats)
                    loss = loss + KD_WEIGHT * kd_loss
                    loss_dict["kd"] = kd_loss
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if USE_KD and teacher is not None:
                t_feats = teacher_fpn_features(teacher, imgs)
                if USE_YOLO:
                    s_feats = model.extract_fpn_features(imgs)
                else:
                    s_feats = get_student_fpn_features(model, imgs)
                kd_loss = feature_kd_loss(s_feats, t_feats)
                loss = loss + KD_WEIGHT * kd_loss
                loss_dict["kd"] = kd_loss

        # NaN guard: skip backward+step if this batch produced a non-finite loss.
        # Protects against rare numerical blowups (e.g. focal loss on an outlier).
        if not torch.isfinite(loss):
            nan_skip_counter += 1
            if nan_skip_counter <= 10 or nan_skip_counter % 100 == 0:
                print(f"  [nan-guard] skipping batch {batch_idx} "
                      f"(total skipped: {nan_skip_counter})  loss_dict="
                      f"{ {k: v.item() if torch.isfinite(v) else 'nan' for k, v in loss_dict.items()} }")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        if USE_YOLO:
            running["cls"]   += loss_dict["cls"].item()
            running["ciou"]  += loss_dict["ciou"].item()
            running["dfl"]   += loss_dict["dfl"].item()
        else:
            running["classification"] += loss_dict["classification"].item()
            running["bbox_regression"] += loss_dict["bbox_regression"].item()
        running["total"] += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            elapsed = time.time() - t0
            its = (batch_idx + 1) / max(1e-6, elapsed)
            mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0
            if USE_YOLO:
                print(f"  iter {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  "
                      f"(cls {loss_dict['cls'].item():.4f} "
                      f"ciou {loss_dict['ciou'].item():.4f} "
                      f"dfl {loss_dict['dfl'].item():.4f})  "
                      f"{its:.2f} it/s  gpu_mem={mem_mb:.0f} MB")
            else:
                print(f"  iter {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  (cls {loss_dict['classification'].item():.4f} "
                      f"bbox {loss_dict['bbox_regression'].item():.4f})  "
                      f"{its:.2f} it/s  gpu_mem={mem_mb:.0f} MB")

    scheduler.step()
    if n_batches == 0:
        print("No batches processed — aborting epoch."); continue

    avg_total = running["total"] / n_batches
    lr_bb, lr_hd = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
    if USE_YOLO:
        avg_cls  = running["cls"]  / n_batches
        avg_ciou = running["ciou"] / n_batches
        avg_dfl  = running["dfl"]  / n_batches
        print(f"Epoch {epoch} train | total={avg_total:.4f} cls={avg_cls:.4f} "
              f"ciou={avg_ciou:.4f} dfl={avg_dfl:.4f} "
              f"| LR bb={lr_bb:.2e} head={lr_hd:.2e} | {n_batches} batches in {time.time()-t0:.1f}s")
    else:
        avg_cls  = running["classification"]  / n_batches
        avg_bbox = running["bbox_regression"] / n_batches
        print(f"Epoch {epoch} train | total={avg_total:.4f} cls={avg_cls:.4f} bbox={avg_bbox:.4f} "
              f"| LR bb={lr_bb:.2e} head={lr_hd:.2e} | {n_batches} batches in {time.time()-t0:.1f}s")

    if do_eval:
        # Eval the EMA model (smoother weights → better mAP) when available
        eval_model = ema.ema if ema is not None else model
        eval_model.eval()
        eval_tag = "EMA" if ema is not None else "raw"
        metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = [img.to(device, non_blocking=True) for img in imgs]
                targets = move_targets(targets, device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        preds = eval_model(imgs)
                else:
                    preds = eval_model(imgs)
                preds_cpu = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
                targs_cpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
                metric.update(preds_cpu, targs_cpu)
        result = metric.compute()
        current_map = float(result["map_50"])
        map_all = float(result["map"])
        print(f"Epoch {epoch} eval ({eval_tag}) | mAP@50={current_map:.3f}  mAP@[.5:.95]={map_all:.3f}")

        if current_map > best_map:
            best_map = current_map
            # Save EMA weights as the primary checkpoint (used for eval/inference)
            primary_state = ema.ema.state_dict() if ema is not None else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state_dict": primary_state,
                "raw_model_state_dict": model.state_dict(),  # also keep raw for resume
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_decay": ema.decay if ema is not None else None,
                "map": best_map, "map_all": map_all,
                "classes": class_names,
                "experiment_name": EXPERIMENT_NAME,
                "num_steps": num_steps, "img_size": img_size,
                "dataset": dataset_name,
            }, checkpoint_path)
            print(f">> Best saved at epoch {epoch} (mAP@50: {best_map:.3f}) [{eval_tag}]")

print("Training complete.")
