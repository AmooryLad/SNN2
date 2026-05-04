"""SpikeYOLO-style AC-vs-MAC energy estimate for the V7/V8 detector.

This is a **hardware-agnostic** energy projection — it does NOT measure
wall-clock energy on the host CPU/GPU (use benchmark_energy.py for that).
Instead it counts FLOPs, measures actual spike rates layer-by-layer on real
COCO images, and applies the standard 45nm CMOS energy constants:

    E_MAC = 4.6 pJ   (32-bit multiply-accumulate, ANN convolution)
    E_AC  = 0.9 pJ   (32-bit accumulate, SNN binary-spike convolution)

Energy accounting per Conv2d:
    - Stem conv (image → spikes, real-valued input):   E = FLOPs × E_MAC
    - Backbone convs (spike input):                     E = T × ⟨input⟩ × FLOPs × E_AC
    - FPN convs (mean-pooled continuous input):         E = FLOPs × E_MAC
    - Head convs (continuous input):                    E = FLOPs × E_MAC

⟨input⟩ is the mean nonneg input value averaged across positions and
timesteps. For a strict {0,1} spike train ⟨input⟩ equals firing rate r;
for {0,1,2} SEW-ADD outputs it equals r × mean_level. Both cases collapse
to the same SNN energy formula.

Usage:
    python -m cifar100_spikedetect.measure_energy \
        --checkpoint cifar100_spikedetect_V7_yolov8 \
        --num-images 100 \
        --img-size 416 \
        --device cuda

Reports per-component breakdown (backbone / FPN / head), total energy in
millijoules per frame, and the SNN/ANN ratio.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cifar100_spikedetect.yolov8_detector import build_yolov8_detector
from cifar100_spikedetect.data_coco import build_dataloaders


# Energy constants (Horowitz 2014, 45nm CMOS, 32-bit ops) — picojoules per op
E_MAC_PJ = 4.6
E_AC_PJ  = 0.9


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="cifar100_spikedetect_V8_512",
                   help="checkpoint name (without _best.pth suffix)")
    p.add_argument("--num-images", type=int, default=100,
                   help="number of COCO val images to measure spike rates over")
    p.add_argument("--img-size", type=int, default=None,
                   help="input size; defaults to checkpoint's saved img_size")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--e-mac", type=float, default=E_MAC_PJ,
                   help=f"pJ per MAC op (default {E_MAC_PJ})")
    p.add_argument("--e-ac", type=float, default=E_AC_PJ,
                   help=f"pJ per AC op (default {E_AC_PJ})")
    return p.parse_args()


def conv_flops(module, input_shape, output_shape):
    """MAC count for a single Conv2d call.
    input_shape:  (B, C_in, H_in, W_in)
    output_shape: (B, C_out, H_out, W_out)
    weight: (C_out, C_in/groups, k_h, k_w)
    """
    if not isinstance(module, nn.Conv2d):
        return 0
    B, C_out, H_out, W_out = output_shape
    k_h, k_w = module.kernel_size
    C_in_per_group = module.in_channels // module.groups
    # MACs per output position = C_in_per_group × k_h × k_w (per output channel)
    # Times C_out output channels times spatial size, times batch.
    return B * C_out * H_out * W_out * C_in_per_group * k_h * k_w


def classify_layer(module_name, model):
    """Return one of: 'stem', 'backbone_spiking', 'fpn', 'head', 'other'.

    Routing rules for V7/V8 (yolov8_detector.YOLOv8Detector):
      - model.backbone.body.conv_stem            → 'stem'    (real input → spikes)
      - model.backbone.body.blocks.*.conv*       → 'backbone_spiking'  (spikes in)
      - model.backbone.body.blocks.*.down_conv   → 'backbone_spiking'
      - model.backbone.fpn.*                     → 'fpn'
      - model.head.*                             → 'head'
    """
    if module_name == "backbone.body.conv_stem":
        return "stem"
    if module_name.startswith("backbone.body."):
        return "backbone_spiking"
    if module_name.startswith("backbone.fpn."):
        return "fpn"
    if module_name.startswith("head."):
        return "head"
    return "other"


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Load checkpoint ------------------------------------------------
    ck_path = PROJECT_ROOT / "checkpoints" / f"{args.checkpoint}_best.pth"
    if not ck_path.exists():
        print(f"ERROR: checkpoint not found at {ck_path}")
        print(f"Available checkpoints:")
        for p in sorted((PROJECT_ROOT / "checkpoints").glob("*spikedetect*best.pth")):
            print(f"  {p.name}")
        sys.exit(1)

    print(f"Loading {ck_path.name}...")
    ckpt = torch.load(str(ck_path), map_location=device, weights_only=False)
    classes = ckpt.get("classes", None)
    saved_img_size = ckpt.get("img_size", 416)
    img_size = args.img_size if args.img_size is not None else saved_img_size
    num_classes = len(classes) if classes else 81
    T = args.num_steps

    print(f"  epoch:    {ckpt['epoch']}")
    print(f"  mAP@50:   {ckpt.get('map', float('nan')):.3f}")
    print(f"  img_size: {img_size} (saved: {saved_img_size})")
    print(f"  classes:  {num_classes}")
    print(f"  T:        {T}")

    # --- Build model ----------------------------------------------------
    model = build_yolov8_detector(
        num_classes=num_classes,
        num_steps=T,
        backbone_source="none",   # weights come from checkpoint
        trainable_backbone_layers=4,
    ).to(device).eval()

    state = ckpt.get("model_state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  warning: {len(unexpected)} unexpected keys")

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  params:   {n_total:,}\n")

    # --- Register hooks -------------------------------------------------
    # For each Conv2d, accumulate (FLOPs_per_call, sum_of_input_means) across calls.
    layer_stats = defaultdict(lambda: {
        "category": None, "calls": 0, "flops_per_call": 0,
        "input_mean_sum": 0.0, "weight_shape": None,
    })

    def make_hook(name, category):
        def hook(module, inputs, output):
            x = inputs[0]
            B, C_out, H_out, W_out = output.shape
            flops = conv_flops(module, x.shape, output.shape)
            mean_input = x.clamp(min=0).mean().item()  # nonneg mean for AC accounting
            s = layer_stats[name]
            s["category"] = category
            s["calls"] += 1
            s["flops_per_call"] = flops          # constant across calls (same shapes)
            s["input_mean_sum"] += mean_input
            s["weight_shape"] = tuple(module.weight.shape)
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            cat = classify_layer(name, model)
            handles.append(module.register_forward_hook(make_hook(name, cat)))

    # --- Build COCO val loader ------------------------------------------
    print("Building COCO val loader...")
    _, val_loader, _ = build_dataloaders(
        root=str(PROJECT_ROOT / "data" / "coco"),
        img_size=img_size, batch_size=1, num_workers=2,
        aug_level="basic",
    )
    print(f"  val batches: {len(val_loader)}\n")

    # --- Run inference --------------------------------------------------
    print(f"Measuring spike rates over {args.num_images} val images...")
    n_done = 0
    with torch.no_grad():
        for imgs, _ in val_loader:
            if n_done >= args.num_images:
                break
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            _ = model(imgs)
            n_done += len(imgs)
            if n_done % 25 == 0:
                print(f"  {n_done}/{args.num_images}")
    print(f"  done: {n_done} images\n")

    for h in handles:
        h.remove()

    # --- Aggregate ------------------------------------------------------
    # Per layer: total_flops_per_image = flops_per_call × calls / num_images
    # Per layer: mean_input = input_mean_sum / calls
    # Per layer: energy_per_image = ...

    rows = []
    for name, s in layer_stats.items():
        if s["calls"] == 0:
            continue
        # On a per-image basis (we ran num_images forwards). For backbone
        # convs there are T calls per image (since the time loop is internal).
        # For FPN/head there's 1 call per image. The hook fired total `calls`
        # times across all images, so calls_per_image = calls / num_images.
        calls_per_image = s["calls"] / n_done
        flops_per_image = s["flops_per_call"] * calls_per_image
        mean_input = s["input_mean_sum"] / s["calls"]

        cat = s["category"]
        if cat == "backbone_spiking":
            # Each call processes one timestep with mean(input) average activation.
            # Total AC ops per image = sum_over_calls(mean_input × flops_per_call)
            # = (calls_per_image) × (mean_input averaged over calls) × flops_per_call
            # = flops_per_image × mean_input
            energy_pj = flops_per_image * mean_input * args.e_ac
            ann_equiv_flops = flops_per_image / max(calls_per_image, 1)  # 1 ANN equivalent pass
            ann_equiv_energy = ann_equiv_flops * args.e_mac
            ratio_kind = "AC"
        elif cat == "stem":
            # Stem conv reads the real-valued image. Counted as MAC.
            # Note: it runs T times per image in the current code (with same x),
            # but architecturally only 1 forward is needed. We charge 1× for fair
            # accounting (the redundancy is a software issue, not architectural).
            flops_per_image = s["flops_per_call"]   # 1 ANN equivalent
            energy_pj = flops_per_image * args.e_mac
            ann_equiv_energy = energy_pj
            ratio_kind = "MAC"
        else:  # fpn, head, other
            energy_pj = flops_per_image * args.e_mac
            ann_equiv_energy = energy_pj
            ratio_kind = "MAC"

        rows.append({
            "name": name, "category": cat, "calls_per_img": calls_per_image,
            "flops_per_img": flops_per_image, "mean_input": mean_input,
            "energy_pj": energy_pj, "ann_energy_pj": ann_equiv_energy,
            "kind": ratio_kind,
        })

    # --- Report ---------------------------------------------------------
    print("=" * 100)
    print(f"{'layer':<55} {'cat':<18} {'calls':>5} {'GFLOPs':>9} {'⟨input⟩':>9} {'kind':>4} {'µJ':>9}")
    print("=" * 100)

    by_cat_snn = defaultdict(float)
    by_cat_ann = defaultdict(float)
    by_cat_flops = defaultdict(float)

    for r in sorted(rows, key=lambda x: (x["category"], x["name"])):
        gflops = r["flops_per_img"] / 1e9
        uj = r["energy_pj"] / 1e6
        print(f"  {r['name']:<53} {r['category']:<18} {r['calls_per_img']:>5.1f} "
              f"{gflops:>9.3f} {r['mean_input']:>9.4f} {r['kind']:>4} {uj:>9.3f}")
        by_cat_snn[r["category"]] += r["energy_pj"]
        by_cat_ann[r["category"]] += r["ann_energy_pj"]
        by_cat_flops[r["category"]] += r["flops_per_img"]

    print("=" * 100)
    print(f"\nPer-category energy (per frame):")
    print(f"  {'category':<22} {'GFLOPs':>10} {'SNN energy':>14} {'ANN energy':>14}")
    print(f"  {'─' * 62}")
    total_snn_pj = 0.0
    total_ann_pj = 0.0
    total_flops = 0.0
    for cat in ("stem", "backbone_spiking", "fpn", "head", "other"):
        if by_cat_flops[cat] == 0:
            continue
        gf = by_cat_flops[cat] / 1e9
        snn_uj = by_cat_snn[cat] / 1e6
        ann_uj = by_cat_ann[cat] / 1e6
        print(f"  {cat:<22} {gf:>10.3f} {snn_uj:>11.2f} µJ {ann_uj:>11.2f} µJ")
        total_snn_pj += by_cat_snn[cat]
        total_ann_pj += by_cat_ann[cat]
        total_flops += by_cat_flops[cat]

    snn_mj = total_snn_pj / 1e9
    ann_mj = total_ann_pj / 1e9

    print(f"  {'─' * 62}")
    print(f"  {'TOTAL':<22} {total_flops/1e9:>10.3f} {snn_mj*1000:>11.3f} µJ {ann_mj*1000:>11.3f} µJ")
    print()
    print(f"Per-frame projected energy (45nm CMOS, 32-bit):")
    print(f"  Hybrid SNN-ANN (this model):  {snn_mj:.4f} mJ/frame")
    print(f"  Equivalent pure ANN:           {ann_mj:.4f} mJ/frame")
    print(f"  Energy ratio (ANN / SNN):      {ann_mj/max(snn_mj, 1e-12):.2f}×")
    print()

    # Where is the SNN saving lost?
    if by_cat_snn["fpn"] + by_cat_snn["head"] > 0:
        ann_only_pj = by_cat_snn["fpn"] + by_cat_snn["head"] + by_cat_snn["stem"]
        spiking_pj = by_cat_snn["backbone_spiking"]
        ann_part_frac = ann_only_pj / (ann_only_pj + spiking_pj)
        print(f"Note: {ann_part_frac*100:.0f}% of total energy is ANN-equivalent "
              f"(stem+FPN+head, all using MAC).")
        print(f"      The {(1-ann_part_frac)*100:.0f}% spiking backbone is where "
              f"the AC advantage applies.")
        print(f"      Making FPN+head spike-native (V6 path) would push the ratio "
              f"closer to {args.e_mac/args.e_ac:.1f}×.")

    print()
    print("Caveats:")
    print(f"  • Constants assume 45nm 32-bit (E_MAC={args.e_mac}, E_AC={args.e_ac} pJ).")
    print(f"    For 7nm 8-bit roughly divide both by ~10 — ratio is similar.")
    print(f"  • This is the architectural energy projection FOR neuromorphic")
    print(f"    silicon. Wall-clock energy on this RTX 5090 / RPi 5 is much")
    print(f"    higher because spiking is simulated, not native.")
    print(f"  • Does NOT count BatchNorm, LIF, SiLU, NMS — these are <2% of FLOPs.")


if __name__ == "__main__":
    main()
