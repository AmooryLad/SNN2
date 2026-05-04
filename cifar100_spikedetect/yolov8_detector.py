"""End-to-end YOLOv8-style detector wrapping our SEW-ResNet backbone + FPN.

Replaces torchvision's RetinaNet at the system level. Exposes the same
high-level interface our train.py expects:
  train mode:  model(images, targets) → dict of loss tensors
  eval  mode:  model(images) → list of {'boxes','scores','labels'} dicts

Internals:
  images → ImageNet-normalize → SEW backbone (T=8 spikes, time-mean) →
  FPN (P3/P4/P5, 256ch) → YOLOv8Head (decoupled cls/box) → either
  YOLOv8Loss (train) or DFL decode + NMS (eval).

KD support: `extract_fpn_features(images)` returns the FPN OrderedDict,
matching the format the existing `feature_kd_loss` expects. Same FPN
output channel count (256) → no shape mismatch with the teacher.
"""

import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from cifar100_spikedetect.backbone import SEWBackboneForDetection
from cifar100_spikedetect.yolov8_head import YOLOv8Head
from cifar100_spikedetect.yolov8_loss import YOLOv8Loss
from cifar100_spikedetect.yolov8_decode import decode_predictions


class SEWBackboneWithFPN3(nn.Module):
    """SEW backbone + 3-level FPN (P3/P4/P5) without LastLevelP6P7.

    YOLOv8 only uses 3 levels — P6/P7 add ~0.6M unused params and slow
    inference. This is a slimmer FPN tailored for V7.
    """

    def __init__(self, num_steps=8, fpn_out_channels=256):
        super().__init__()
        self.body = SEWBackboneForDetection(num_steps=num_steps)
        # No extra_blocks → FPN outputs only P3/P4/P5 (named '0','1','2')
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512],
            out_channels=fpn_out_channels,
            extra_blocks=None,
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        feats = self.body(x)
        return self.fpn(feats)


class YOLOv8Detector(nn.Module):
    """Anchor-free, DFL-based detector built on the spiking backbone."""

    def __init__(
        self,
        num_classes=80,
        num_steps=8,
        strides=(8, 16, 32),
        reg_max=16,
        score_thresh=0.05,
        nms_thresh=0.5,
        max_dets=300,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.backbone = SEWBackboneWithFPN3(
            num_steps=num_steps, fpn_out_channels=256,
        )
        self.head = YOLOv8Head(
            in_channels=256,
            num_classes=num_classes,
            reg_max=reg_max,
            num_levels=len(strides),
            strides=strides,
        )
        self.loss_fn = YOLOv8Loss(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=strides,
        )
        self.strides = strides
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.max_dets = max_dets

        self.register_buffer("_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("_std",  torch.tensor(image_std).view(1, 3, 1, 1))

    # ------------------------------------------------------------------ utils

    def _stack_and_normalize(self, images):
        """list of [3, H, W] tensors → [B, 3, H, W] normalized."""
        x = torch.stack(list(images), dim=0).float()
        return (x - self._mean) / self._std

    def _fpn_features(self, images):
        """Run backbone + FPN; return OrderedDict {'0','1','2'}."""
        x = self._stack_and_normalize(images)
        return self.backbone(x)

    def extract_fpn_features(self, images):
        """KD hook: returns FPN feature dict in the same format the teacher uses."""
        return self._fpn_features(images)

    # ----------------------------------------------------------------- forward

    def forward(self, images, targets=None):
        x = self._stack_and_normalize(images)
        feats_dict = self.backbone(x)
        # Convert OrderedDict to ordered list (P3, P4, P5)
        feats = [feats_dict["0"], feats_dict["1"], feats_dict["2"]]

        cls_logits, box_logits = self.head(feats)

        if self.training:
            assert targets is not None, "Training requires targets"
            return self.loss_fn(cls_logits, box_logits, targets)

        # Eval: DFL decode + NMS, return torchvision-compatible dicts
        image_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        return decode_predictions(
            cls_logits, box_logits,
            strides=self.strides,
            reg_max=self.reg_max,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            top_k=self.max_dets,
            image_sizes=image_sizes,
        )


def build_yolov8_detector(
    num_classes=80,
    num_steps=8,
    backbone_ckpt=None,
    backbone_source="cifar100",   # 'cifar100' | 'imagenet_ann' | 'none'
    trainable_backbone_layers=3,
):
    """Build YOLOv8 detector with optional backbone warm-start.

    Mirrors the build_retinanet signature so train.py can branch on a flag.
    """
    model = YOLOv8Detector(num_classes=num_classes, num_steps=num_steps)

    if backbone_source == "cifar100" and backbone_ckpt is not None:
        ck = Path(backbone_ckpt)
        if ck.exists():
            model.backbone.body.load_weights(source="cifar100", path=str(ck))
        else:
            print(f"(warning) backbone ckpt not found: {ck} — skipping load")
    elif backbone_source == "imagenet_ann":
        model.backbone.body.load_weights(source="imagenet_ann")
    elif backbone_source == "none":
        print("(info) backbone initialized from random — no pretrained weights")

    # Differential fine-tune: freeze early backbone blocks
    if trainable_backbone_layers < 4:
        block_cutoff = (4 - trainable_backbone_layers) * 2
        for i, block in enumerate(model.backbone.body.blocks):
            if i < block_cutoff:
                for p in block.parameters():
                    p.requires_grad = False

    return model


if __name__ == "__main__":
    # M4 smoke test: full forward (train + eval) on dummy inputs
    model = build_yolov8_detector(
        num_classes=80, num_steps=8, backbone_source="none",
        trainable_backbone_layers=3,
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"YOLOv8Detector params: {n_total:,}  ({n_total*4/1024**2:.1f} MB fp32)")
    print(f"  trainable: {n_train:,}")
    print(f"  backbone: {sum(p.numel() for p in model.backbone.parameters()):,}")
    print(f"  head:     {sum(p.numel() for p in model.head.parameters()):,}")

    images = [torch.randn(3, 416, 416), torch.randn(3, 416, 416)]
    targets = [
        {
            "boxes":  torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
            "labels": torch.tensor([5], dtype=torch.long),
        },
        {
            "boxes":  torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
    ]

    # Train mode
    model.train()
    losses = model(images, targets)
    total = sum(losses.values())
    print(f"\nTrain output (loss dict): {[(k, round(v.item(), 4)) for k, v in losses.items()]}")
    print(f"  total: {total.item():.4f}")
    total.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  params with gradients: {n_with_grad}")

    # Eval mode
    model.eval()
    with torch.no_grad():
        outs = model(images)
    print(f"\nEval output: {len(outs)} images")
    for i, o in enumerate(outs):
        print(f"  image {i}: boxes {tuple(o['boxes'].shape)}  "
              f"scores {tuple(o['scores'].shape)}  labels {tuple(o['labels'].shape)}")

    # KD hook
    feats = model.extract_fpn_features(images)
    print(f"\nFPN features (for KD): {[(k, tuple(v.shape)) for k, v in feats.items()]}")
