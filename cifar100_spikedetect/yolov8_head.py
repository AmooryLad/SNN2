"""YOLOv8-style decoupled detection head.

Anchor-free, per-grid-cell prediction with DFL bbox regression.

Per FPN level (P3/P4/P5 — strides 8/16/32), the head emits:
  cls_logits  [B, num_classes, Hi, Wi]   class logits per grid cell
  box_logits  [B, 4*reg_max,  Hi, Wi]    distribution over distance-to-side
                                          (each side gets reg_max=16 bins)

Decoded by DFL: distance = sum(softmax(logits) * arange(reg_max)) per side.

This is a pure float head (matches YOLOv8 reference). V8 will swap the
`Conv-BN-SiLU` activations for `BNTT-IntegerLIF` to make it spike-native.
"""

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Standard YOLOv8 building block: Conv → BN → SiLU."""

    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv8Head(nn.Module):
    """Decoupled cls/box heads applied at each FPN level.

    Channel widths follow the YOLOv8 reference:
      cls branch: max(in_ch, num_classes) → max(in_ch, num_classes) → num_classes
      box branch: max(16, in_ch//4, 4*reg_max) → same → 4*reg_max
    """

    def __init__(
        self,
        in_channels=256,
        num_classes=80,
        reg_max=16,
        num_levels=3,
        strides=(8, 16, 32),
    ):
        super().__init__()
        assert num_levels == len(strides)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_levels = num_levels
        self.strides = strides
        self.no = num_classes + 4 * reg_max  # outputs per anchor point

        cls_ch = max(in_channels, num_classes)
        box_ch = max(16, in_channels // 4, 4 * reg_max)

        self.cls_branches = nn.ModuleList()
        self.box_branches = nn.ModuleList()
        for _ in range(num_levels):
            self.cls_branches.append(nn.Sequential(
                ConvBNAct(in_channels, cls_ch, k=3),
                ConvBNAct(cls_ch, cls_ch, k=3),
                nn.Conv2d(cls_ch, num_classes, kernel_size=1),
            ))
            self.box_branches.append(nn.Sequential(
                ConvBNAct(in_channels, box_ch, k=3),
                ConvBNAct(box_ch, box_ch, k=3),
                nn.Conv2d(box_ch, 4 * reg_max, kernel_size=1),
            ))

        # YOLOv8 init: bias the cls output so initial sigmoid ≈ 0.01 per class
        # (prevents loss explosion on first iter from random logits).
        self._bias_init()

    def _bias_init(self):
        for cls_seq, stride in zip(self.cls_branches, self.strides):
            cls_conv = cls_seq[-1]
            cls_conv.bias.data.fill_(-4.0)  # sigmoid(-4) ≈ 0.018
        for box_seq, stride in zip(self.box_branches, self.strides):
            box_conv = box_seq[-1]
            # init to favor middle bin → distance ≈ reg_max/2 (no preference)
            box_conv.bias.data.fill_(0.0)

    def forward(self, feats):
        """feats: list of [B, in_channels, Hi, Wi] from FPN (P3, P4, P5).
        Returns:
          cls_logits: list of [B, num_classes,  Hi, Wi]
          box_logits: list of [B, 4*reg_max,    Hi, Wi]
        """
        assert len(feats) == self.num_levels
        cls_logits = [self.cls_branches[i](feats[i]) for i in range(self.num_levels)]
        box_logits = [self.box_branches[i](feats[i]) for i in range(self.num_levels)]
        return cls_logits, box_logits


if __name__ == "__main__":
    head = YOLOv8Head(in_channels=256, num_classes=80, reg_max=16, num_levels=3)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"YOLOv8Head params: {n_params:,}  ({n_params * 4 / 1024**2:.2f} MB fp32)")

    # Smoke test: feed dummy FPN outputs at P3/P4/P5 strides
    feats = [
        torch.randn(2, 256, 52, 52),  # P3 stride 8 (input 416/8)
        torch.randn(2, 256, 26, 26),  # P4 stride 16
        torch.randn(2, 256, 13, 13),  # P5 stride 32
    ]
    cls_logits, box_logits = head(feats)
    for i, (c, b) in enumerate(zip(cls_logits, box_logits)):
        print(f"  level {i}: cls {tuple(c.shape)}  box {tuple(b.shape)}")

    total_anchors = sum(c.shape[2] * c.shape[3] for c in cls_logits)
    print(f"  total anchor points: {total_anchors}")
