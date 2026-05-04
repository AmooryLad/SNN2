"""Pascal VOC-analogous COCO 2017 detection loader.

Uses torchvision.datasets.CocoDetection + pycocotools. Returns the same
(image_tensor, target_dict) shape as our VOC adapter so train.py can stay
simple. COCO's 80 classes have discontiguous category IDs (1..90 with gaps) —
we build a contiguous 0..80 mapping (0 = background).

Expected layout at `root`:
    root/
      annotations/
        instances_train2017.json
        instances_val2017.json
      train2017/*.jpg
      val2017/*.jpg
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2


# 80 COCO classes in alphabetical order of their official names — this ordering
# is arbitrary; what matters is the consistent mapping inside this module.
# We build the mapping dynamically from the loaded annotations at init time.


class CocoDetectionAdapter(torch.utils.data.Dataset):
    """Wrap torchvision CocoDetection, converting to RetinaNet target dicts."""

    def __init__(self, root, image_set="train", transforms=None, year="2017"):
        root = Path(root)
        ann_file = root / "annotations" / f"instances_{image_set}{year}.json"
        img_dir = root / f"{image_set}{year}"
        if not ann_file.exists() or not img_dir.exists():
            raise FileNotFoundError(
                f"Expected {ann_file} and {img_dir} — run COCO download first."
            )

        self.coco_ds = CocoDetection(
            root=str(img_dir), annFile=str(ann_file),
        )
        self.transforms = transforms

        # Build mapping COCO-cat-id → contiguous 1..80 (0 reserved for bg).
        cats = self.coco_ds.coco.loadCats(self.coco_ds.coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        self.cat_id_to_idx = {c["id"]: i + 1 for i, c in enumerate(cats_sorted)}
        self.class_names = ["__background__"] + [c["name"] for c in cats_sorted]

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img, coco_anns = self.coco_ds[idx]
        W, H = img.size

        boxes, labels = [], []
        for ann in coco_anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        if len(boxes) == 0:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY", canvas_size=(H, W),
            )
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = tv_tensors.BoundingBoxes(
                torch.tensor(boxes, dtype=torch.float32),
                format="XYXY", canvas_size=(H, W),
            )
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["boxes"] = target["boxes"].as_subclass(torch.Tensor).float()
        return img, target


def build_transforms(img_size=416, train=True):
    tfms = []
    if train:
        tfms.append(v2.RandomHorizontalFlip(p=0.5))
        tfms.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
    tfms.append(v2.Resize((img_size, img_size), antialias=True))
    tfms.append(v2.ToImage())
    tfms.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(tfms)


def detection_collate(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def build_dataloaders(
    root,
    img_size=416,
    batch_size=16,
    num_workers=8,
    aug_level="basic",     # "basic" | "strong" (strong = + Mosaic wrapper)
    mosaic_prob=0.5,
):
    train_set = CocoDetectionAdapter(
        root, image_set="train",
        transforms=build_transforms(img_size, train=True),
    )
    if aug_level == "strong":
        from cifar100_spikedetect.augs import MosaicDetection
        train_set = MosaicDetection(train_set, img_size=img_size, prob=mosaic_prob)
        print(f"Wrapped train_set with MosaicDetection (prob={mosaic_prob})")
    val_set = CocoDetectionAdapter(
        root, image_set="val",
        transforms=build_transforms(img_size, train=False),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=detection_collate, persistent_workers=num_workers > 0,
        prefetch_factor=6 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=max(2, num_workers // 2), pin_memory=True,
        collate_fn=detection_collate, persistent_workers=num_workers > 0,
        prefetch_factor=6 if num_workers > 0 else None,
    )
    # val_set is always CocoDetectionAdapter (not wrapped), has class_names
    return train_loader, val_loader, val_set.class_names


if __name__ == "__main__":
    train_loader, val_loader, class_names = build_dataloaders(
        root=PROJECT_ROOT / "data" / "coco",
        img_size=416, batch_size=4, num_workers=0,
    )
    print(f"classes: {len(class_names)}  (first 10: {class_names[:10]})")
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")
    imgs, targets = next(iter(train_loader))
    print(f"img[0] {imgs[0].shape}  target[0] boxes {tuple(targets[0]['boxes'].shape)}  labels {targets[0]['labels'].tolist()[:5]}")
