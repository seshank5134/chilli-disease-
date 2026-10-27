"""Train a lightweight chilli-leaf classifier with source-family-safe splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRANSFORM_SUFFIX = re.compile(r"^(.*)_(?:bright|flipped|rotated|sheared|shifted|zoomed)_\d+$", re.I)
ARCHITECTURE = "mobilenet_v3_small"
IMAGE_SIZE = 224


def group_stem(path: Path) -> str:
    match = TRANSFORM_SUFFIX.match(path.stem)
    return match.group(1) if match else path.stem


def collect(root: Path, source: str) -> list[dict[str, str]]:
    items = []
    for class_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        for path in sorted(class_dir.rglob("*"), key=lambda p: p.as_posix().casefold()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                items.append({"path": str(path), "relative_path": path.relative_to(root).as_posix(), "label": class_dir.name, "group": group_stem(path), "source": source, "sha256": digest})
    return items


def split_groups(items: list[dict[str, str]], seed: int, val_fraction: float, test_fraction: float):
    by_label = defaultdict(set)
    for item in items:
        by_label[item["label"]].add(item["group"])
    partitions = {}
    rng = random.Random(seed)
    for label, groups in sorted(by_label.items(), key=lambda kv: kv[0].casefold()):
        ordered = sorted(groups, key=str.casefold)
        rng.shuffle(ordered)
        n = len(ordered)
        n_test = max(1, round(n * test_fraction)) if n >= 3 else 0
        n_val = max(1, round(n * val_fraction)) if n - n_test >= 2 else 0
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
        for i, group in enumerate(ordered):
            partitions[(label, group)] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    # If byte-identical images have different filename stems, link their groups so
    # an exact copy cannot cross the train/validation/test boundary.
    parent = {key: key for key in partitions}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    first_group_for_hash = {}
    for item in items:
        group_key = (item["label"], item["group"])
        digest = item["sha256"]
        if digest in first_group_for_hash:
            left, right = find(group_key), find(first_group_for_hash[digest])
            if left != right:
                parent[right] = left
        else:
            first_group_for_hash[digest] = group_key
    priority = {"train": 0, "validation": 1, "test": 2}
    component_partition = {}
    for key, part in partitions.items():
        component = find(key)
        current = component_partition.get(component)
        if current is None or priority[part] > priority[current]:
            component_partition[component] = part
    return {key: component_partition[find(key)] for key in partitions}


class LeafDataset(Dataset):
    def __init__(self, items, class_to_idx, training=False):
        self.items = items
        self.class_to_idx = class_to_idx
        base = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if training:
            base += [transforms.RandomHorizontalFlip(p=0.5)]
        base += [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item["path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), self.class_to_idx[item["label"]]


def make_model(num_classes: int, pretrained: bool):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    confusion = None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += int((preds == labels).sum())
        total += labels.numel()
        if confusion is None:
            n = model.classifier[-1].out_features
            confusion = torch.zeros((n, n), dtype=torch.int64)
        for actual, predicted in zip(labels.cpu(), preds.cpu()):
            confusion[actual, predicted] += 1
    return (correct / total if total else 0.0), (confusion.tolist() if confusion is not None else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--augmented-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use pretrained ImageNet weights")
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.test_fraction <= 0 or args.val_fraction + args.test_fraction >= 1:
        parser.error("Validation and test fractions must be positive and sum to less than 1.")

    original_root, augmented_root = args.original_root.resolve(), args.augmented_root.resolve()
    originals = collect(original_root, "original")
    augmented = collect(augmented_root, "augmented")
    if not originals or not augmented:
        parser.error("Both roots must contain class subdirectories with supported images.")
    original_labels = {i["label"] for i in originals}
    augmented_labels = {i["label"] for i in augmented}
    if original_labels != augmented_labels:
        parser.error(f"Class folder labels differ. Original-only={sorted(original_labels-augmented_labels)}; augmented-only={sorted(augmented_labels-original_labels)}")
    classes = sorted(original_labels, key=str.casefold)
    items = originals + augmented
    labels_by_hash = defaultdict(set)
    for item in items:
        labels_by_hash[item["sha256"]].add(item["label"])
    ambiguous_hashes = {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}
    excluded_items = [item for item in items if item["sha256"] in ambiguous_hashes]
    candidates = [item for item in items if item["sha256"] not in ambiguous_hashes]
    partitions = split_groups(candidates, args.seed, args.val_fraction, args.test_fraction)
    # Validation and test use originals only. All related variants stay out of training for held-out groups.
    train_items = [i for i in candidates if partitions[(i["label"], i["group"])] == "train"]
    val_items = [i for i in candidates if i["source"] == "original" and partitions[(i["label"], i["group"])] == "validation"]
    test_items = [i for i in candidates if i["source"] == "original" and partitions[(i["label"], i["group"])] == "test"]
    if not train_items or not val_items or not test_items:
        parser.error("A split is empty. Check that each class has at least three source groups.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    with (args.reports / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["partition", "label", "source", "relative_path", "source_group"])
        w.writeheader()
        for part, collection in (("train", train_items), ("validation", val_items), ("test", test_items)):
            for item in collection:
                w.writerow({"partition": part, "label": item["label"], "source": item["source"], "relative_path": item["relative_path"], "source_group": item["group"]})
        for item in excluded_items:
            w.writerow({"partition": "excluded_conflicting_exact_duplicate", "label": item["label"], "source": item["source"], "relative_path": item["relative_path"], "source_group": item["group"]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Images: train={len(train_items)}, validation originals={len(val_items)}, test originals={len(test_items)}")
    class_to_idx = {name: i for i, name in enumerate(classes)}
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(LeafDataset(train_items, class_to_idx, True), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(LeafDataset(val_items, class_to_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(LeafDataset(test_items, class_to_idx), shuffle=False, **loader_kwargs)
    model = make_model(len(classes), not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)
        val_accuracy, _ = evaluate(model, val_loader, device)
        avg_loss = loss_sum / max(count, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, "validation_accuracy": val_accuracy})
        print(f"Epoch {epoch:02d}/{args.epochs}: loss={avg_loss:.4f}, validation accuracy={val_accuracy:.4f}")
        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save({"architecture": ARCHITECTURE, "state_dict": model.cpu().state_dict(), "class_names": classes, "image_size": IMAGE_SIZE, "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225]}, args.output)
            model.to(device)

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_accuracy, confusion = evaluate(model, test_loader, device)
    metrics = {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "architecture": ARCHITECTURE, "class_names": classes, "train_image_count": len(train_items), "validation_original_count": len(val_items), "test_original_count": len(test_items), "excluded_conflicting_exact_duplicate_count": len(excluded_items), "excluded_conflicting_exact_duplicate_hash_count": len(ambiguous_hashes), "best_validation_accuracy": best_val, "test_original_accuracy": test_accuracy, "confusion_matrix_labels_order": classes, "test_confusion_matrix": confusion, "history": history, "split_strategy": "group by pre-transform filename stem and exact SHA-256; validation and test use original images only; conflicting-label exact duplicates are excluded"}
    (args.reports / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint to {args.output}; original-only test accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
