#!/usr/bin/env python3
"""UC Merced evaluation with pooled features + MLP classification head.

This script evaluates foundation checkpoints by:
1) Extracting pooled backbone features (global-average-pooled inside DynamicVis).
2) Training a classification head (linear or MLP) on train split features.
3) Reporting classification metrics on validation split or stratified k-fold.

The backbone is kept frozen; only the classification head is optimized.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CBIR_ROOT = REPO_ROOT / "eval" / "cbir"
DYNAMICVIS_ROOT = REPO_ROOT / "architectures" / "DynamicVis"

if str(CBIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CBIR_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if DYNAMICVIS_ROOT.is_dir() and str(DYNAMICVIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICVIS_ROOT))


def _load_cbir_model_factory():
    models_path = CBIR_ROOT / "models.py"
    if not models_path.is_file():
        raise FileNotFoundError(f"CBIR models file not found: {models_path}")

    spec = importlib.util.spec_from_file_location("cbir_models", str(models_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {models_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_model


create_model = _load_cbir_model_factory()

SUPPORTED_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png")

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "eval" / "UCMerced_LandUse"
DEFAULT_IMAGES_SUBDIR = "Images"
DEFAULT_TRAIN_LIST = REPO_ROOT / "architectures" / "DynamicVis" / "datainfo" / "ucmerced" / "train.txt"
DEFAULT_VAL_LIST = REPO_ROOT / "architectures" / "DynamicVis" / "datainfo" / "ucmerced" / "val.txt"
DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "architectures"
    / "DynamicVis"
    / "configs_DynamicVis"
    / "UCMerced"
    / "dynamicvis_b_uc_mamba.py"
)
DEFAULT_MODEL_PATH = REPO_ROOT / "outputs" / "bovw_training_8262" / "epoch_20.pth"


class UCMercedDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int, in_chans: int) -> None:
        self.items = items
        self.img_size = img_size
        self.in_chans = in_chans

        if self.in_chans not in (3, 4):
            raise ValueError("in_chans must be 3 or 4.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]

        image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0

        if self.in_chans == 4:
            nir = np.zeros((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, nir], axis=-1)

        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), label


class ClassificationHead(nn.Module):
    """Classification head over pooled backbone features.

    If hidden_dim <= 0, this becomes a linear classifier.
    Otherwise: Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_split_file(
    images_root: Path,
    split_file: Path,
    class_to_idx: Dict[str, int],
) -> List[Tuple[str, int]]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    items: List[Tuple[str, int]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            rel_path = line.split()[0].replace("\\", "/")
            rel = Path(rel_path)
            if len(rel.parts) < 2:
                raise ValueError(f"Invalid split line (expected class/file): {line}")

            class_name = rel.parts[0]
            if class_name not in class_to_idx:
                raise ValueError(f"Class '{class_name}' from split file not found under {images_root}")

            abs_path = images_root / rel
            if not abs_path.is_file():
                raise FileNotFoundError(f"Image from split file not found: {abs_path}")

            items.append((str(abs_path), class_to_idx[class_name]))

    if not items:
        raise RuntimeError(f"No valid items found in split file: {split_file}")

    return items


def discover_classes(images_root: Path) -> Dict[str, int]:
    class_names = sorted(
        [d.name for d in images_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    if not class_names:
        raise RuntimeError(f"No class folders found under: {images_root}")

    return {name: idx for idx, name in enumerate(class_names)}


def load_all_items(images_root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []

    for class_name in sorted(class_to_idx.keys()):
        class_dir = images_root / class_name
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(SUPPORTED_EXTS):
                items.append((str(class_dir / file_name), class_to_idx[class_name]))

    if not items:
        raise RuntimeError(f"No images found under: {images_root}")

    return items


def build_stratified_kfold_indices(
    labels: np.ndarray,
    num_folds: int,
    seed: int,
) -> List[np.ndarray]:
    if num_folds < 2:
        raise ValueError("--num-folds must be >= 2")

    rng = random.Random(seed)
    grouped: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        grouped.setdefault(int(label), []).append(idx)

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for class_indices in grouped.values():
        rng.shuffle(class_indices)
        for pos, sample_idx in enumerate(class_indices):
            folds[pos % num_folds].append(sample_idx)

    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]


def maybe_subsample_indices(indices: np.ndarray, max_items: int | None, seed: int) -> np.ndarray:
    if max_items is None or len(indices) <= max_items:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_items, replace=False)
    return np.sort(sampled.astype(np.int64))


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []

    for batch_images, batch_labels in tqdm(loader, desc="Embedding", leave=False):
        batch_images = batch_images.to(device, non_blocking=True)
        embeddings = model(batch_images)

        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        elif isinstance(embeddings, dict):
            embeddings = embeddings.get("embeddings", next(iter(embeddings.values())))

        all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))
        all_labels.extend(batch_labels.numpy().tolist())

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.vstack(all_embeddings), np.array(all_labels, dtype=np.int64)


def standardize_features(
    train_features: np.ndarray,
    eval_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    train_norm = (train_features - mean) / std
    eval_norm = (eval_features - mean) / std
    return train_norm.astype(np.float32), eval_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def train_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> ClassificationHead:
    in_dim = int(train_features.shape[1])
    head = ClassificationHead(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.head_dropout,
    ).to(device)

    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels),
    )

    batch_size = min(args.head_batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.head_lr,
        weight_decay=args.head_weight_decay,
    )

    for epoch in range(args.head_epochs):
        head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = head(feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=1) == labels).sum().item())
            running_total += int(labels.size(0))

        if running_total == 0:
            raise RuntimeError("No training samples available for head training.")

        if (
            epoch == 0
            or (epoch + 1) == args.head_epochs
            or ((epoch + 1) % args.head_log_interval == 0)
        ):
            avg_loss = running_loss / running_total
            avg_acc = 100.0 * running_correct / running_total
            print(
                f"  Head epoch {epoch + 1:03d}/{args.head_epochs:03d} | "
                f"loss={avg_loss:.4f} | acc={avg_acc:.2f}%"
            )

    return head


@torch.no_grad()
def predict_logits(
    head: ClassificationHead,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, 0), dtype=np.float32)

    head.eval()
    outputs: List[np.ndarray] = []
    step = max(1, batch_size)

    for start in range(0, len(features), step):
        end = min(start + step, len(features))
        feats = torch.from_numpy(features[start:end]).to(device, non_blocking=True)
        logits = head(feats)
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))

    return np.vstack(outputs)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        top_k_accuracy_score,
    )

    num_classes = int(logits.shape[1])
    all_class_ids = list(range(num_classes))

    metrics: Dict[str, float] = {}
    metrics["top1_accuracy"] = float(accuracy_score(labels, preds) * 100.0)

    k = min(5, num_classes)
    if k > 1:
        metrics["top5_accuracy"] = float(
            top_k_accuracy_score(labels, logits, k=k, labels=all_class_ids) * 100.0
        )

    metrics["precision_macro"] = float(
        precision_score(labels, preds, average="macro", zero_division=0) * 100.0
    )
    metrics["recall_macro"] = float(
        recall_score(labels, preds, average="macro", zero_division=0) * 100.0
    )
    metrics["f1_macro"] = float(f1_score(labels, preds, average="macro", zero_division=0) * 100.0)

    return metrics


def infer_img_size_from_config(config_path: str, fallback: int = 512) -> int:
    try:
        mmengine = importlib.import_module("mmengine")
        Config = getattr(mmengine, "Config")

        cfg = Config.fromfile(config_path)
        cfg_img_size = cfg.get("img_size", None)
        if cfg_img_size is None:
            model_cfg = cfg.get("model", None)
            if isinstance(model_cfg, dict):
                backbone_cfg = model_cfg.get("backbone", None)
                if isinstance(backbone_cfg, dict):
                    cfg_img_size = backbone_cfg.get("img_size", None)

        if cfg_img_size is None:
            return fallback

        return int(cfg_img_size)
    except Exception as exc:
        print(
            f"Warning: failed to infer img_size from config {config_path}: {exc}. "
            f"Falling back to {fallback}."
        )
        return fallback


def make_loader(
    items: List[Tuple[str, int]],
    img_size: int,
    in_chans: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = UCMercedDataset(items=items, img_size=img_size, in_chans=in_chans)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"{'Top-1 Accuracy':<24}: {metrics['top1_accuracy']:.2f}%")
    if "top5_accuracy" in metrics:
        print(f"{'Top-5 Accuracy':<24}: {metrics['top5_accuracy']:.2f}%")
    print(f"{'Precision (macro)':<24}: {metrics['precision_macro']:.2f}%")
    print(f"{'Recall (macro)':<24}: {metrics['recall_macro']:.2f}%")
    print(f"{'F1 (macro)':<24}: {metrics['f1_macro']:.2f}%")


def maybe_save_head(
    head: ClassificationHead,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    class_to_idx: Dict[str, int],
    split_tag: str,
    args: argparse.Namespace,
) -> None:
    if not args.save_head_dir:
        return

    save_dir = Path(args.save_head_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_tag = Path(args.model_path).stem
    hidden_tag = args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else "linear"
    save_path = save_dir / f"ucmerced_{model_tag}_{split_tag}_head_{hidden_tag}.pth"

    payload = {
        "head_state_dict": head.state_dict(),
        "class_to_idx": class_to_idx,
        "model_path": args.model_path,
        "config_path": args.config_path,
        "img_size": args.img_size,
        "embedding_dim": args.embedding_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "head_dropout": args.head_dropout,
        "standardize_features": args.standardize_features,
    }

    if mean is not None and std is not None:
        payload["feature_mean"] = mean
        payload["feature_std"] = std

    torch.save(payload, save_path)
    print(f"Saved trained head: {save_path}")


def evaluate_fixed_split(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    class_to_idx: Dict[str, int],
    images_root: Path,
) -> None:
    train_items = parse_split_file(images_root, Path(args.train_list), class_to_idx)
    val_items = parse_split_file(images_root, Path(args.val_list), class_to_idx)

    train_indices = np.arange(len(train_items), dtype=np.int64)
    val_indices = np.arange(len(val_items), dtype=np.int64)

    train_indices = maybe_subsample_indices(train_indices, args.max_train, args.seed + 101)
    val_indices = maybe_subsample_indices(val_indices, args.max_test, args.seed + 202)

    train_items = [train_items[i] for i in train_indices]
    val_items = [val_items[i] for i in val_indices]

    if not train_items or not val_items:
        raise RuntimeError("Empty train/val split after applying max_train/max_test.")

    print(f"Classes: {len(class_to_idx)}")
    print(f"Train split size: {len(train_items)}")
    print(f"Val split size:   {len(val_items)}")

    if args.dry_run:
        return

    train_loader = make_loader(
        items=train_items,
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        items=val_items,
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_features, train_labels = extract_embeddings(model, train_loader, device)
    val_features, val_labels = extract_embeddings(model, val_loader, device)

    if train_features.size == 0 or val_features.size == 0:
        raise RuntimeError("Failed to extract embeddings for fixed split evaluation.")

    feature_mean = None
    feature_std = None
    if args.standardize_features:
        train_features, val_features, feature_mean, feature_std = standardize_features(
            train_features,
            val_features,
        )

    head = train_head(
        train_features=train_features,
        train_labels=train_labels,
        num_classes=len(class_to_idx),
        args=args,
        device=device,
    )

    logits = predict_logits(
        head=head,
        features=val_features,
        device=device,
        batch_size=args.head_batch_size,
    )
    preds = logits.argmax(axis=1)
    metrics = compute_metrics(val_labels, preds, logits)

    print_metric_block("UC Merced Evaluation (Pooled Features + MLP Head)", metrics)
    maybe_save_head(head, feature_mean, feature_std, class_to_idx, "fixed", args)


def evaluate_kfold(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    class_to_idx: Dict[str, int],
    images_root: Path,
) -> None:
    all_items = load_all_items(images_root, class_to_idx)
    all_labels = np.array([label for _, label in all_items], dtype=np.int64)
    folds = build_stratified_kfold_indices(all_labels, args.num_folds, args.seed)

    print(f"Classes: {len(class_to_idx)}")
    print(f"Total items: {len(all_items)}")
    print(f"Folds: {args.num_folds}")

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(all_items), dtype=np.int64), test_idx, assume_unique=False)
        train_idx = maybe_subsample_indices(train_idx, args.max_train, args.seed + 1000 + fold_idx)
        test_idx = maybe_subsample_indices(test_idx, args.max_test, args.seed + 2000 + fold_idx)
        print(
            f"  Fold {fold_idx + 1}/{args.num_folds}: "
            f"train={len(train_idx)} | test={len(test_idx)}"
        )

    if args.dry_run:
        return

    all_loader = make_loader(
        items=all_items,
        img_size=args.img_size,
        in_chans=args.in_chans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    all_features, all_labels = extract_embeddings(model, all_loader, device)

    if all_features.size == 0:
        raise RuntimeError("Failed to extract embeddings for k-fold evaluation.")

    metric_names = ["top1_accuracy", "top5_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    fold_metrics: Dict[str, List[float]] = {name: [] for name in metric_names}

    for fold_idx, test_idx_raw in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(all_items), dtype=np.int64), test_idx_raw, assume_unique=False)
        train_idx = maybe_subsample_indices(train_idx, args.max_train, args.seed + 1000 + fold_idx)
        test_idx = maybe_subsample_indices(test_idx_raw, args.max_test, args.seed + 2000 + fold_idx)

        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(
                f"Fold {fold_idx + 1}/{args.num_folds} has empty train/test split after subsampling."
            )

        train_features = all_features[train_idx]
        train_labels = all_labels[train_idx]
        test_features = all_features[test_idx]
        test_labels = all_labels[test_idx]

        feature_mean = None
        feature_std = None
        if args.standardize_features:
            train_features, test_features, feature_mean, feature_std = standardize_features(
                train_features,
                test_features,
            )

        print(f"\nTraining head for fold {fold_idx + 1}/{args.num_folds}...")
        head = train_head(
            train_features=train_features,
            train_labels=train_labels,
            num_classes=len(class_to_idx),
            args=args,
            device=device,
        )

        logits = predict_logits(
            head=head,
            features=test_features,
            device=device,
            batch_size=args.head_batch_size,
        )
        preds = logits.argmax(axis=1)
        metrics = compute_metrics(test_labels, preds, logits)

        for name in metric_names:
            if name in metrics:
                fold_metrics[name].append(metrics[name])

        print(f"\nFold {fold_idx + 1}/{args.num_folds}")
        print_metric_block("Fold Metrics", metrics)

        maybe_save_head(head, feature_mean, feature_std, class_to_idx, f"kfold{fold_idx + 1}", args)

    print("\n" + "=" * 64)
    print("UC Merced Evaluation (Stratified K-Fold)")
    print("=" * 64)
    for name in metric_names:
        values = fold_metrics[name]
        if not values:
            continue
        vals = np.array(values, dtype=np.float32)
        print(f"{name:<20}: {vals.mean():.2f} +/- {vals.std(ddof=0):.2f}")


def resolve_images_root(data_dir: str, images_subdir: str) -> Path:
    data_root = Path(data_dir)
    candidate = data_root / images_subdir

    if candidate.is_dir():
        return candidate
    if data_root.is_dir():
        return data_root

    raise FileNotFoundError(
        f"Could not find dataset directory at {data_root} or image subdir {candidate}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UC Merced evaluation with pooled features + MLP head")

    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--images_subdir", type=str, default=DEFAULT_IMAGES_SUBDIR)

    parser.add_argument("--split_mode", type=str, choices=["fixed", "kfold"], default="fixed")
    parser.add_argument("--train_list", type=str, default=str(DEFAULT_TRAIN_LIST))
    parser.add_argument("--val_list", type=str, default=str(DEFAULT_VAL_LIST))
    parser.add_argument("--num_folds", type=int, default=5)

    parser.add_argument(
        "--model_type",
        type=str,
        default="dynamicvis",
        choices=["dynamicvis"],
    )
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)

    parser.add_argument("--head_epochs", type=int, default=30)
    parser.add_argument("--head_batch_size", type=int, default=256)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--head_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_log_interval", type=int, default=5)
    parser.add_argument("--save_head_dir", type=str, default="")

    parser.add_argument(
        "--standardize_features",
        dest="standardize_features",
        action="store_true",
        help="Apply z-score normalization to features before training the head.",
    )
    parser.add_argument(
        "--no_standardize_features",
        dest="standardize_features",
        action="store_false",
        help="Disable z-score normalization before head training.",
    )
    parser.set_defaults(standardize_features=True)

    parser.add_argument("--dry_run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_type == "dynamicvis":
        args.in_chans = 3
        if not Path(args.config_path).is_file():
            raise FileNotFoundError(f"DynamicVis config not found: {args.config_path}")

    if not Path(args.model_path).is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if args.head_epochs < 1:
        raise ValueError("--head_epochs must be >= 1")
    if args.head_batch_size < 1:
        raise ValueError("--head_batch_size must be >= 1")
    if args.head_lr <= 0:
        raise ValueError("--head_lr must be > 0")

    images_root = resolve_images_root(args.data_dir, args.images_subdir)
    class_to_idx = discover_classes(images_root)

    if args.img_size is None:
        args.img_size = infer_img_size_from_config(args.config_path, fallback=512)

    print("=" * 64)
    print("UC Merced DynamicVis Evaluation")
    print("=" * 64)
    print(f"Data directory:      {args.data_dir}")
    print(f"Images root:         {images_root}")
    print(f"Split mode:          {args.split_mode}")
    print(f"Checkpoint:          {args.model_path}")
    print(f"Config:              {args.config_path}")
    print(f"Image size:          {args.img_size}")
    print(f"Embedding dim:       {args.embedding_dim}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Head epochs:         {args.head_epochs}")
    print(f"Head batch size:     {args.head_batch_size}")
    print(f"Head lr:             {args.head_lr}")
    print(f"Head weight decay:   {args.head_weight_decay}")
    print(f"MLP hidden dim:      {args.mlp_hidden_dim}")
    print(f"Head dropout:        {args.head_dropout}")
    print(f"Standardize feats:   {args.standardize_features}")
    print(f"Seed:                {args.seed}")
    print("=" * 64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # create_model returns a frozen foundation encoder that already outputs
    # pooled feature vectors (DynamicVis out_type='avg_featmap').
    model = create_model(
        model_type=args.model_type,
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        device=device,
        config_path=args.config_path,
        use_multi_scale=False,
        layer_indices=None,
        img_size=args.img_size,
        in_chans=args.in_chans,
    )

    if args.split_mode == "fixed":
        evaluate_fixed_split(
            args=args,
            model=model,
            device=device,
            class_to_idx=class_to_idx,
            images_root=images_root,
        )
    else:
        evaluate_kfold(
            args=args,
            model=model,
            device=device,
            class_to_idx=class_to_idx,
            images_root=images_root,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
