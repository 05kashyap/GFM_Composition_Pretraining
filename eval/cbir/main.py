#!/usr/bin/env python3
"""Stage-1 retrieval evaluation for AID and ForestNet datasets."""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CBIR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(CBIR_ROOT) not in sys.path:
    sys.path.insert(0, str(CBIR_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from indexer import build_faiss_index, load_model


def _ensure_dynamicvis_on_path() -> None:
    dynamicvis_root = os.environ.get("DYNAMICVIS_ROOT", "")
    if dynamicvis_root:
        root = Path(dynamicvis_root).expanduser()
    else:
        root = REPO_ROOT / "architectures" / "DynamicVis"
    if not root.exists():
        raise FileNotFoundError(
            f"DynamicVis repo not found at: {root}. "
            "Set DYNAMICVIS_ROOT or clone into architectures/DynamicVis."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


FORESTNET_FINE_LABELS: List[str] = [
    "Oil palm plantation",
    "Timber plantation",
    "Other large-scale plantations",
    "Grassland shrubland",
    "Small-scale agriculture",
    "Small-scale mixed plantation",
    "Small-scale oil palm plantation",
    "Mining",
    "Fish pond",
    "Logging",
    "Secondary forest",
    "Other",
]

FORESTNET_MERGED_LABELS: List[str] = [
    "Plantation",
    "Grassland shrubland",
    "Smallholder agriculture",
    "Other",
]


class AIDDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int, in_chans: int) -> None:
        self.items = items
        self.img_size = img_size
        self.in_chans = in_chans

        if self.in_chans not in (3, 4):
            raise ValueError("in_chans must be 3 or 4 for AID RGB images.")

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


class ForestNetDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], img_size: int, in_chans: int) -> None:
        self.items = items
        self.img_size = img_size
        self.in_chans = in_chans

        if self.in_chans not in (3, 4):
            raise ValueError("in_chans must be 3 or 4 for ForestNet visible composites.")

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


def load_aid_items(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"AID data directory not found: {data_dir}")

    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    items: List[Tuple[str, int]] = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                items.append((os.path.join(class_dir, file_name), class_to_idx[class_name]))

    if not items:
        raise RuntimeError(f"No images found under: {data_dir}")

    return items, idx_to_class


def _get_forestnet_label_spec(mode: str) -> Tuple[str, List[str]]:
    if mode == "12":
        return "label", FORESTNET_FINE_LABELS
    if mode == "4":
        return "merged_label", FORESTNET_MERGED_LABELS
    raise ValueError(f"Unsupported ForestNet label mode: {mode}")


def load_forestnet_items(
    data_dir: str,
    split: str,
    mode: str,
) -> Tuple[List[Tuple[str, int]], Dict[int, str]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"ForestNet directory not found: {data_dir}")

    label_column, label_names = _get_forestnet_label_spec(mode)
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    idx_to_class = {idx: name for name, idx in label_to_idx.items()}

    csv_path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"ForestNet split CSV not found: {csv_path}")

    items: List[Tuple[str, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_name = row.get(label_column, "")
            if label_name not in label_to_idx:
                continue

            example_path = row.get("example_path", "")
            if not example_path:
                continue

            image_path = os.path.join(data_dir, example_path, "images", "visible", "composite.png")
            if not os.path.isfile(image_path):
                continue

            items.append((image_path, label_to_idx[label_name]))

    if not items:
        raise RuntimeError(f"No ForestNet images found for split={split} mode={mode} under {data_dir}")

    return items, idx_to_class


def limit_items_per_class(
    items: List[Tuple[str, int]],
    max_per_class: int | None,
    seed: int,
) -> List[Tuple[str, int]]:
    if max_per_class is None:
        return items

    rng = random.Random(seed)
    grouped: Dict[int, List[Tuple[str, int]]] = {}
    for path, label in items:
        grouped.setdefault(label, []).append((path, label))

    limited: List[Tuple[str, int]] = []
    for class_items in grouped.values():
        rng.shuffle(class_items)
        limited.extend(class_items[:max_per_class])

    return limited


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

    fold_arrays: List[np.ndarray] = []
    for fold in folds:
        fold_arrays.append(np.array(sorted(fold), dtype=np.int64))
    return fold_arrays


def maybe_subsample_indices(
    indices: np.ndarray,
    max_items: int | None,
    seed: int,
) -> np.ndarray:
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

    for batch_images, batch_labels in tqdm(loader, desc="Embedding"):
        batch_images = batch_images.to(device)
        embeddings = model(batch_images)
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        elif isinstance(embeddings, dict):
            embeddings = embeddings.get("embeddings", next(iter(embeddings.values())))
        all_embeddings.append(embeddings.detach().cpu().numpy())
        all_labels.extend(batch_labels.numpy().tolist())

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.vstack(all_embeddings).astype(np.float32), np.array(all_labels, dtype=np.int64)


def compute_metrics(
    index,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[Dict[int, float], Dict[int, float]]:
    max_k = max(ks)
    distances, indices = index.search(query_embeddings.astype(np.float32), max_k)

    recall_at = {k: 0 for k in ks}
    map_at = {k: 0.0 for k in ks}

    for q_idx in range(len(query_labels)):
        q_label = query_labels[q_idx]
        retrieved_labels = train_labels[indices[q_idx]]

        for k in ks:
            if np.any(retrieved_labels[:k] == q_label):
                recall_at[k] += 1

        for k in ks:
            rel = (retrieved_labels[:k] == q_label).astype(np.int32)
            if rel.sum() == 0:
                ap = 0.0
            else:
                precisions = [rel[:i + 1].sum() / (i + 1) for i in range(k) if rel[i]]
                ap = float(np.sum(precisions) / rel.sum())
            map_at[k] += ap

    total = max(1, len(query_labels))
    recall_at = {k: recall_at[k] / total for k in ks}
    map_at = {k: map_at[k] / total for k in ks}
    return recall_at, map_at


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-1 retrieval evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="aid",
        choices=["aid", "forestnet"],
        help="Dataset to evaluate.",
    )
    parser.add_argument("--data_dir", type=str, default="data/eval/AID")
    parser.add_argument(
        "--forestnet_mode",
        type=str,
        default="both",
        choices=["both", "12", "4"],
        help="ForestNet evaluation mode: 12-class, 4-class, or both.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dynamicvis",
        choices=["prithvi", "prithvi2", "dynamicvis"],
    )
    parser.add_argument("--model_path", type=str, default="outputs/bovw_training_8262/epoch_20.pth")
    parser.add_argument(
        "--config_path",
        type=str,
        default="architectures/DynamicVis/configs_DynamicVis/AID/dynamicvis_b_aid_mamba.py",
    )
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--in_chans", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        dest="num_folds",
        help="Number of stratified folds for rotated evaluation (default: 5)",
    )
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--index_type", type=str, default="Flat", choices=["Flat", "IVF"])
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--index_dir", type=str, default="index")
    parser.add_argument("--save_index", action="store_true")
    parser.add_argument("--use_multi_scale", action="store_true")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model_type == "dynamicvis" and args.config_path is None:
        raise ValueError("--config_path is required when using --model_type dynamicvis")

    if args.model_type == "dynamicvis":
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"DynamicVis checkpoint not found: {args.model_path}")
        if not Path(args.config_path).exists():
            raise FileNotFoundError(f"DynamicVis config not found: {args.config_path}")

    if args.model_type in {"prithvi", "prithvi2", "prithvi_v2"}:
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"Prithvi v2 checkpoint not found: {args.model_path}")
        args.config_path = None

    if args.model_type == "dynamicvis":
        # DynamicVis is RGB-only; keep AIDDataset in 3-channel mode.
        args.in_chans = 3

        # If user kept the default img_size, align it to the DynamicVis config to avoid shape mismatches.
        if args.config_path is not None and args.img_size == 224:
            try:
                _ensure_dynamicvis_on_path()
                from mmengine import Config

                cfg = Config.fromfile(args.config_path)
                cfg_img_size = cfg.get("img_size", None)
                if cfg_img_size is None:
                    model_cfg = cfg.get("model", None)
                    if isinstance(model_cfg, dict):
                        backbone_cfg = model_cfg.get("backbone", None)
                        if isinstance(backbone_cfg, dict):
                            cfg_img_size = backbone_cfg.get("img_size", None)
                if cfg_img_size is not None:
                    args.img_size = int(cfg_img_size)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read DynamicVis config for img_size: {args.config_path}. "
                    f"Pass --img_size explicitly or fix the config. Root error: {e}"
                )

    if args.model_type in {"prithvi", "prithvi2", "prithvi_v2"}:
        args.in_chans = 3

    if args.model_type == "dynamicvis" and args.embedding_dim == 384:
        args.embedding_dim = 768
    elif args.model_type in {"prithvi", "prithvi2", "prithvi_v2"} and args.embedding_dim == 384:
        args.embedding_dim = 512

    if args.dataset == "aid":
        _run_aid(args)
    elif args.dataset == "forestnet":
        _run_forestnet(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


def _run_aid(args: argparse.Namespace) -> None:
    items, idx_to_class = load_aid_items(args.data_dir)
    items = limit_items_per_class(items, args.max_per_class, args.seed)

    if len(items) == 0:
        raise RuntimeError("No items available after applying --max_per_class.")

    all_labels = np.array([label for _, label in items], dtype=np.int64)
    folds = build_stratified_kfold_indices(all_labels, args.num_folds, args.seed)

    print(f"Classes: {len(idx_to_class)}")
    print(f"Total items: {len(items)}")
    print(f"Folds: {args.num_folds}")

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(items), dtype=np.int64), test_idx, assume_unique=False)
        train_idx = maybe_subsample_indices(train_idx, args.max_train, args.seed + 1000 + fold_idx)
        test_idx_limited = maybe_subsample_indices(test_idx, args.max_test, args.seed + 2000 + fold_idx)
        print(
            f"  Fold {fold_idx + 1}/{args.num_folds}: "
            f"train={len(train_idx)} | test={len(test_idx_limited)}"
        )

    if args.dry_run:
        return

    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(items), dtype=np.int64), test_idx, assume_unique=False)
        train_idx = maybe_subsample_indices(train_idx, args.max_train, args.seed + 1000 + fold_idx)
        test_idx_limited = maybe_subsample_indices(test_idx, args.max_test, args.seed + 2000 + fold_idx)
        if len(train_idx) == 0 or len(test_idx_limited) == 0:
            raise RuntimeError(
                f"Fold {fold_idx + 1}/{args.num_folds} has empty train/test split. "
                "Try lowering --num-folds or increasing samples."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_loader = DataLoader(
        AIDDataset(items, img_size=args.img_size, in_chans=args.in_chans),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        embedding_dim=args.embedding_dim,
        config_path=args.config_path,
        use_multi_scale=args.use_multi_scale,
        layer_indices=args.layer_indices,
        img_size=args.img_size,
        in_chans=args.in_chans,
    )

    all_embeddings, all_labels = extract_embeddings(model, all_loader, device)

    if all_embeddings.size == 0:
        raise RuntimeError("Failed to extract embeddings. Check inputs and model.")

    model_tag = Path(args.model_path).stem
    index_dir = Path(args.index_dir)
    ks = (1, 5, 10)

    import faiss

    fold_recalls: List[Dict[int, float]] = []
    fold_maps: List[Dict[int, float]] = []

    for fold_idx, fold_test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(items), dtype=np.int64), fold_test_idx, assume_unique=False)
        train_idx = maybe_subsample_indices(train_idx, args.max_train, args.seed + 1000 + fold_idx)
        test_idx = maybe_subsample_indices(fold_test_idx, args.max_test, args.seed + 2000 + fold_idx)

        train_embeddings = all_embeddings[train_idx]
        train_labels = all_labels[train_idx]
        test_embeddings = all_embeddings[test_idx]
        test_labels = all_labels[test_idx]

        index_path = index_dir / (
            f"aid_{model_tag}_k{args.num_folds}_fold{fold_idx + 1}_"
            f"ntrain{len(train_idx)}_ntest{len(test_idx)}.index"
        )

        if index_path.exists():
            print(f"[Fold {fold_idx + 1}/{args.num_folds}] Loading FAISS index: {index_path}")
            index = faiss.read_index(str(index_path))
        else:
            print(f"[Fold {fold_idx + 1}/{args.num_folds}] Building FAISS index...")
            index = build_faiss_index(train_embeddings, index_type=args.index_type, nlist=args.nlist)
            if args.save_index:
                index_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_path))
                print(f"[Fold {fold_idx + 1}/{args.num_folds}] Saved FAISS index: {index_path}")

        recall_at, map_at = compute_metrics(index, train_labels, test_embeddings, test_labels, ks=ks)
        fold_recalls.append(recall_at)
        fold_maps.append(map_at)

        print(f"[Fold {fold_idx + 1}/{args.num_folds}] Train={len(train_idx)} | Test={len(test_idx)}")
        for k in ks:
            print(
                f"[Fold {fold_idx + 1}/{args.num_folds}] "
                f"Recall@{k}: {recall_at[k]:.4f} | mAP@{k}: {map_at[k]:.4f}"
            )

    print("\nAID Retrieval Evaluation (Stage 1, Stratified K-Fold)")
    print(f"Model: {args.model_type} | {Path(args.model_path).name}")
    print(f"Folds: {args.num_folds}")

    for k in ks:
        recall_vals = np.array([m[k] for m in fold_recalls], dtype=np.float32)
        map_vals = np.array([m[k] for m in fold_maps], dtype=np.float32)
        print(
            f"Recall@{k}: {recall_vals.mean():.4f} ± {recall_vals.std(ddof=0):.4f} | "
            f"mAP@{k}: {map_vals.mean():.4f} ± {map_vals.std(ddof=0):.4f}"
        )


def _run_forestnet(args: argparse.Namespace) -> None:
    mode_specs: List[Tuple[str, str]] = []
    if args.forestnet_mode in ("12", "both"):
        mode_specs.append(("12", "ForestNet12"))
    if args.forestnet_mode in ("4", "both"):
        mode_specs.append(("4", "ForestNet4"))

    for mode, mode_name in mode_specs:
        val_items, idx_to_class = load_forestnet_items(args.data_dir, split="val", mode=mode)
        test_items, _ = load_forestnet_items(args.data_dir, split="test", mode=mode)

        val_indices = np.arange(len(val_items), dtype=np.int64)
        test_indices = np.arange(len(test_items), dtype=np.int64)

        # Keep CLI semantics: max_train limits database size and max_test limits query size.
        test_indices = maybe_subsample_indices(test_indices, args.max_train, args.seed + 5000)
        val_indices = maybe_subsample_indices(val_indices, args.max_test, args.seed + 6000)

        val_items = [val_items[i] for i in val_indices]
        test_items = [test_items[i] for i in test_indices]

        if len(val_items) == 0 or len(test_items) == 0:
            raise RuntimeError(
                f"{mode_name} has empty query/database split after subsampling. "
                "Adjust --max_train/--max_test."
            )

        print(f"\n{mode_name} Retrieval Evaluation (fixed split)")
        print(f"Classes: {len(idx_to_class)}")
        print(f"Query split (val): {len(val_items)}")
        print(f"Database split (test): {len(test_items)}")

    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        embedding_dim=args.embedding_dim,
        config_path=args.config_path,
        use_multi_scale=args.use_multi_scale,
        layer_indices=args.layer_indices,
        img_size=args.img_size,
        in_chans=args.in_chans,
    )

    model_tag = Path(args.model_path).stem
    index_dir = Path(args.index_dir)
    ks = (1, 5, 10)

    import faiss

    for mode, mode_name in mode_specs:
        val_items, idx_to_class = load_forestnet_items(args.data_dir, split="val", mode=mode)
        test_items, _ = load_forestnet_items(args.data_dir, split="test", mode=mode)

        val_indices = np.arange(len(val_items), dtype=np.int64)
        test_indices = np.arange(len(test_items), dtype=np.int64)
        test_indices = maybe_subsample_indices(test_indices, args.max_train, args.seed + 5000)
        val_indices = maybe_subsample_indices(val_indices, args.max_test, args.seed + 6000)

        val_items = [val_items[i] for i in val_indices]
        test_items = [test_items[i] for i in test_indices]

        val_loader = DataLoader(
            ForestNetDataset(val_items, img_size=args.img_size, in_chans=args.in_chans),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            ForestNetDataset(test_items, img_size=args.img_size, in_chans=args.in_chans),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        query_embeddings, query_labels = extract_embeddings(model, val_loader, device)
        db_embeddings, db_labels = extract_embeddings(model, test_loader, device)

        if query_embeddings.size == 0 or db_embeddings.size == 0:
            raise RuntimeError(f"Failed to extract embeddings for {mode_name}.")

        index_path = index_dir / (
            f"forestnet{mode}_{model_tag}_"
            f"ndb{len(db_labels)}_nquery{len(query_labels)}.index"
        )

        if index_path.exists():
            print(f"[{mode_name}] Loading FAISS index: {index_path}")
            index = faiss.read_index(str(index_path))
        else:
            print(f"[{mode_name}] Building FAISS index...")
            index = build_faiss_index(db_embeddings, index_type=args.index_type, nlist=args.nlist)
            if args.save_index:
                index_dir.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_path))
                print(f"[{mode_name}] Saved FAISS index: {index_path}")

        recall_at, map_at = compute_metrics(index, db_labels, query_embeddings, query_labels, ks=ks)

        print(f"[{mode_name}] Query={len(query_labels)} | Database={len(db_labels)}")
        for k in ks:
            print(f"[{mode_name}] Recall@{k}: {recall_at[k]:.4f} | mAP@{k}: {map_at[k]:.4f}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
