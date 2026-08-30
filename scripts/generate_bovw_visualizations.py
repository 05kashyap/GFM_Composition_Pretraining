#!/usr/bin/env python3
"""Generate publication-ready BoVW visualizations.

This script creates up to seven figures for the BoVW pipeline and saves them as
300-DPI PNG files under ``outputs/visualizations`` by default.

Each visualization runs independently. If one fails, a warning is printed and
execution continues to the next visualization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - handled at runtime if unavailable
    torch = None

try:
    from sklearn.manifold import TSNE
except Exception:  # pragma: no cover - handled at runtime if unavailable
    TSNE = None


Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FMOW_CATEGORIES: List[str] = [
    "zoo",
    "wind_farm",
    "water_treatment_facility",
    "waste_disposal",
    "tunnel_opening",
    "tower",
    "toll_booth",
    "swimming_pool",
    "surface_mine",
    "storage_tank",
    "stadium",
    "space_facility",
    "solar_farm",
    "smokestack",
    "single-unit_residential",
    "shopping_mall",
    "shipyard",
    "runway",
    "road_bridge",
    "recreational_facility",
    "railway_bridge",
    "race_track",
    "prison",
    "port",
    "police_station",
    "place_of_worship",
    "parking_lot_or_garage",
    "park",
    "oil_or_gas_facility",
    "office_building",
    "nuclear_powerplant",
    "multi-unit_residential",
    "military_facility",
    "lighthouse",
    "lake_or_pond",
    "interchange",
    "impoverished_settlement",
    "hospital",
    "helipad",
    "ground_transportation_station",
    "golf_course",
    "gas_station",
    "fountain",
    "flooded_road",
    "fire_station",
    "factory_or_powerplant",
    "electric_substation",
    "educational_institution",
    "debris_or_rubble",
    "dam",
    "crop_field",
    "construction_site",
    "car_dealership",
    "burial_site",
    "border_checkpoint",
    "barn",
    "archaeological_site",
    "aquaculture",
    "amusement_park",
    "airport_terminal",
    "airport_hangar",
    "airport",
    "false_detection",
]

CAT2LABEL: Dict[str, int] = {cat: i for i, cat in enumerate(FMOW_CATEGORIES)}
NUM_CLASSES = len(FMOW_CATEGORIES)

KNOWN_MANIFEST_PREFIXES = (
    "Hosted-Datasets/fmow/fmow-rgb/",
    "Hosted-Datasets/fmow/fmow-rgb-prepped/",
)

DEFAULT_DINOV3_WEIGHTS = "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VIZ_FILENAMES = {
    1: "viz_01_histogram_gallery.png",
    2: "viz_02_vocabulary_gallery.png",
    3: "viz_03_similarity_matrix.png",
    4: "viz_04_centroid_tsne.png",
    5: "viz_05_prediction_convergence.png",
    6: "viz_06_spatial_heatmaps.png",
    7: "viz_07_category_prototypes.png",
    8: "viz_08_cluster_similarity.png",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CellRecord:
    manifest_index: int
    raw_entry: dict
    image_path_raw: str
    image_path_resolved: Optional[Path]
    cell_box: Optional[Tuple[int, int, int, int]]
    cell_row: Optional[int]
    cell_col: Optional[int]
    dominant_label_from_manifest: int


@dataclass
class VizContext:
    workspace_root: Path
    manifest_path: Path
    histogram_dir: Path
    vocab_dir: Path
    patch_token_dir: Path
    dinov3_embed_dir: Path
    checkpoint_dir: Path
    output_dir: Path
    seed: int
    rng: np.random.Generator

    records: List[CellRecord]
    manifest_meta: dict

    histograms: np.memmap
    cell_ids: np.ndarray
    manifest_to_hist_row: Dict[int, int]

    labels_per_row: np.ndarray

    centroids: np.ndarray
    ground_cost: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Helpers: formatting, paths, loading
# ---------------------------------------------------------------------------


def setup_matplotlib_style() -> None:
    """Set a clean, consistent, publication-oriented style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "image.interpolation": "nearest",
        }
    )


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def parse_skip(skip_arg: str) -> set[int]:
    if not skip_arg:
        return set()
    out: set[int] = set()
    for token in skip_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            warn(f"Ignoring invalid --skip token: {token}")
    return out


def strip_known_prefixes(path_str: str) -> str:
    s = path_str.replace("\\", "/")
    for prefix in KNOWN_MANIFEST_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def canonical_rel_path(path_str: str) -> str:
    s = strip_known_prefixes(path_str)
    s = s.replace("\\", "/")
    marker = "/data/fmow/"
    if marker in s:
        s = s.split(marker, 1)[1]
    if s.startswith("data/fmow/"):
        s = s[len("data/fmow/") :]
    s = s.lstrip("./")
    return s


def infer_workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_image_path(raw_path: str, workspace_root: Path) -> Optional[Path]:
    if not raw_path:
        return None

    p = Path(raw_path)
    if p.exists():
        return p

    rel = strip_known_prefixes(raw_path)
    rel = rel.lstrip("/")

    candidates: List[Path] = []

    # Most common repo-local data root.
    candidates.append(workspace_root / "data" / "fmow" / rel)

    # If manifest already stores train/... paths.
    candidates.append(workspace_root / rel)

    # Sometimes relative to CWD.
    candidates.append(Path.cwd() / rel)

    # If rel still starts with data/fmow.
    if rel.startswith("data/fmow/"):
        candidates.append(workspace_root / rel)

    for cand in candidates:
        if cand.exists():
            return cand

    return None


def extract_cell_box(entry: dict) -> Optional[Tuple[int, int, int, int]]:
    key_set = {"cell_x0", "cell_y0", "cell_x1", "cell_y1"}
    if key_set.issubset(entry.keys()):
        return (
            int(round(float(entry["cell_x0"]))),
            int(round(float(entry["cell_y0"]))),
            int(round(float(entry["cell_x1"]))),
            int(round(float(entry["cell_y1"]))),
        )

    if "coords" in entry:
        coords = entry["coords"]
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            return tuple(int(round(float(v))) for v in coords)  # type: ignore[return-value]
        if isinstance(coords, dict):
            if all(k in coords for k in ("x0", "y0", "x1", "y1")):
                return (
                    int(round(float(coords["x0"]))),
                    int(round(float(coords["y0"]))),
                    int(round(float(coords["x1"]))),
                    int(round(float(coords["y1"]))),
                )

    alt = ("x0", "y0", "x1", "y1")
    if all(k in entry for k in alt):
        return (
            int(round(float(entry["x0"]))),
            int(round(float(entry["y0"]))),
            int(round(float(entry["x1"]))),
            int(round(float(entry["y1"]))),
        )

    return None


def category_from_path(path_str: str) -> Optional[str]:
    m = re.search(r"(?:train|val)/([^/]+)/", path_str)
    if not m:
        return None
    return m.group(1)


def parse_manifest(manifest_path: Path, workspace_root: Path) -> Tuple[List[CellRecord], dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    manifest_meta: dict = raw if isinstance(raw, dict) else {}

    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict) and "cells" in raw:
        entries = raw["cells"]
    else:
        raise ValueError(f"Unsupported manifest format in {manifest_path}")

    records: List[CellRecord] = []
    for idx, entry in enumerate(entries):
        img_raw = str(entry.get("img_path", entry.get("image_path", "")))
        box = extract_cell_box(entry)
        row = entry.get("cell_row")
        col = entry.get("cell_col")

        dominant_label = -1
        if "dominant_label" in entry:
            try:
                dominant_label = int(entry["dominant_label"])
            except Exception:
                dominant_label = -1

        rec = CellRecord(
            manifest_index=idx,
            raw_entry=entry,
            image_path_raw=img_raw,
            image_path_resolved=resolve_image_path(img_raw, workspace_root),
            cell_box=box,
            cell_row=int(row) if row is not None else None,
            cell_col=int(col) if col is not None else None,
            dominant_label_from_manifest=dominant_label,
        )
        records.append(rec)

    return records, manifest_meta


def open_hist_memmap(hist_path: Path) -> np.memmap:
    # Explicit memmap open as requested.
    return np.lib.format.open_memmap(hist_path, mode="r")


def load_cell_ids(cell_ids_path: Path, n_rows: int) -> np.ndarray:
    if cell_ids_path.exists():
        cell_ids = np.load(cell_ids_path)
        if cell_ids.shape[0] != n_rows:
            warn(
                f"cell_ids length ({cell_ids.shape[0]}) != hist rows ({n_rows}); using row indices"
            )
            return np.arange(n_rows, dtype=np.int64)
        return cell_ids.astype(np.int64, copy=False)
    warn(f"Missing {cell_ids_path}; assuming row index == manifest index")
    return np.arange(n_rows, dtype=np.int64)


def load_labels(
    labels_path: Path,
    records: Sequence[CellRecord],
    cell_ids: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    labels = np.full(n_rows, -1, dtype=np.int64)

    labels_arr: Optional[np.ndarray] = None
    if labels_path.exists():
        loaded = np.load(labels_path, mmap_mode="r")
        if loaded.ndim == 1:
            labels_arr = loaded
        else:
            warn(f"Ignoring non-1D labels array: {labels_path} shape={loaded.shape}")
    else:
        warn(f"Labels file not found: {labels_path}; using manifest/path-derived labels")

    if labels_arr is not None:
        max_id = int(cell_ids.max()) if cell_ids.size else -1
        if labels_arr.shape[0] > max_id:
            labels = np.asarray(labels_arr[cell_ids], dtype=np.int64)
        elif labels_arr.shape[0] == n_rows:
            labels = np.asarray(labels_arr, dtype=np.int64)
        else:
            warn(
                "Could not align label array with cell_ids or histogram rows; "
                "falling back to manifest/path labels"
            )

    # Fill missing labels from manifest dominant_label or path category.
    missing = np.where(labels < 0)[0]
    if missing.size > 0:
        for row_idx in missing:
            m_idx = int(cell_ids[row_idx])
            if 0 <= m_idx < len(records):
                rec = records[m_idx]

                if rec.dominant_label_from_manifest >= 0:
                    labels[row_idx] = rec.dominant_label_from_manifest
                    continue

                cat = category_from_path(rec.image_path_raw)
                if cat is not None and cat in CAT2LABEL:
                    labels[row_idx] = CAT2LABEL[cat]

    return labels


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def soft_assign(tokens: np.ndarray, centroids: np.ndarray, beta: float = 10.0) -> np.ndarray:
    sim = tokens @ centroids.T
    dist = 1.0 - sim
    w = np.exp(-beta * dist)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return w


def histogram_entropy(h: np.ndarray) -> float:
    x = np.asarray(h, dtype=np.float64)
    return float(-(x * np.log(x + 1e-12)).sum())


def compute_entropies_chunked(hist: np.memmap, chunk: int = 4096) -> np.ndarray:
    n = hist.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = np.asarray(hist[s:e], dtype=np.float64)
        out[s:e] = (-(block * np.log(block + 1e-12)).sum(axis=1)).astype(np.float32)
    return out


def to_rgb_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def crop_and_resize_cell_image(rec: CellRecord, out_size: int = 512) -> Optional[np.ndarray]:
    if rec.image_path_resolved is None or not rec.image_path_resolved.exists():
        return None

    try:
        img = Image.open(rec.image_path_resolved).convert("RGB")
    except Exception:
        return None

    if rec.cell_box is not None:
        x0, y0, x1, y1 = rec.cell_box
        x0 = max(0, min(x0, img.width - 1))
        y0 = max(0, min(y0, img.height - 1))
        x1 = max(x0 + 1, min(x1, img.width))
        y1 = max(y0 + 1, min(y1, img.height))
        img = img.crop((x0, y0, x1, y1))

    if img.size != (out_size, out_size):
        img = img.resize((out_size, out_size), Image.BICUBIC)

    return to_rgb_uint8(img)


# ---------------------------------------------------------------------------
# Patch-token resolver
# ---------------------------------------------------------------------------


class PatchTokenResolver:
    """Resolve patch-token .npz paths for manifest entries.

    Primary mode uses the same cache-key construction as Phase 1 extraction.
    If that fails, it can build a fallback index from NPZ metadata.
    """

    def __init__(
        self,
        patch_token_dir: Path,
        workspace_root: Path,
        weights_path: Optional[str] = None,
    ):
        self.patch_token_dir = patch_token_dir
        self.workspace_root = workspace_root

        default_weights = str((workspace_root / DEFAULT_DINOV3_WEIGHTS))
        self.weight_candidates = [
            weights_path or DEFAULT_DINOV3_WEIGHTS,
            default_weights,
            str(Path(DEFAULT_DINOV3_WEIGHTS)),
        ]

        # Remove duplicates while preserving order.
        seen = set()
        uniq = []
        for w in self.weight_candidates:
            if w in seen:
                continue
            uniq.append(w)
            seen.add(w)
        self.weight_candidates = uniq

        self._fallback_index: Optional[Dict[str, Path]] = None

    @staticmethod
    def _weights_fingerprint(weights_path: str) -> Tuple[Optional[float], Optional[int]]:
        p = Path(weights_path)
        if not p.exists():
            return None, None
        try:
            st = p.stat()
            return float(st.st_mtime), int(st.st_size)
        except Exception:
            return None, None

    def _compute_cache_key(self, image_path_str: str, weights_path: str) -> str:
        mtime, fsize = self._weights_fingerprint(weights_path)
        payload = {
            "image": str(image_path_str),
            "weights_path": str(weights_path),
            "weights_mtime": mtime,
            "weights_size": fsize,
            "cell_size": 512,
            "output_type": "patch_tokens_bovw",
            "output_shape": [256, 1024],
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _candidate_image_strings(self, rec: CellRecord) -> List[str]:
        cands: List[str] = []

        if rec.image_path_raw:
            cands.append(rec.image_path_raw)

        stripped = strip_known_prefixes(rec.image_path_raw)
        if stripped:
            cands.append(stripped)
            cands.append(str(Path("data/fmow") / stripped))

        canon = canonical_rel_path(rec.image_path_raw)
        if canon:
            cands.append(canon)
            cands.append(str(Path("data/fmow") / canon))

        if rec.image_path_resolved is not None:
            cands.append(str(rec.image_path_resolved))
            try:
                rel = rec.image_path_resolved.relative_to(self.workspace_root)
                cands.append(str(rel))
            except Exception:
                pass

        # De-duplicate preserving order.
        seen = set()
        out = []
        for s in cands:
            if s in seen:
                continue
            out.append(s)
            seen.add(s)
        return out

    def _build_fallback_index(self) -> None:
        info("Building fallback patch-token index from NPZ metadata (one-time)...")
        index: Dict[str, Path] = {}
        files = sorted(self.patch_token_dir.glob("*.npz"))
        for npz_path in files:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "img_path" not in data:
                        continue
                    img_path = str(data["img_path"])
            except Exception:
                continue

            key = canonical_rel_path(img_path)
            if key and key not in index:
                index[key] = npz_path

        self._fallback_index = index
        info(f"Fallback index entries: {len(index)}")

    def resolve(self, rec: CellRecord) -> Optional[Path]:
        if not self.patch_token_dir.exists():
            return None

        image_cands = self._candidate_image_strings(rec)
        for img_cand in image_cands:
            for w_cand in self.weight_candidates:
                key = self._compute_cache_key(img_cand, w_cand)
                path = self.patch_token_dir / f"{key}.npz"
                if path.exists():
                    return path

        # Fallback: match by canonical image path in NPZ metadata.
        if self._fallback_index is None:
            self._build_fallback_index()

        if self._fallback_index:
            for img_cand in image_cands:
                canon = canonical_rel_path(img_cand)
                if canon in self._fallback_index:
                    return self._fallback_index[canon]

        return None


def load_patch_tokens(npz_path: Path) -> Optional[np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "patch_tokens" not in data:
                return None
            tok = np.asarray(data["patch_tokens"], dtype=np.float32)
            return tok
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def resolve_checkpoint_dir(requested: Path, workspace_root: Path) -> Path:
    if requested.exists():
        return requested

    outputs_dir = workspace_root / "outputs"
    if not outputs_dir.exists():
        return requested

    # Common fallback patterns.
    candidates = []
    for d in outputs_dir.glob("bovw_training*"):
        if d.is_dir():
            has_ckpt = any(d.glob("epoch_*.pth")) or (d / "best_model.pth").exists()
            if has_ckpt:
                candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = candidates[0]
        warn(f"Checkpoint dir {requested} not found; using {chosen}")
        return chosen

    return requested


def build_context(args: argparse.Namespace) -> VizContext:
    workspace_root = infer_workspace_root()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = workspace_root / manifest_path

    histogram_dir = Path(args.histogram_dir)
    if not histogram_dir.is_absolute():
        histogram_dir = workspace_root / histogram_dir

    vocab_dir = Path(args.vocab_dir)
    if not vocab_dir.is_absolute():
        vocab_dir = workspace_root / vocab_dir

    patch_token_dir = Path(args.patch_token_dir)
    if not patch_token_dir.is_absolute():
        patch_token_dir = workspace_root / patch_token_dir

    dinov3_embed_dir = Path(args.dinov3_embed_dir)
    if not dinov3_embed_dir.is_absolute():
        dinov3_embed_dir = workspace_root / dinov3_embed_dir

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = workspace_root / checkpoint_dir
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir, workspace_root)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, manifest_meta = parse_manifest(manifest_path, workspace_root)

    hist_path = histogram_dir / "histograms.npy"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing histogram file: {hist_path}")
    histograms = open_hist_memmap(hist_path)

    n_rows, k_bins = histograms.shape
    if k_bins <= 0:
        raise ValueError(f"Invalid histogram shape: {histograms.shape}")

    cell_ids = load_cell_ids(histogram_dir / "cell_ids.npy", n_rows)
    manifest_to_hist_row = {int(m_idx): int(r_idx) for r_idx, m_idx in enumerate(cell_ids)}

    if args.cell_labels:
        labels_path = Path(args.cell_labels)
        if not labels_path.is_absolute():
            labels_path = workspace_root / labels_path
    else:
        labels_path = histogram_dir / "cell_labels.npy"

    labels_per_row = load_labels(labels_path, records, cell_ids, n_rows)

    centroids_path = vocab_dir / "centroids.npy"
    if not centroids_path.exists():
        raise FileNotFoundError(f"Missing centroids: {centroids_path}")
    centroids = np.load(centroids_path).astype(np.float32)
    centroids = l2_normalize(centroids, axis=1)

    ground_cost_path = vocab_dir / "ground_cost.npy"
    ground_cost = None
    if ground_cost_path.exists():
        ground_cost = np.load(ground_cost_path).astype(np.float32)

    ctx = VizContext(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        histogram_dir=histogram_dir,
        vocab_dir=vocab_dir,
        patch_token_dir=patch_token_dir,
        dinov3_embed_dir=dinov3_embed_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        seed=int(args.seed),
        rng=np.random.default_rng(int(args.seed)),
        records=records,
        manifest_meta=manifest_meta,
        histograms=histograms,
        cell_ids=cell_ids,
        manifest_to_hist_row=manifest_to_hist_row,
        labels_per_row=labels_per_row,
        centroids=centroids,
        ground_cost=ground_cost,
    )

    return ctx


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def label_name(label_id: int) -> str:
    if 0 <= int(label_id) < len(FMOW_CATEGORIES):
        return FMOW_CATEGORIES[int(label_id)]
    return "unlabeled"


def row_to_record(ctx: VizContext, row_idx: int) -> Optional[CellRecord]:
    m_idx = int(ctx.cell_ids[row_idx])
    if 0 <= m_idx < len(ctx.records):
        return ctx.records[m_idx]
    return None


def find_rows_for_labels(ctx: VizContext, label_ids: Sequence[int]) -> np.ndarray:
    if not label_ids:
        return np.array([], dtype=np.int64)
    return np.where(np.isin(ctx.labels_per_row, np.array(label_ids, dtype=np.int64)))[0]


def choose_high_entropy_row(
    ctx: VizContext,
    candidate_rows: np.ndarray,
    entropies: np.ndarray,
    resolver: PatchTokenResolver,
    require_patch: bool = True,
    require_image: bool = True,
) -> Optional[int]:
    if candidate_rows.size == 0:
        return None

    # Highest entropy first.
    order = np.argsort(entropies[candidate_rows])[::-1]
    for idx in order:
        row = int(candidate_rows[idx])
        rec = row_to_record(ctx, row)
        if rec is None:
            continue
        if require_image:
            img = crop_and_resize_cell_image(rec)
            if img is None:
                continue
        if require_patch:
            npz = resolver.resolve(rec)
            if npz is None:
                continue
        return row
    return None


def choose_sparse_row(
    ctx: VizContext,
    candidate_rows: np.ndarray,
    resolver: PatchTokenResolver,
) -> Optional[int]:
    if candidate_rows.size == 0:
        return None

    # Prefer peaked histograms: high max weight, then low entropy.
    best_row = None
    best_tuple = None
    for row in candidate_rows:
        rec = row_to_record(ctx, int(row))
        if rec is None:
            continue
        if crop_and_resize_cell_image(rec) is None:
            continue
        if resolver.resolve(rec) is None:
            continue

        h = np.asarray(ctx.histograms[int(row)], dtype=np.float32)
        score = (float(h.max()), -histogram_entropy(h))
        if best_tuple is None or score > best_tuple:
            best_tuple = score
            best_row = int(row)

    return best_row


def top_populated_categories(labels_per_row: np.ndarray, top_n: int = 8) -> List[int]:
    valid = labels_per_row[labels_per_row >= 0]
    if valid.size == 0:
        return []
    counts = np.bincount(valid, minlength=NUM_CLASSES)
    order = np.argsort(counts)[::-1]
    out = [int(i) for i in order[:top_n] if counts[i] > 0]
    return out


# ---------------------------------------------------------------------------
# DINO pooled embeddings loader
# ---------------------------------------------------------------------------


class DinoEmbeddingIndex:
    """Index pooled DINO embeddings from cache NPZ files.

    Expected NPZ keys from ``pipeline_utils.save_embeddings_to_cache``:
      - emb: (N_patches, D)
      - image_path: str
      - cell_row: int
      - cell_col: int
    """

    def __init__(self, dinov3_dir: Path):
        self.dir = dinov3_dir
        self.by_key: Dict[Tuple[str, int, int], np.ndarray] = {}
        self.by_image: Dict[str, np.ndarray] = {}
        self._built = False

    def build(self, needed_records: Sequence[CellRecord]) -> None:
        if self._built:
            return
        self._built = True

        if not self.dir.exists():
            warn(f"DINO cache dir not found: {self.dir}")
            return

        needed_paths = {canonical_rel_path(r.image_path_raw) for r in needed_records}

        files = sorted(self.dir.glob("*.npz"))
        if not files:
            warn(f"No NPZ files found in DINO cache dir: {self.dir}")
            return

        info(f"Indexing pooled DINO embeddings from {len(files)} files...")

        for npz_path in files:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "emb" not in data:
                        continue
                    emb = np.asarray(data["emb"], dtype=np.float32)
                    if emb.ndim != 2 or emb.shape[0] == 0:
                        continue

                    img_path = str(data["image_path"]) if "image_path" in data else ""
                    if not img_path:
                        continue
                    canon = canonical_rel_path(img_path)
                    if needed_paths and canon not in needed_paths:
                        continue

                    row = int(data["cell_row"]) if "cell_row" in data else 0
                    col = int(data["cell_col"]) if "cell_col" in data else 0
                    pooled = emb.mean(axis=0).astype(np.float32)

                    key = (canon, row, col)
                    if key not in self.by_key:
                        self.by_key[key] = pooled

                    if canon not in self.by_image:
                        self.by_image[canon] = pooled
            except Exception:
                continue

        info(
            "DINO index built: "
            f"{len(self.by_key)} keyed embeddings, {len(self.by_image)} image-level entries"
        )

    def get_for_record(self, rec: CellRecord) -> Optional[np.ndarray]:
        canon = canonical_rel_path(rec.image_path_raw)

        row = 0 if rec.cell_row is None else int(rec.cell_row)
        col = 0 if rec.cell_col is None else int(rec.cell_col)

        key = (canon, row, col)
        if key in self.by_key:
            return self.by_key[key]

        # Fallback to image-level pooled vector.
        if canon in self.by_image:
            return self.by_image[canon]

        # Another common fallback key.
        key00 = (canon, 0, 0)
        if key00 in self.by_key:
            return self.by_key[key00]

        return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def highlight_topk_barcolors(values: np.ndarray, topk: int, base: str, highlight: str) -> List[str]:
    colors = [base] * values.shape[0]
    k = min(topk, values.shape[0])
    idx = np.argsort(values)[-k:]
    for i in idx:
        colors[int(i)] = highlight
    return colors


def make_patch_mosaic(patches: List[np.ndarray], grid: int = 4, patch_size: int = 32) -> np.ndarray:
    canvas = np.zeros((grid * patch_size, grid * patch_size, 3), dtype=np.uint8)
    for i in range(min(len(patches), grid * grid)):
        r = i // grid
        c = i % grid
        p = patches[i]
        if p.shape[0] != patch_size or p.shape[1] != patch_size:
            p_img = Image.fromarray(p)
            p = np.asarray(p_img.resize((patch_size, patch_size), Image.BICUBIC), dtype=np.uint8)
        canvas[r * patch_size : (r + 1) * patch_size, c * patch_size : (c + 1) * patch_size] = p

    # Draw thin separators.
    for g in range(1, grid):
        canvas[g * patch_size - 1 : g * patch_size + 1, :, :] = 255
        canvas[:, g * patch_size - 1 : g * patch_size + 1, :] = 255
    return canvas


def pairwise_l1(X: np.ndarray) -> np.ndarray:
    return np.abs(X[:, None, :] - X[None, :, :]).sum(axis=-1)


def pairwise_cosine_similarity(X: np.ndarray) -> np.ndarray:
    Xn = l2_normalize(X, axis=1)
    return Xn @ Xn.T


def within_between_means(dist: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    n = dist.shape[0]
    if n < 2:
        return float("nan"), float("nan")

    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    if same.any():
        within = float(np.mean(dist[iu][same]))
    else:
        within = float("nan")

    if (~same).any():
        between = float(np.mean(dist[iu][~same]))
    else:
        between = float("nan")

    return within, between


def sinkhorn_emd_single(
    pred: np.ndarray,
    target: np.ndarray,
    ground_cost: Optional[np.ndarray],
    eps: float = 0.05,
    iters: int = 50,
) -> float:
    """Single-sample Sinkhorn EMD approximation in numpy."""
    p = np.asarray(pred, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)

    p = np.maximum(p, 1e-8)
    q = np.maximum(q, 1e-8)
    p = p / p.sum()
    q = q / q.sum()

    k = p.shape[0]
    if ground_cost is None:
        idx = np.arange(k, dtype=np.float64)
        C = np.abs(idx[:, None] - idx[None, :])
        C /= np.maximum(C.max(), 1e-8)
    else:
        C = np.asarray(ground_cost, dtype=np.float64)

    logK = -C / eps
    log_u = np.zeros_like(p)
    log_v = np.zeros_like(p)

    log_p = np.log(p)
    log_q = np.log(q)

    for _ in range(iters):
        log_u = log_q - np.logaddexp.reduce(logK + log_v[None, :], axis=1)
        log_v = log_p - np.logaddexp.reduce(logK.T + log_u[None, :], axis=1)

    log_T = log_u[:, None] + logK + log_v[None, :]
    T = np.exp(log_T)
    return float(np.sum(T * C))


# ---------------------------------------------------------------------------
# Visualization 1
# ---------------------------------------------------------------------------


def viz_01_histogram_gallery(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[1]

    entropies = compute_entropies_chunked(ctx.histograms)
    target_examples = 8

    theme_to_cats = {
        "Water/Harbour": ["port", "shipyard", "lake_or_pond", "aquaculture", "water_treatment_facility"],
        "Airport/Runway": ["airport", "runway", "airport_terminal", "airport_hangar"],
        "Dense Urban": [
            "office_building",
            "multi-unit_residential",
            "parking_lot_or_garage",
            "shopping_mall",
            "interchange",
            "ground_transportation_station",
        ],
        "Dense Forest": ["park", "golf_course"],
        "Farmland/Agriculture": ["crop_field", "barn", "aquaculture"],
        "Desert/Barren": ["surface_mine", "debris_or_rubble", "waste_disposal"],
    }

    selected_rows: List[int] = []
    selected_titles: List[str] = []

    for theme, cat_names in theme_to_cats.items():
        label_ids = [CAT2LABEL[c] for c in cat_names if c in CAT2LABEL]
        rows = find_rows_for_labels(ctx, label_ids)
        row = choose_high_entropy_row(ctx, rows, entropies, resolver, require_patch=False, require_image=True)

        if row is None:
            warn(f"Viz1: no usable row found for theme '{theme}', using global high-entropy fallback")
            all_rows = np.arange(ctx.histograms.shape[0], dtype=np.int64)
            row = choose_high_entropy_row(
                ctx,
                all_rows,
                entropies,
                resolver,
                require_patch=False,
                require_image=True,
            )
            if row is None:
                continue

        selected_rows.append(int(row))
        label_id = int(ctx.labels_per_row[row])
        if label_id >= 0:
            selected_titles.append(label_name(label_id))
        else:
            selected_titles.append(theme)

    # Fill with additional high-entropy, image-valid cells to reach target count.
    if len(selected_rows) < target_examples:
        seen_rows = set(selected_rows)
        for row in np.argsort(entropies)[::-1]:
            row = int(row)
            if row in seen_rows:
                continue
            rec = row_to_record(ctx, row)
            if rec is None:
                continue
            if crop_and_resize_cell_image(rec) is None:
                continue

            selected_rows.append(row)
            label_id = int(ctx.labels_per_row[row])
            if label_id >= 0:
                selected_titles.append(label_name(label_id))
            else:
                selected_titles.append("high-entropy")

            seen_rows.add(row)
            if len(selected_rows) >= target_examples:
                break

    # Keep exactly up to 8 examples.
    selected_rows = selected_rows[:target_examples]
    selected_titles = selected_titles[:target_examples]

    if not selected_rows:
        raise RuntimeError("Viz1: failed to select any cells")

    n_cells = len(selected_rows)
    n_cols = 4
    n_groups = int(math.ceil(n_cells / n_cols))

    # Use explicit spacer rows between groups to avoid title/xlabel overlap.
    row_heights: List[float] = []
    for g in range(n_groups):
        row_heights.extend([1.0, 0.9])
        if g < n_groups - 1:
            row_heights.append(0.38)

    fig = plt.figure(figsize=(18, 5.9 * n_groups))
    gs = fig.add_gridspec(
        nrows=len(row_heights),
        ncols=n_cols,
        hspace=0.30,
        wspace=0.18,
        height_ratios=row_heights,
    )

    for i, (row, title_txt) in enumerate(zip(selected_rows, selected_titles)):
        g = i // n_cols
        c = i % n_cols

        row_base = g * 3
        ax_img = fig.add_subplot(gs[row_base, c])
        ax_hist = fig.add_subplot(gs[row_base + 1, c])

        rec = row_to_record(ctx, row)
        if rec is None:
            continue

        img = crop_and_resize_cell_image(rec)
        if img is None:
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax_img.axis("off")
        else:
            ax_img.imshow(img)
            ax_img.set_title(title_txt, pad=6)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.grid(False)

        hist = np.asarray(ctx.histograms[row], dtype=np.float32)
        colors = highlight_topk_barcolors(hist, topk=5, base="#4C78A8", highlight="coral")
        ax_hist.bar(np.arange(hist.shape[0]), hist, color=colors, width=1.0, linewidth=0)
        ax_hist.set_xlim(0, hist.shape[0] - 1)
        ax_hist.set_xlabel("Vocabulary index")
        ax_hist.set_ylabel("Weight")
        ax_hist.set_xticks([0, 128, 256, 384, 511])
        ax_hist.grid(False)

    # Hide unused panels.
    total_slots = n_groups * n_cols
    for i in range(n_cells, total_slots):
        g = i // n_cols
        c = i % n_cols
        row_base = g * 3
        ax1 = fig.add_subplot(gs[row_base, c])
        ax2 = fig.add_subplot(gs[row_base + 1, c])
        ax1.axis("off")
        ax2.axis("off")

    fig.suptitle("BoVW Histogram Gallery: Diverse Cell Semantics", y=0.995, fontsize=13)
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 2
# ---------------------------------------------------------------------------


def viz_02_vocabulary_gallery(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[2]

    n_rows = ctx.histograms.shape[0]
    sample_n = min(1000, n_rows)
    sample_rows = ctx.rng.choice(n_rows, size=sample_n, replace=False)
    sample_hist = np.asarray(ctx.histograms[sample_rows], dtype=np.float32)

    total_activation = sample_hist.sum(axis=0)
    top_words = np.argsort(total_activation)[-20:][::-1]

    # Cache row-level resources to avoid repeated disk I/O.
    token_cache: Dict[int, np.ndarray] = {}
    image_cache: Dict[int, np.ndarray] = {}

    def get_tokens_for_row(row_idx: int) -> Optional[np.ndarray]:
        if row_idx in token_cache:
            return token_cache[row_idx]
        rec = row_to_record(ctx, row_idx)
        if rec is None:
            return None
        npz = resolver.resolve(rec)
        if npz is None:
            return None
        tok = load_patch_tokens(npz)
        if tok is None:
            return None
        token_cache[row_idx] = tok
        return tok

    def get_image_for_row(row_idx: int) -> Optional[np.ndarray]:
        if row_idx in image_cache:
            return image_cache[row_idx]
        rec = row_to_record(ctx, row_idx)
        if rec is None:
            return None
        img = crop_and_resize_cell_image(rec)
        if img is None:
            return None
        image_cache[row_idx] = img
        return img

    panel_mosaics: List[np.ndarray] = []
    panel_titles: List[str] = []

    for k in top_words:
        # Candidate cells from sample ranked by histogram bin k.
        ranking = np.argsort(sample_hist[:, k])[::-1]
        ranked_rows = sample_rows[ranking]

        patch_scores: List[Tuple[float, int, int]] = []

        # Search top cells for strong patch-level assignment to centroid k.
        for row in ranked_rows[:120]:
            row = int(row)
            tok = get_tokens_for_row(row)
            if tok is None:
                continue

            img = get_image_for_row(row)
            if img is None:
                continue

            tok = np.asarray(tok, dtype=np.float32)
            tok = l2_normalize(tok, axis=1)

            # Full soft assignment for this cell.
            weights = soft_assign(tok, ctx.centroids, beta=10.0)
            wk = weights[:, int(k)]

            # Keep strongest few patches from this cell.
            top_local = np.argsort(wk)[-6:][::-1]
            for patch_idx in top_local:
                patch_scores.append((float(wk[patch_idx]), row, int(patch_idx)))

        patch_scores.sort(key=lambda x: x[0], reverse=True)
        patch_scores = patch_scores[:16]

        patches_rgb: List[np.ndarray] = []
        for _, row, patch_idx in patch_scores:
            img = get_image_for_row(row)
            tok = get_tokens_for_row(row)
            if img is None or tok is None:
                continue

            n_tokens = tok.shape[0]
            side = int(round(math.sqrt(float(n_tokens))))
            if side <= 0:
                continue
            patch_size = img.shape[0] // side
            if patch_size <= 0:
                continue

            pr = patch_idx // side
            pc = patch_idx % side
            y0 = pr * patch_size
            x0 = pc * patch_size
            y1 = min(y0 + patch_size, img.shape[0])
            x1 = min(x0 + patch_size, img.shape[1])
            patch = img[y0:y1, x0:x1, :]
            patches_rgb.append(patch)

        if not patches_rgb:
            # Placeholder blank mosaic.
            mosaic = np.zeros((128, 128, 3), dtype=np.uint8)
            panel_mosaics.append(mosaic)
        else:
            panel_mosaics.append(make_patch_mosaic(patches_rgb, grid=4, patch_size=32))

        panel_titles.append(f"k={int(k)} | total={float(total_activation[k]):.3f}")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 5, wspace=0.15, hspace=0.22)

    for i in range(20):
        r = i // 5
        c = i % 5
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(panel_mosaics[i])
        ax.set_title(panel_titles[i], pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("Vocabulary Word Gallery: Top-Activating 32x32 Patches", y=0.995, fontsize=13)
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 3
# ---------------------------------------------------------------------------


def viz_03_similarity_matrix(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[3]

    top8 = top_populated_categories(ctx.labels_per_row, top_n=8)
    if not top8:
        raise RuntimeError("Viz3: no labeled categories available")

    selected_rows: List[int] = []
    selected_labels: List[int] = []
    category_counts: List[int] = []

    for cat in top8:
        rows = np.where(ctx.labels_per_row == cat)[0]
        if rows.size == 0:
            continue
        take = min(10, rows.size)
        if rows.size > take:
            rows = ctx.rng.choice(rows, size=take, replace=False)
        rows = np.sort(rows)
        selected_rows.extend(int(x) for x in rows)
        selected_labels.extend([int(cat)] * len(rows))
        category_counts.append(len(rows))

    if len(selected_rows) < 8:
        raise RuntimeError("Viz3: too few selected rows")

    # Sort by category.
    order = np.argsort(np.array(selected_labels, dtype=np.int64))
    selected_rows = [selected_rows[i] for i in order]
    selected_labels = [selected_labels[i] for i in order]

    H = np.asarray(ctx.histograms[np.array(selected_rows, dtype=np.int64)], dtype=np.float32)
    dist_l1 = pairwise_l1(H)

    within_l1, between_l1 = within_between_means(dist_l1, np.array(selected_labels, dtype=np.int64))

    # Load pooled embeddings from DINO cache if available; else fallback to patch-token mean.
    dino_index = DinoEmbeddingIndex(ctx.dinov3_embed_dir)
    needed_records = [row_to_record(ctx, r) for r in selected_rows]
    needed_records = [r for r in needed_records if r is not None]
    dino_index.build(needed_records)

    pooled_list: List[np.ndarray] = []
    missing_dino = 0

    for row in selected_rows:
        rec = row_to_record(ctx, row)
        if rec is None:
            continue

        emb = dino_index.get_for_record(rec)
        if emb is None:
            # Fallback: mean of patch tokens from Phase 1.
            npz = resolver.resolve(rec)
            tok = load_patch_tokens(npz) if npz is not None else None
            if tok is not None:
                emb = np.asarray(tok, dtype=np.float32).mean(axis=0)
            else:
                missing_dino += 1
                emb = None

        if emb is None:
            # Keep matrix dimensions aligned by inserting zeros.
            pooled_list.append(np.zeros((ctx.centroids.shape[1],), dtype=np.float32))
        else:
            pooled_list.append(np.asarray(emb, dtype=np.float32))

    if missing_dino > 0:
        warn(
            f"Viz3: missing pooled DINO embeddings for {missing_dino} selected cells; "
            "used zero vectors where necessary"
        )

    E = np.stack(pooled_list, axis=0)
    sim_cos = pairwise_cosine_similarity(E)
    cos_dist = 1.0 - sim_cos
    within_cd, between_cd = within_between_means(cos_dist, np.array(selected_labels, dtype=np.int64))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))

    ax0, ax1 = axes

    im0 = ax0.imshow(dist_l1, cmap="viridis", aspect="equal")
    ax0.set_title("BoVW histogram distances")
    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cbar0.set_label("L1 distance")

    im1 = ax1.imshow(sim_cos, cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)
    ax1.set_title("Pooled DINOv3 cosine similarity")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Cosine similarity")

    # Category boundaries and labels.
    labels_arr = np.array(selected_labels, dtype=np.int64)
    unique_in_order: List[int] = []
    counts_in_order: List[int] = []
    for lab in labels_arr:
        if not unique_in_order or unique_in_order[-1] != int(lab):
            unique_in_order.append(int(lab))
            counts_in_order.append(1)
        else:
            counts_in_order[-1] += 1

    boundaries = np.cumsum(counts_in_order)[:-1] - 0.5
    centers = np.cumsum(counts_in_order) - np.array(counts_in_order) / 2.0 - 0.5
    tick_labels = [label_name(c) for c in unique_in_order]

    for ax in (ax0, ax1):
        for b in boundaries:
            ax.axhline(b, color="white", linewidth=0.8)
            ax.axvline(b, color="white", linewidth=0.8)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Category-sorted cells")
        ax.set_ylabel("Category-sorted cells")

    ax0.text(
        0.02,
        0.02,
        f"Within L1: {within_l1:.3f}\nBetween L1: {between_l1:.3f}",
        transform=ax0.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax1.text(
        0.02,
        0.02,
        f"Within cos-dist: {within_cd:.3f}\nBetween cos-dist: {between_cd:.3f}",
        transform=ax1.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    fig.suptitle("Category Separation: BoVW vs Pooled Embeddings", y=0.995, fontsize=13)
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 4
# ---------------------------------------------------------------------------


def viz_04_centroid_tsne(ctx: VizContext) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[4]

    if TSNE is None:
        raise RuntimeError("scikit-learn is required for t-SNE (viz 4)")

    K = ctx.centroids.shape[0]
    n_rows = ctx.histograms.shape[0]

    vote_mat = np.zeros((K, NUM_CLASSES), dtype=np.int64)
    activation_freq = np.zeros(K, dtype=np.float64)

    chunk = 4096
    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        block = np.asarray(ctx.histograms[s:e], dtype=np.float32)
        activation_freq += block.sum(axis=0)

        argmax_idx = np.argmax(block, axis=1)
        labels = ctx.labels_per_row[s:e]
        mask = labels >= 0
        if np.any(mask):
            np.add.at(vote_mat, (argmax_idx[mask], labels[mask]), 1)

    centroid_labels = np.full(K, -1, dtype=np.int64)
    has_votes = vote_mat.sum(axis=1) > 0
    centroid_labels[has_votes] = np.argmax(vote_mat[has_votes], axis=1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb2d = tsne.fit_transform(ctx.centroids)

    # Use tab10 for the 10 most common assigned categories.
    valid_labels = centroid_labels[centroid_labels >= 0]
    if valid_labels.size > 0:
        counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
        top10 = [int(i) for i in np.argsort(counts)[::-1][:10] if counts[i] > 0]
    else:
        top10 = []

    color_map = plt.get_cmap("tab10")
    label_to_color: Dict[int, Tuple[float, float, float, float]] = {}
    for i, lab in enumerate(top10):
        label_to_color[lab] = color_map(i % 10)

    point_colors = []
    for lab in centroid_labels:
        if int(lab) in label_to_color:
            point_colors.append(label_to_color[int(lab)])
        else:
            point_colors.append((0.7, 0.7, 0.7, 0.7))

    # Scale point sizes by activation frequency.
    af = activation_freq.astype(np.float64)
    af = af - af.min()
    af = af / (af.max() + 1e-12)
    sizes = 18.0 + 130.0 * af

    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    ax.scatter(emb2d[:, 0], emb2d[:, 1], c=point_colors, s=sizes, alpha=0.85, linewidths=0.2, edgecolors="k")
    ax.set_title("Vocabulary Centroid t-SNE (colored by dominant category)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    handles = []
    labels_txt = []
    for i, lab in enumerate(top10):
        handles.append(
            mpl.lines.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=color_map(i % 10))
        )
        labels_txt.append(label_name(lab))

    if handles:
        ax.legend(handles, labels_txt, loc="best", frameon=True, fontsize=8, title="Dominant category")

    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 5
# ---------------------------------------------------------------------------


def list_checkpoint_files(checkpoint_dir: Path) -> List[Tuple[str, Path, Optional[int]]]:
    """Return checkpoint tuples: (label, path, epoch_num_or_none)."""
    out: List[Tuple[str, Path, Optional[int]]] = []

    epoch_files = []
    for p in sorted(checkpoint_dir.glob("epoch_*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if not m:
            continue
        epoch_files.append((int(m.group(1)), p))

    # Prefer requested canonical epochs if present.
    preferred = [1, 10, 25, 50, 100]
    by_epoch = {ep: p for ep, p in epoch_files}
    selected: List[Tuple[int, Path]] = [(ep, by_epoch[ep]) for ep in preferred if ep in by_epoch]

    if len(selected) < 2:
        # Fall back to up to 5 available epoch checkpoints.
        selected = sorted(epoch_files)[:5] if epoch_files else []

    for ep, p in selected:
        out.append((f"epoch {ep}", p, ep))

    # If no epoch checkpoints exist, try best/final.
    if not out:
        best = checkpoint_dir / "best_model.pth"
        final = checkpoint_dir / "final_model.pth"
        if best.exists():
            out.append(("best", best, None))
        if final.exists():
            out.append(("final", final, None))

    return out


def _extract_state_dict(ckpt_obj) -> dict:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            sd = ckpt_obj["state_dict"]
        elif "model" in ckpt_obj:
            sd = ckpt_obj["model"]
        else:
            sd = ckpt_obj
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt_obj)}")

    # Strip possible DDP prefix.
    cleaned = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def _build_bovw_model(vocab_size: int, hidden_dim: int, ground_cost_path: Optional[str]):
    if torch is None:
        raise RuntimeError("PyTorch is required for viz 5")

    project_root = Path(__file__).resolve().parent.parent
    dynvis_root = project_root / "architectures" / "DynamicVis"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(dynvis_root) not in sys.path:
        sys.path.insert(0, str(dynvis_root))

    # Ensure DynamicVis backbone is registered.
    import dynamicvis  # noqa: F401
    from models.bovw_head import BoVWDynamicVis

    model = BoVWDynamicVis(
        backbone=dict(
            type="DynamicVisBackbone",
            arch="b",
            out_type="avg_featmap",
            out_indices=(3,),
        ),
        vocab_size=int(vocab_size),
        hidden_dim=int(hidden_dim),
        num_classes=63,
        ground_cost_path=ground_cost_path,
        lambda_emd=1.0,
        lambda_cls=0.5,
        lambda_mil=0.25,
        sinkhorn_eps=0.05,
        sinkhorn_iters=50,
    )
    model.eval()
    return model


def _infer_hidden_dim_from_state_dict(sd: dict, fallback: int = 512) -> int:
    k = "head.prediction_head.0.weight"
    if k in sd and hasattr(sd[k], "shape"):
        return int(sd[k].shape[0])
    return fallback


def preprocess_for_model(img_uint8: np.ndarray) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for viz 5")
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - IMG_MEAN[None, None, :]) / IMG_STD[None, None, :]
    x = np.transpose(x, (2, 0, 1))[None, ...]  # 1,C,H,W
    return torch.from_numpy(x)


def viz_05_prediction_convergence(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[5]

    if torch is None:
        raise RuntimeError("PyTorch is required for viz 5")

    ckpt_files = list_checkpoint_files(ctx.checkpoint_dir)
    if not ckpt_files:
        raise RuntimeError(f"Viz5: no checkpoints found in {ctx.checkpoint_dir}")

    # Select a sparse, visually distinctive airport-like cell.
    airport_like = [
        CAT2LABEL["airport"],
        CAT2LABEL["runway"],
        CAT2LABEL["airport_terminal"],
        CAT2LABEL["airport_hangar"],
    ]
    rows_air = find_rows_for_labels(ctx, airport_like)
    chosen_row = choose_sparse_row(ctx, rows_air, resolver)
    if chosen_row is None:
        warn("Viz5: no airport-like sparse cell found; using global sparse fallback")
        all_rows = np.arange(ctx.histograms.shape[0], dtype=np.int64)
        chosen_row = choose_sparse_row(ctx, all_rows, resolver)
    if chosen_row is None:
        raise RuntimeError("Viz5: failed to select a cell for convergence plot")

    rec = row_to_record(ctx, chosen_row)
    if rec is None:
        raise RuntimeError("Viz5: missing selected record")

    img = crop_and_resize_cell_image(rec)
    if img is None:
        raise RuntimeError("Viz5: selected cell image could not be loaded")

    target_hist = np.asarray(ctx.histograms[chosen_row], dtype=np.float32)

    # Build a model from the first checkpoint's dimensionality.
    first_ckpt = torch.load(ckpt_files[0][1], map_location="cpu", weights_only=False)
    first_sd = _extract_state_dict(first_ckpt)
    hidden_dim = _infer_hidden_dim_from_state_dict(first_sd)

    ground_cost_path = str(ctx.vocab_dir / "ground_cost.npy") if (ctx.vocab_dir / "ground_cost.npy").exists() else None
    model = _build_bovw_model(vocab_size=ctx.centroids.shape[0], hidden_dim=hidden_dim, ground_cost_path=ground_cost_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inp = preprocess_for_model(img).to(device)

    pred_curves: List[np.ndarray] = []
    pred_labels: List[str] = []
    emd_vals: List[float] = []
    emd_x: List[float] = []

    for lbl, ckpt_path, ep in ckpt_files:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            # Expected for some final-model variants that drop aux head.
            pass
        if unexpected:
            pass

        model.eval()
        with torch.no_grad():
            pred = model(inp, mode="tensor")
        pred_np = pred.detach().cpu().numpy().reshape(-1).astype(np.float32)

        pred_curves.append(pred_np)
        pred_labels.append(lbl)

        emd = sinkhorn_emd_single(pred_np, target_hist, ctx.ground_cost, eps=0.05, iters=50)
        emd_vals.append(float(emd))
        emd_x.append(float(ep) if ep is not None else float(len(emd_x) + 1))

    # Plot: left panel overlays histograms (bins 0-100), right panel EMD trend.
    fig = plt.figure(figsize=(14.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.2], wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(101)

    ax0.plot(x, target_hist[:101], color="black", linewidth=2.3, label="target")

    cmap = plt.get_cmap("winter")  # blue -> green
    n_pred = max(1, len(pred_curves))
    for i, (curve, lbl) in enumerate(zip(pred_curves, pred_labels)):
        color = cmap(i / max(1, n_pred - 1))
        ax0.plot(x, curve[:101], color=color, linewidth=1.6, alpha=0.95, label=lbl)

    ax0.set_title("Histogram Prediction Convergence (bins 0-100)")
    ax0.set_xlabel("Vocabulary bin")
    ax0.set_ylabel("Assignment weight")
    ax0.set_xlim(0, 100)
    ax0.grid(False)
    ax0.legend(loc="upper right", fontsize=8, ncol=2)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(emd_x, emd_vals, marker="o", color="#1f77b4", linewidth=1.8)
    for x_i, y_i, lbl in zip(emd_x, emd_vals, pred_labels):
        ax1.text(x_i, y_i, lbl, fontsize=7, ha="left", va="bottom")
    ax1.set_title("Cell-level EMD")
    ax1.set_xlabel("Checkpoint epoch")
    ax1.set_ylabel("Sinkhorn EMD")
    ax1.grid(False)

    cell_lab = int(ctx.labels_per_row[chosen_row])
    fig.suptitle(
        f"Prediction Convergence on Fixed Cell (label: {label_name(cell_lab)}, row={chosen_row})",
        y=0.995,
        fontsize=12,
    )

    save_figure(fig, out_path)
    return out_path


def select_diverse_words_for_cell(
    hist: np.ndarray,
    weights: np.ndarray,
    num_words: int = 6,
    pool_size: int = 64,
    min_rel_weight: float = 0.18,
    diversity_lambda: float = 0.82,
) -> List[int]:
    """Select high-weight yet diverse centroid indices for one cell.

    Diversity is measured by low cosine similarity between patch-activation maps.
    """
    hist = np.asarray(hist, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    k_vocab = int(hist.shape[0])
    if k_vocab <= 0:
        return []

    ranked = np.argsort(hist)[::-1]
    pool_size = int(min(max(1, pool_size), k_vocab))
    candidates = ranked[:pool_size]

    base_k = int(candidates[0])
    cutoff = float(hist[base_k]) * float(min_rel_weight)
    strong = candidates[hist[candidates] >= cutoff]
    if strong.size >= num_words:
        candidates = strong

    vec_cache: Dict[int, np.ndarray] = {}

    def get_norm_vec(word_idx: int) -> np.ndarray:
        if word_idx in vec_cache:
            return vec_cache[word_idx]
        v = np.asarray(weights[:, int(word_idx)], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n <= 1e-8:
            out = np.zeros_like(v)
        else:
            out = v / n
        vec_cache[word_idx] = out
        return out

    selected: List[int] = [base_k]
    cand_list = [int(c) for c in candidates.tolist()]

    target = int(min(max(1, num_words), k_vocab))
    while len(selected) < min(target, len(cand_list)):
        best_k: Optional[int] = None
        best_key: Optional[Tuple[float, float, float, float]] = None

        for k in cand_list:
            if k in selected:
                continue

            vk = get_norm_vec(k)
            max_sim = max(float(np.dot(vk, get_norm_vec(s))) for s in selected)
            rel_w = float(hist[k]) / float(hist[selected[0]] + 1e-8)
            score = rel_w - float(diversity_lambda) * max_sim

            # Prefer high composite score, then lower similarity, then higher weight,
            # then smaller centroid index for deterministic tie-breaks.
            key = (score, -max_sim, rel_w, -float(k))
            if best_key is None or key > best_key:
                best_key = key
                best_k = int(k)

        if best_k is None:
            break
        selected.append(best_k)

    if len(selected) < target:
        for k in ranked.tolist():
            k = int(k)
            if k in selected:
                continue
            selected.append(k)
            if len(selected) >= target:
                break

    return selected[:target]


def normalize_heat_for_overlay(heat_vec: np.ndarray, side: int) -> np.ndarray:
    """Normalize a patch-activation heatmap to emphasize its salient regions."""
    heat = np.asarray(heat_vec, dtype=np.float32).reshape(side, side)
    lo = float(np.percentile(heat, 55))
    hi = float(np.percentile(heat, 99))
    if hi <= lo + 1e-8:
        lo = float(heat.min())
        hi = float(heat.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(heat)
    return np.clip((heat - lo) / (hi - lo + 1e-8), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualization 6
# ---------------------------------------------------------------------------


def viz_06_spatial_heatmaps(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[6]

    harbour_ids = [
        CAT2LABEL["port"],
        CAT2LABEL["shipyard"],
        CAT2LABEL["lake_or_pond"],
        CAT2LABEL["aquaculture"],
        CAT2LABEL["water_treatment_facility"],
    ]
    forest_ids = [CAT2LABEL["park"], CAT2LABEL["golf_course"]]
    airport_ids = [
        CAT2LABEL["airport"],
        CAT2LABEL["runway"],
        CAT2LABEL["airport_terminal"],
        CAT2LABEL["airport_hangar"],
    ]
    urban_ids = [
        CAT2LABEL["office_building"],
        CAT2LABEL["multi-unit_residential"],
        CAT2LABEL["parking_lot_or_garage"],
        CAT2LABEL["shopping_mall"],
        CAT2LABEL["interchange"],
        CAT2LABEL["ground_transportation_station"],
    ]

    rows_h = find_rows_for_labels(ctx, harbour_ids)
    rows_f = find_rows_for_labels(ctx, forest_ids)
    rows_a = find_rows_for_labels(ctx, airport_ids)
    rows_u = find_rows_for_labels(ctx, urban_ids)

    entropies = compute_entropies_chunked(ctx.histograms)

    def pick_unique_high_entropy_row(candidate_rows: np.ndarray, used_rows: set[int]) -> Optional[int]:
        if candidate_rows.size == 0:
            return None

        order = np.argsort(entropies[candidate_rows])[::-1]
        for idx in order:
            row = int(candidate_rows[idx])
            if row in used_rows:
                continue
            rec = row_to_record(ctx, row)
            if rec is None:
                continue
            if crop_and_resize_cell_image(rec) is None:
                continue
            if resolver.resolve(rec) is None:
                continue
            return row
        return None

    selected_rows: List[int] = []
    row_source_tags: List[str] = []
    used_rows: set[int] = set()

    groups: List[Tuple[str, np.ndarray]] = [
        ("harbour/coastal", rows_h),
        ("forest", rows_f),
        ("airport", rows_a),
        ("urban", rows_u),
    ]

    for tag, candidates in groups:
        row = pick_unique_high_entropy_row(candidates, used_rows)
        if row is None:
            continue
        selected_rows.append(int(row))
        row_source_tags.append(tag)
        used_rows.add(int(row))

    target_cells = 4
    min_cells = 3
    if len(selected_rows) < target_cells:
        all_rows = np.arange(ctx.histograms.shape[0], dtype=np.int64)
        extra = pick_unique_high_entropy_row(all_rows, used_rows)
        while extra is not None and len(selected_rows) < target_cells:
            selected_rows.append(int(extra))
            row_source_tags.append("global-fallback")
            used_rows.add(int(extra))
            extra = pick_unique_high_entropy_row(all_rows, used_rows)

    if len(selected_rows) < min_cells:
        raise RuntimeError("Viz6: failed to select enough diverse cells")

    info(f"Viz6 selected rows: {selected_rows}")

    num_maps = 6
    n_cols = 1 + num_maps
    overlay_cmaps = ["viridis", "magma", "plasma", "inferno", "cividis", "turbo", "cubehelix", "YlOrRd"]
    n_rows = len(selected_rows)

    fig = plt.figure(figsize=(3.1 * n_cols, 3.35 * n_rows + 0.7))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.10, hspace=0.22)

    for r_idx, row in enumerate(selected_rows):
        rec = row_to_record(ctx, row)
        if rec is None:
            continue

        img = crop_and_resize_cell_image(rec)
        if img is None:
            continue

        npz = resolver.resolve(rec)
        if npz is None:
            continue
        tok = load_patch_tokens(npz)
        if tok is None:
            continue

        tok = l2_normalize(np.asarray(tok, dtype=np.float32), axis=1)
        weights = soft_assign(tok, ctx.centroids, beta=10.0)

        hist = np.asarray(ctx.histograms[row], dtype=np.float32)
        selected_words = select_diverse_words_for_cell(
            hist,
            weights,
            num_words=num_maps,
            pool_size=64,
            min_rel_weight=0.18,
            diversity_lambda=0.82,
        )

        word_vecs = []
        for k in selected_words:
            v = np.asarray(weights[:, int(k)], dtype=np.float32)
            n = float(np.linalg.norm(v))
            word_vecs.append(v / max(n, 1e-8))

        if word_vecs:
            primary = word_vecs[0]
        else:
            primary = None

        # Column 0: original cell image.
        ax0 = fig.add_subplot(gs[r_idx, 0])
        ax0.imshow(img)
        lbl = int(ctx.labels_per_row[row])
        src = row_source_tags[r_idx] if r_idx < len(row_source_tags) else "auto"
        ax0.set_title(f"{label_name(lbl)}\n{src}")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.grid(False)

        side = int(round(math.sqrt(float(tok.shape[0]))))
        side = max(1, side)

        for c_idx, k in enumerate(selected_words, start=1):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            ax.imshow(img)

            heat = normalize_heat_for_overlay(weights[:, int(k)], side)
            cmap = overlay_cmaps[(c_idx - 1) % len(overlay_cmaps)]
            ax.imshow(
                heat,
                cmap=cmap,
                alpha=0.62,
                interpolation="bilinear",
                extent=(0, img.shape[1], img.shape[0], 0),
            )

            if primary is not None and (c_idx - 1) < len(word_vecs):
                sim_primary = float(np.dot(word_vecs[c_idx - 1], primary))
                ax.set_title(f"k={int(k)}\nweight={hist[int(k)]:.3f}, map-sim={sim_primary:.2f}")
            else:
                ax.set_title(f"k={int(k)}\nweight={hist[int(k)]:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

        # Hide any unused map columns if fewer words were selected.
        for c_idx in range(1 + len(selected_words), n_cols):
            ax_blank = fig.add_subplot(gs[r_idx, c_idx])
            ax_blank.axis("off")

    fig.suptitle(
        f"Spatial Coherence of Diverse Vocabulary Activations (Top-{num_maps} per cell, {n_rows} cells)",
        y=0.995,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        "weight = BoVW histogram value for centroid k in this cell; "
        "map-sim = cosine similarity of this activation map to the first selected map in the same row.",
        ha="center",
        va="center",
        fontsize=9,
    )
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 7
# ---------------------------------------------------------------------------


def mean_hist_for_rows(hist: np.memmap, rows: np.ndarray, chunk: int = 4096) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((hist.shape[1],), dtype=np.float32)
    acc = np.zeros((hist.shape[1],), dtype=np.float64)
    for s in range(0, rows.size, chunk):
        sub = rows[s : s + chunk]
        acc += np.asarray(hist[sub], dtype=np.float64).sum(axis=0)
    return (acc / float(rows.size)).astype(np.float32)


def mean_within_l1(hist: np.memmap, rows: np.ndarray, rng: np.random.Generator, max_n: int = 240) -> float:
    if rows.size < 2:
        return float("nan")

    if rows.size > max_n:
        rows = rng.choice(rows, size=max_n, replace=False)

    X = np.asarray(hist[rows], dtype=np.float32)
    n = X.shape[0]

    total = 0.0
    count = 0
    for i in range(n - 1):
        d = np.abs(X[i + 1 :] - X[i]).sum(axis=1)
        total += float(d.sum())
        count += int(d.size)
    if count == 0:
        return float("nan")
    return total / float(count)


def centroid_activation_stats(
    hist: np.memmap,
    min_mass: Optional[float] = None,
    chunk: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-centroid activation sum and support count across cells."""
    _, k = hist.shape
    if min_mass is None:
        min_mass = 1.0 / float(max(1, 8 * k))

    activation = np.zeros(k, dtype=np.float64)
    support = np.zeros(k, dtype=np.int64)

    for s in range(0, hist.shape[0], chunk):
        e = min(s + chunk, hist.shape[0])
        block = np.asarray(hist[s:e], dtype=np.float32)
        activation += block.sum(axis=0)
        support += (block > float(min_mass)).sum(axis=0)

    return activation, support


def select_auto_near_far_pairs(
    centroids: np.ndarray,
    activation: np.ndarray,
    support: np.ndarray,
    top_active: int = 128,
    support_schedule: Sequence[int] = (200, 80, 20, 5, 1),
    near_gap_schedule: Sequence[int] = (24, 40, 64, 96),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int], int]:
    """Select one near and one far centroid pair automatically.

    Selection is deterministic:
      1) prioritize active centroids,
      2) enforce minimum support with staged fallback,
            3) choose near pair with high cosine under a small index-gap preference,
            4) choose far pair by minimum cosine.
    """
    k = centroids.shape[0]
    if k < 2:
        raise RuntimeError("Need at least 2 centroids for pair comparison")

    order = np.argsort(activation)[::-1]
    order = order[activation[order] > 0]
    if order.size == 0:
        order = np.arange(k, dtype=np.int64)
    if order.size > top_active:
        order = order[:top_active]

    chosen_support = 0
    candidate = order
    for thresh in support_schedule:
        eligible = order[support[order] >= int(thresh)]
        if eligible.size >= 4:
            candidate = eligible
            chosen_support = int(thresh)
            break

    if candidate.size < 4:
        eligible = np.where(support > 0)[0]
        if eligible.size >= 4:
            keep = np.argsort(activation[eligible])[::-1]
            limit = min(top_active, eligible.size)
            candidate = eligible[keep[:limit]]
            chosen_support = 1
        else:
            candidate = np.arange(k, dtype=np.int64)
            if candidate.size > top_active:
                keep = np.argsort(activation[candidate])[::-1]
                candidate = candidate[keep[:top_active]]
            chosen_support = 0

    if candidate.size < 2:
        raise RuntimeError("Failed to select enough centroid candidates")

    sim = pairwise_cosine_similarity(centroids[candidate])
    iu = np.triu_indices(candidate.size, k=1)
    if iu[0].size == 0:
        raise RuntimeError("No centroid pairs available after candidate selection")

    pair_vals = sim[iu]
    pair_gap = np.abs(candidate[iu[0]] - candidate[iu[1]])

    near_pos: Optional[int] = None
    for max_gap in near_gap_schedule:
        mask = pair_gap <= int(max_gap)
        if np.any(mask):
            local = np.where(mask)[0]
            near_pos = int(local[np.argmax(pair_vals[mask])])
            break

    if near_pos is None:
        near_pos = int(np.argmax(pair_vals))

    far_pos = int(np.argmin(pair_vals))

    if near_pos == far_pos and pair_vals.size > 1:
        mask = np.ones(pair_vals.size, dtype=bool)
        mask[near_pos] = False
        alt_far = int(np.argmin(pair_vals[mask]))
        far_pos = int(np.flatnonzero(mask)[alt_far])

    near_pair = (
        int(candidate[iu[0][near_pos]]),
        int(candidate[iu[1][near_pos]]),
    )
    far_pair = (
        int(candidate[iu[0][far_pos]]),
        int(candidate[iu[1][far_pos]]),
    )

    return candidate.astype(np.int64), sim, pair_vals, near_pair, far_pair, chosen_support


def top_rows_for_centroid(
    ctx: VizContext,
    word_idx: int,
    max_rows: int = 256,
    min_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    col = np.asarray(ctx.histograms[:, word_idx], dtype=np.float32)
    order = np.argsort(col)[::-1]
    if min_weight > 0:
        order = order[col[order] >= float(min_weight)]
    if max_rows > 0:
        order = order[:max_rows]
    return order.astype(np.int64, copy=False), col


def pair_histogram_metrics(
    ctx: VizContext,
    k_a: int,
    k_b: int,
    max_rows: int = 240,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int, int]:
    rows_a, col_a = top_rows_for_centroid(ctx, k_a, max_rows=max_rows, min_weight=0.0)
    rows_b, col_b = top_rows_for_centroid(ctx, k_b, max_rows=max_rows, min_weight=0.0)

    if rows_a.size > 0:
        rows_a = rows_a[col_a[rows_a] > 0]
    if rows_b.size > 0:
        rows_b = rows_b[col_b[rows_b] > 0]

    if rows_a.size == 0:
        rows_a = np.argsort(col_a)[::-1][: max(1, min(max_rows, col_a.shape[0]))]
    if rows_b.size == 0:
        rows_b = np.argsort(col_b)[::-1][: max(1, min(max_rows, col_b.shape[0]))]

    mean_a = np.asarray(ctx.histograms[rows_a], dtype=np.float32).mean(axis=0)
    mean_b = np.asarray(ctx.histograms[rows_b], dtype=np.float32).mean(axis=0)

    if float(np.std(mean_a)) < 1e-10 or float(np.std(mean_b)) < 1e-10:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(mean_a, mean_b)[0, 1])

    l1 = float(np.abs(mean_a - mean_b).sum())

    top_n = int(min(10, mean_a.shape[0]))
    bins_a = set(np.argsort(mean_a)[-top_n:].tolist())
    bins_b = set(np.argsort(mean_b)[-top_n:].tolist())
    overlap = float(len(bins_a & bins_b)) / float(max(1, top_n))

    return mean_a, mean_b, corr, l1, overlap, int(rows_a.size), int(rows_b.size)


def build_centroid_exemplar_canvas(
    ctx: VizContext,
    word_specs: Sequence[Tuple[int, str]],
    per_word: int = 3,
    thumb: int = 96,
    pad: int = 4,
) -> Tuple[Optional[np.ndarray], List[str]]:
    rows_imgs: List[List[np.ndarray]] = []
    row_labels: List[str] = []

    for word_idx, label_txt in word_specs:
        rows, _ = top_rows_for_centroid(ctx, word_idx, max_rows=300, min_weight=0.0)

        chosen: List[np.ndarray] = []
        for row in rows:
            rec = row_to_record(ctx, int(row))
            if rec is None:
                continue

            img = crop_and_resize_cell_image(rec)
            if img is None:
                continue

            thumb_img = np.asarray(
                Image.fromarray(img).resize((thumb, thumb), Image.BICUBIC),
                dtype=np.uint8,
            )
            chosen.append(thumb_img)
            if len(chosen) >= per_word:
                break

        while len(chosen) < per_word:
            chosen.append(np.full((thumb, thumb, 3), 235, dtype=np.uint8))

        rows_imgs.append(chosen)
        row_labels.append(label_txt)

    if not rows_imgs:
        return None, []

    n_rows = len(rows_imgs)
    width = per_word * thumb + (per_word - 1) * pad
    height = n_rows * thumb + (n_rows - 1) * pad

    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    for r in range(n_rows):
        y0 = r * (thumb + pad)
        for c in range(per_word):
            x0 = c * (thumb + pad)
            canvas[y0 : y0 + thumb, x0 : x0 + thumb] = rows_imgs[r][c]

    return canvas, row_labels


def build_centroid_patch_mosaic(
    ctx: VizContext,
    resolver: PatchTokenResolver,
    word_idx: int,
    grid: int = 4,
    patch_size: int = 40,
    max_rows: int = 300,
    patches_per_cell: int = 4,
) -> Tuple[np.ndarray, int]:
    """Build a patch mosaic with top patch-level activations for one centroid."""
    rows, _ = top_rows_for_centroid(ctx, word_idx, max_rows=max_rows, min_weight=0.0)

    scored_patches: List[Tuple[float, np.ndarray]] = []
    for row in rows:
        rec = row_to_record(ctx, int(row))
        if rec is None:
            continue

        img = crop_and_resize_cell_image(rec)
        if img is None:
            continue

        npz = resolver.resolve(rec)
        if npz is None:
            continue
        tok = load_patch_tokens(npz)
        if tok is None or tok.ndim != 2 or tok.shape[0] == 0:
            continue

        tok = l2_normalize(np.asarray(tok, dtype=np.float32), axis=1)
        weights = soft_assign(tok, ctx.centroids, beta=10.0)
        wk = weights[:, int(word_idx)]
        if wk.size == 0:
            continue

        side = int(round(math.sqrt(float(tok.shape[0]))))
        if side <= 0:
            continue
        local_patch = max(1, img.shape[0] // side)

        top_local = np.argsort(wk)[-patches_per_cell:][::-1]
        for patch_idx in top_local:
            pr = int(patch_idx) // side
            pc = int(patch_idx) % side
            y0 = pr * local_patch
            x0 = pc * local_patch
            y1 = min(y0 + local_patch, img.shape[0])
            x1 = min(x0 + local_patch, img.shape[1])
            patch = img[y0:y1, x0:x1, :]
            if patch.size == 0:
                continue
            scored_patches.append((float(wk[int(patch_idx)]), patch))

    scored_patches.sort(key=lambda x: x[0], reverse=True)

    target_n = grid * grid
    patches: List[np.ndarray] = [p for _, p in scored_patches[:target_n]]
    used_count = len(patches)

    while len(patches) < target_n:
        patches.append(np.full((patch_size, patch_size, 3), 235, dtype=np.uint8))

    mosaic = make_patch_mosaic(patches, grid=grid, patch_size=patch_size)
    return mosaic, used_count


def viz_07_category_prototypes(ctx: VizContext) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[7]

    top8 = top_populated_categories(ctx.labels_per_row, top_n=8)
    if len(top8) < 1:
        raise RuntimeError("Viz7: no labeled categories available")

    tab10 = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(17, 8.2))
    gs = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.38)

    for i, cat in enumerate(top8[:8]):
        r = i // 4
        c = i % 4
        ax = fig.add_subplot(gs[r, c])

        rows = np.where(ctx.labels_per_row == cat)[0]
        if rows.size == 0:
            ax.axis("off")
            continue

        mean_h = mean_hist_for_rows(ctx.histograms, rows)
        within = mean_within_l1(ctx.histograms, rows, ctx.rng)

        base_colors = ["#9a9a9a"] * mean_h.shape[0]
        top5 = np.argsort(mean_h)[-5:]
        cat_color = tab10(i % 10)
        for b in top5:
            base_colors[int(b)] = cat_color

        ax.bar(np.arange(mean_h.shape[0]), mean_h, color=base_colors, width=1.0, linewidth=0)
        ax.set_xlim(0, mean_h.shape[0] - 1)
        ax.set_xticks([0, 128, 256, 384, 511])
        ax.grid(False)
        ax.set_title(label_name(cat), pad=4)
        ax.set_xlabel(f"Within L1: {within:.3f}")

    # Hide unused panels if fewer than 8 categories.
    for i in range(len(top8), 8):
        r = i // 4
        c = i % 4
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.suptitle("Category Histogram Prototypes (Top-8 fMoW categories)", y=0.995, fontsize=13)
    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Visualization 8
# ---------------------------------------------------------------------------


def viz_08_cluster_similarity(ctx: VizContext, resolver: PatchTokenResolver) -> Path:
    out_path = ctx.output_dir / VIZ_FILENAMES[8]

    activation, support = centroid_activation_stats(ctx.histograms)
    (
        _,
        _,
        _,
        near_pair,
        far_pair,
        support_thresh,
    ) = select_auto_near_far_pairs(ctx.centroids, activation, support, top_active=128)

    near_sim = float(ctx.centroids[near_pair[0]] @ ctx.centroids[near_pair[1]])
    far_sim = float(ctx.centroids[far_pair[0]] @ ctx.centroids[far_pair[1]])
    info(
        "Viz8 selected pairs: "
        f"near=({near_pair[0]}, {near_pair[1]}) gap={abs(near_pair[0]-near_pair[1])} cos={near_sim:.3f}; "
        f"far=({far_pair[0]}, {far_pair[1]}) gap={abs(far_pair[0]-far_pair[1])} cos={far_sim:.3f}"
    )
    near_a_mosaic, near_a_used = build_centroid_patch_mosaic(ctx, resolver, near_pair[0], grid=4, patch_size=40)
    near_b_mosaic, near_b_used = build_centroid_patch_mosaic(ctx, resolver, near_pair[1], grid=4, patch_size=40)
    far_a_mosaic, far_a_used = build_centroid_patch_mosaic(ctx, resolver, far_pair[0], grid=4, patch_size=40)
    far_b_mosaic, far_b_used = build_centroid_patch_mosaic(ctx, resolver, far_pair[1], grid=4, patch_size=40)

    fig = plt.figure(figsize=(14.8, 13.6))
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.10)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.imshow(near_a_mosaic)
    ax1.imshow(near_b_mosaic)
    ax2.imshow(far_a_mosaic)
    ax3.imshow(far_b_mosaic)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    ax0.set_title(
        (
            f"Near pair A: k={near_pair[0]} | support={int(support[near_pair[0]])}\n"
            f"Top patches shown={near_a_used}"
        ),
        pad=8,
    )
    ax1.set_title(
        (
            f"Near pair B: k={near_pair[1]} | support={int(support[near_pair[1]])}\n"
            f"Top patches shown={near_b_used}"
        ),
        pad=8,
    )
    ax2.set_title(
        (
            f"Far pair A: k={far_pair[0]} | support={int(support[far_pair[0]])}\n"
            f"Top patches shown={far_a_used}"
        ),
        pad=8,
    )
    ax3.set_title(
        (
            f"Far pair B: k={far_pair[1]} | support={int(support[far_pair[1]])}\n"
            f"Top patches shown={far_b_used}"
        ),
        pad=8,
    )

    fig.suptitle(
        (
            "Patch-Level Near-vs-Far Cluster Comparison (No Graphs)"
            f" | near=({near_pair[0]}, {near_pair[1]}) cos={near_sim:.3f}"
            f" | far=({far_pair[0]}, {far_pair[1]}) cos={far_sim:.3f}"
            f" | candidate support >= {support_thresh}"
        ),
        y=0.995,
        fontsize=13,
    )

    save_figure(fig, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_visualization(
    viz_num: int,
    fn,
    ctx: VizContext,
    resolver: PatchTokenResolver,
    skip: set[int],
    outputs: Dict[int, Path],
) -> None:
    if viz_num in skip:
        info(f"Skipping visualization {viz_num} via --skip")
        return

    try:
        info(f"Generating visualization {viz_num} -> {VIZ_FILENAMES[viz_num]}")
        if viz_num in (1, 2, 3, 5, 6, 8):
            out = fn(ctx, resolver)
        else:
            out = fn(ctx)
        outputs[viz_num] = out
        info(f"Saved visualization {viz_num}: {out}")
    except Exception as exc:
        warn(f"Visualization {viz_num} failed: {exc}")
        traceback.print_exc()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BoVW visualization figures (1-8).")

    p.add_argument("--manifest", type=str, default="data/fmow_manifest_train.json", help="Path to manifest.json")
    p.add_argument("--histogram-dir", type=str, default="outputs/bovw_histograms", help="Path to histogram directory")
    p.add_argument("--vocab-dir", type=str, default="outputs/bovw_vocabulary", help="Path to vocabulary directory")
    p.add_argument("--patch-token-dir", type=str, default="outputs/patch_tokens_bovw", help="Path to patch token directory")
    p.add_argument(
        "--dinov3-embed-dir",
        type=str,
        default="outputs/preprocess_cache_dinov3",
        help="Path to pooled DINO cache directory",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/bovw_checkpoints",
        help="Path to BoVW checkpoints directory",
    )
    p.add_argument("--output-dir", type=str, default="outputs/visualizations", help="Output directory")
    p.add_argument("--cell-labels", type=str, default="", help="Path to cell_labels.npy")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated visualization numbers to skip (e.g. '5' or '2,5,8')",
    )

    return p.parse_args()


def main() -> int:
    setup_matplotlib_style()
    args = parse_args()

    skip = parse_skip(args.skip)

    ctx = build_context(args)
    resolver = PatchTokenResolver(
        patch_token_dir=ctx.patch_token_dir,
        workspace_root=ctx.workspace_root,
        weights_path=ctx.manifest_meta.get("weights_path", None),
    )

    info("Context loaded:")
    info(f"  Manifest records: {len(ctx.records)}")
    info(f"  Histograms: {ctx.histograms.shape}")
    info(f"  Centroids: {ctx.centroids.shape}")
    info(f"  Checkpoint dir: {ctx.checkpoint_dir}")
    info(f"  Output dir: {ctx.output_dir}")

    outputs: Dict[int, Path] = {}

    run_visualization(1, viz_01_histogram_gallery, ctx, resolver, skip, outputs)
    run_visualization(2, viz_02_vocabulary_gallery, ctx, resolver, skip, outputs)
    run_visualization(3, viz_03_similarity_matrix, ctx, resolver, skip, outputs)
    run_visualization(4, viz_04_centroid_tsne, ctx, resolver, skip, outputs)
    run_visualization(5, viz_05_prediction_convergence, ctx, resolver, skip, outputs)
    run_visualization(6, viz_06_spatial_heatmaps, ctx, resolver, skip, outputs)
    run_visualization(7, viz_07_category_prototypes, ctx, resolver, skip, outputs)
    run_visualization(8, viz_08_cluster_similarity, ctx, resolver, skip, outputs)

    print("\n=== Visualization Summary ===")
    if outputs:
        for k in sorted(outputs):
            p = outputs[k]
            if p.exists():
                size_mb = p.stat().st_size / (1024.0 * 1024.0)
                print(f"  [{k}] {p}  ({size_mb:.2f} MB)")
            else:
                print(f"  [{k}] {p}  (missing)")
    else:
        print("  No visualizations were generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
