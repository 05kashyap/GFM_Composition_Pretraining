#!/usr/bin/env python3
"""DynamicVis-backed object detection on LEVIR-ship.

This script adapts a Faster R-CNN detection pipeline to use the DynamicVis
backbone (initialized from BoVW checkpoints) and trains/evaluates on a
YOLO-label-formatted LEVIR-ship dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from eval.adapters.prithvi_v2_adapter import PrithviEncoderV2

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "eval" / "object-det"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "object_detection"
DEFAULT_BACKBONE_WEIGHTS = REPO_ROOT / "outputs" / "bovw_training_8262" / "epoch_20.pth"
DEFAULT_PRITHVI_WEIGHTS = REPO_ROOT / "weights" / "Prithvi_EO_V2_600M.pt"
DEFAULT_DYNVIS_CONFIG = (
    REPO_ROOT / "configs_dynamicvis" / "fmow_pretrain" / "dynamicvis_b_fmow_s3_pretrain.py"
)

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install_dynamicvis_stubs() -> None:
    """Install minimal stubs when full mmdet/mmseg stacks are unavailable."""
    if "mmdet" not in sys.modules:
        mmdet_mod = types.ModuleType("mmdet")
        mmdet_models_mod = types.ModuleType("mmdet.models")
        mmdet_structures_mod = types.ModuleType("mmdet.structures")
        mmdet_structures_bbox_mod = types.ModuleType("mmdet.structures.bbox")

        def _nlc_to_nchw(x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
            h, w = hw_shape
            b, n, c = x.shape
            if n != h * w:
                raise ValueError(f"nlc_to_nchw: N={n} must equal H*W={h * w}")
            return x.transpose(1, 2).contiguous().view(b, c, h, w)

        def _nchw_to_nlc(x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            return x.view(b, c, h * w).transpose(1, 2).contiguous()

        class _FPNStub(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("FPN requires full mmdet/mmcv; unsupported in lightweight setup.")

        def _bbox2roi_stub(*args, **kwargs):
            raise RuntimeError("bbox2roi requires full mmdet; unsupported in lightweight setup.")

        mmdet_models_mod.nlc_to_nchw = _nlc_to_nchw
        mmdet_models_mod.nchw_to_nlc = _nchw_to_nlc
        mmdet_models_mod.FPN = _FPNStub
        mmdet_structures_bbox_mod.bbox2roi = _bbox2roi_stub

        sys.modules.setdefault("mmdet", mmdet_mod)
        sys.modules.setdefault("mmdet.models", mmdet_models_mod)
        sys.modules.setdefault("mmdet.structures", mmdet_structures_mod)
        sys.modules.setdefault("mmdet.structures.bbox", mmdet_structures_bbox_mod)

    if "mmseg" not in sys.modules:
        mmseg_mod = types.ModuleType("mmseg")
        mmseg_models_mod = types.ModuleType("mmseg.models")
        mmseg_models_backbones_mod = types.ModuleType("mmseg.models.backbones")
        mmseg_models_backbones_unet_mod = types.ModuleType("mmseg.models.backbones.unet")
        mmseg_models_utils_mod = types.ModuleType("mmseg.models.utils")

        class _MMSEGStubBlock(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("mmseg blocks are not supported in lightweight setup.")

        mmseg_models_backbones_unet_mod.BasicConvBlock = _MMSEGStubBlock
        mmseg_models_utils_mod.UpConvBlock = _MMSEGStubBlock

        sys.modules.setdefault("mmseg", mmseg_mod)
        sys.modules.setdefault("mmseg.models", mmseg_models_mod)
        sys.modules.setdefault("mmseg.models.backbones", mmseg_models_backbones_mod)
        sys.modules.setdefault("mmseg.models.backbones.unet", mmseg_models_backbones_unet_mod)
        sys.modules.setdefault("mmseg.models.utils", mmseg_models_utils_mod)

    if "mmpretrain.models.multimodal" not in sys.modules:
        mmpr_mm_stub = types.ModuleType("mmpretrain.models.multimodal")
        mmpr_mm_stub.__all__ = []
        sys.modules.setdefault("mmpretrain.models.multimodal", mmpr_mm_stub)


def _report_mamba_fast_path_status() -> None:
    missing = []
    if importlib.util.find_spec("mamba_ssm") is None:
        missing.append("mamba-ssm")
    if importlib.util.find_spec("causal_conv1d") is None:
        missing.append("causal-conv1d")

    if missing:
        warnings.warn(
            "DynamicVis fast path is unavailable because dependencies are missing: "
            f"{', '.join(missing)}.",
            stacklevel=2,
        )


def _extract_backbone_cfg(cfg: Any) -> Dict[str, Any]:
    model_cfg = cfg.get("model", None) if hasattr(cfg, "get") else getattr(cfg, "model", None)
    if isinstance(model_cfg, dict):
        return dict(model_cfg.get("backbone", {}))
    return {}


def parse_yolo_label(txt_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes: List[List[float]] = []
    labels: List[int] = []

    if not txt_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls, cx, cy, w, h = map(float, parts[:5])

            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h

            x1 = float(np.clip(x1, 0, max(img_w - 1, 0)))
            y1 = float(np.clip(y1, 0, max(img_h - 1, 0)))
            x2 = float(np.clip(x2, 0, max(img_w - 1, 0)))
            y2 = float(np.clip(y2, 0, max(img_h - 1, 0)))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            # Reserve label 0 for background as expected by Faster R-CNN.
            labels.append(int(cls) + 1)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class LEVIRShipDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        img_size: int,
        augment: bool = False,
        max_items: int | None = None,
        seed: int = 42,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = int(img_size)
        self.augment = augment

        all_paths = sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )
        if max_items is not None and len(all_paths) > max_items:
            rng = random.Random(seed)
            rng.shuffle(all_paths)
            all_paths = sorted(all_paths[:max_items])

        self.img_paths = all_paths

        if not self.img_paths:
            raise RuntimeError(f"No images found in {images_dir}")

        print(f"Loaded {len(self.img_paths)} images from {images_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        h, w = img_np.shape[:2]
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes, labels = parse_yolo_label(lbl_path, w, h)

        if self.augment and boxes.shape[0] > 0 and random.random() < 0.5:
            img_np = img_np[:, ::-1, :].copy()
            boxes[:, [0, 2]] = (w - 1) - boxes[:, [2, 0]]

        image_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float()

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        area_t = torch.zeros((boxes_t.shape[0],), dtype=torch.float32)
        if boxes_t.shape[0] > 0:
            area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "area": area_t,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return images, targets


class DynamicVisDetectionBackbone(nn.Module):
    """DynamicVis feature extractor + torchvision FPN for Faster R-CNN."""

    def __init__(
        self,
        dynamicvis_config: str,
        img_size: int,
        backbone_checkpoint: str | None = None,
        fpn_out_channels: int = 256,
    ):
        super().__init__()

        dynamicvis_root = Path(os.environ.get("DYNAMICVIS_ROOT", REPO_ROOT / "architectures" / "DynamicVis"))
        if not dynamicvis_root.exists():
            raise FileNotFoundError(
                f"DynamicVis repo not found at: {dynamicvis_root}. "
                "Set DYNAMICVIS_ROOT or clone into architectures/DynamicVis."
            )

        _report_mamba_fast_path_status()
        if str(dynamicvis_root) not in sys.path:
            sys.path.insert(0, str(dynamicvis_root))

        has_bundled_mmdet = (dynamicvis_root / "mmdet" / "__init__.py").exists()
        if not has_bundled_mmdet:
            _install_dynamicvis_stubs()

        from mmengine import Config
        from dynamicvis.models.models import DynamicVisBackbone  # type: ignore

        cfg = Config.fromfile(dynamicvis_config)
        backbone_cfg = _extract_backbone_cfg(cfg)
        backbone_cfg.pop("type", None)
        backbone_cfg.setdefault("arch", "b")
        backbone_cfg["img_size"] = int(img_size)
        backbone_cfg.setdefault("in_channels", 3)
        backbone_cfg["out_indices"] = (0, 1, 2, 3)
        backbone_cfg["out_type"] = "featmap"

        self.backbone = DynamicVisBackbone(**backbone_cfg)
        in_channels_list = list(self.backbone.c_embed_dims)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=fpn_out_channels)
        self.out_channels = int(fpn_out_channels)

        if backbone_checkpoint:
            self._load_backbone_weights(backbone_checkpoint)

    def _load_backbone_weights(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt.get("model", ckpt)
        else:
            state_dict = ckpt

        if not isinstance(state_dict, dict):
            raise ValueError("Unexpected checkpoint format for DynamicVis backbone loading.")

        expected_keys = set(self.backbone.state_dict().keys())

        normalized: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            normalized[nk] = v

        filtered: Dict[str, torch.Tensor] = {}
        for k, v in normalized.items():
            if k.startswith("backbone."):
                candidate = k[len("backbone.") :]
            elif ".backbone." in k:
                candidate = k.split(".backbone.", 1)[1]
            else:
                candidate = k

            if candidate in expected_keys:
                filtered[candidate] = v

        if not filtered:
            raise RuntimeError(
                "No DynamicVis backbone weights were found in checkpoint. "
                "Expected backbone.* keys or direct backbone state_dict keys."
            )

        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(
            "Backbone load summary: "
            f"total_keys={len(state_dict)}, retained={len(filtered)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError("Expected DynamicVis featmap output as list/tuple of feature maps.")

        feat_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
            (str(i), feat) for i, feat in enumerate(feats)
        )
        return self.fpn(feat_dict)


class PrithviDetectionBackbone(nn.Module):
    """Prithvi v2 feature extractor + torchvision FPN for Faster R-CNN."""

    def __init__(
        self,
        prithvi_checkpoint: str,
        img_size: int,
        fpn_out_channels: int = 256,
        layer_indices: List[int] | None = None,
    ):
        super().__init__()
        self.layer_indices = layer_indices or [-4, -3, -2, -1]
        self.encoder = PrithviEncoderV2(
            model_path=prithvi_checkpoint,
            embedding_dim=512,
            img_size=img_size,
            in_chans=3,
        )
        self.out_channels = [self.encoder.feature_dim] * len(self.layer_indices)
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.out_channels, out_channels=fpn_out_channels)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        feats = self.encoder.forward_feature_maps(x, layer_indices=self.layer_indices)
        feat_dict: OrderedDict[str, torch.Tensor] = OrderedDict(
            (str(i), feat) for i, feat in enumerate(feats)
        )
        return self.fpn(feat_dict)


def build_model(args: argparse.Namespace) -> FasterRCNN:
    if args.model_type == "dynamicvis":
        backbone = DynamicVisDetectionBackbone(
            dynamicvis_config=args.dynamicvis_config,
            img_size=args.img_size,
            backbone_checkpoint=args.backbone_checkpoint,
            fpn_out_channels=args.fpn_out_channels,
        )
    else:
        backbone = PrithviDetectionBackbone(
            prithvi_checkpoint=args.backbone_checkpoint,
            img_size=args.img_size,
            fpn_out_channels=args.fpn_out_channels,
        )

    anchor_gen = AnchorGenerator(
        sizes=((16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=roi_pool,
        min_size=args.img_size,
        max_size=args.img_size,
    )
    return model


def train_one_epoch(
    model: FasterRCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

        if step % log_every == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / max(len(loader), 1)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


@torch.no_grad()
def evaluate_map(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    iou_thresh: float,
) -> Dict[str, float]:
    model.eval()

    all_scores: List[float] = []
    all_matches: List[int] = []
    total_gt = 0

    for images, targets in loader:
        images_gpu = [img.to(device, non_blocking=True) for img in images]
        preds = model(images_gpu)

        for pred, tgt in zip(preds, targets):
            pred_boxes = pred["boxes"].detach().cpu()
            pred_scores = pred["scores"].detach().cpu()
            pred_labels = pred["labels"].detach().cpu()

            gt_boxes = tgt["boxes"].detach().cpu()
            gt_labels = tgt["labels"].detach().cpu()

            gt_mask = gt_labels == 1
            gt_boxes = gt_boxes[gt_mask]
            total_gt += int(gt_boxes.shape[0])

            pred_mask = pred_labels == 1
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]

            if pred_boxes.numel() == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            matched_gt: set[int] = set()

            for pbox, pscore in zip(pred_boxes, pred_scores):
                if gt_boxes.numel() == 0:
                    all_scores.append(float(pscore.item()))
                    all_matches.append(0)
                    continue

                ious = box_iou(pbox.unsqueeze(0), gt_boxes).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)
                best_idx = int(max_idx.item())

                if float(max_iou.item()) >= iou_thresh and best_idx not in matched_gt:
                    all_scores.append(float(pscore.item()))
                    all_matches.append(1)
                    matched_gt.add(best_idx)
                else:
                    all_scores.append(float(pscore.item()))
                    all_matches.append(0)

    if not all_scores or total_gt == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0}

    scores = np.asarray(all_scores, dtype=np.float32)
    matches = np.asarray(all_matches, dtype=np.int32)
    order = np.argsort(-scores)
    matches = matches[order]

    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)

    recall = tp / (total_gt + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    ap = compute_ap(recall, precision)

    final_precision = float(precision[-1]) if precision.size else 0.0
    final_recall = float(recall[-1]) if recall.size else 0.0

    return {"ap": ap, "precision": final_precision, "recall": final_recall}


@torch.no_grad()
def evaluate_multi_iou(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    iou_thresholds: Sequence[float],
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    for thr in iou_thresholds:
        metrics = evaluate_map(model, loader, device, float(thr))
        results[float(thr)] = metrics
        print(
            f"mAP@{thr:.2f}: {metrics['ap']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )
    return results


@torch.no_grad()
def visualize_predictions(
    model: FasterRCNN,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    num_images: int,
    score_thresh: float,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    shown = 0
    for images, targets in loader:
        preds = model([img.to(device, non_blocking=True) for img in images])

        for image, pred, tgt in zip(images, preds, targets):
            if shown >= num_images:
                return

            img_np = image.permute(1, 2, 0).detach().cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(img_np)

            for box in tgt["boxes"]:
                x1, y1, x2, y2 = box.tolist()
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
                ax.add_patch(rect)

            for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                if int(label.item()) != 1 or float(score.item()) < score_thresh:
                    continue

                x1, y1, x2, y2 = box.detach().cpu().tolist()
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 3, 0), f"{score.item():.2f}", color="red", fontsize=8)

            ax.set_axis_off()
            ax.set_title(f"Sample {shown + 1} | GT: {len(tgt['boxes'])}")
            fig.tight_layout()
            save_path = out_dir / f"prediction_{shown + 1:03d}.png"
            fig.savefig(save_path, dpi=200)
            plt.close(fig)

            shown += 1


def save_detector_checkpoint(model: FasterRCNN, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, output_path)


def load_detector_checkpoint(model: FasterRCNN, checkpoint_path: str, device: torch.device) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unexpected detector checkpoint format.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded detector checkpoint: {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")


def parse_iou_thresholds(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val <= 0.0 or val >= 1.0:
            raise ValueError("IoU thresholds must be in (0, 1).")
        values.append(val)
    if not values:
        raise ValueError("At least one IoU threshold must be provided.")
    return values


def build_dataloaders(args: argparse.Namespace):
    train_ds = LEVIRShipDataset(
        images_dir=Path(args.data_root) / "train" / "images",
        labels_dir=Path(args.data_root) / "train" / "labels",
        img_size=args.img_size,
        augment=not args.no_augment,
        max_items=args.max_train,
        seed=args.seed,
    )
    val_ds = LEVIRShipDataset(
        images_dir=Path(args.data_root) / "val" / "images",
        labels_dir=Path(args.data_root) / "val" / "labels",
        img_size=args.img_size,
        augment=False,
        max_items=args.max_val,
        seed=args.seed,
    )
    test_ds = LEVIRShipDataset(
        images_dir=Path(args.data_root) / "test" / "images",
        labels_dir=Path(args.data_root) / "test" / "labels",
        img_size=args.img_size,
        augment=False,
        max_items=args.max_test,
        seed=args.seed,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicVis LEVIR-ship object detection")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument(
        "--model-type",
        type=str,
        default="dynamicvis",
        choices=["dynamicvis", "prithvi", "prithvi2", "prithvi_v2"],
    )

    parser.add_argument("--dynamicvis-config", type=str, default=str(DEFAULT_DYNVIS_CONFIG))
    parser.add_argument("--backbone-checkpoint", type=str, default=str(DEFAULT_BACKBONE_WEIGHTS))
    parser.add_argument("--detector-checkpoint", type=str, default="")

    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--fpn-out-channels", type=int, default=256)

    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iou-thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--num-visualize", type=int, default=5)

    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)

    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.model_type in {"prithvi", "prithvi2", "prithvi_v2"} and args.backbone_checkpoint == str(DEFAULT_BACKBONE_WEIGHTS):
        args.backbone_checkpoint = str(DEFAULT_PRITHVI_WEIGHTS)

    if args.dry_run:
        print("Dry-run configuration:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        return

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    for split in ("train", "val", "test"):
        images_dir = data_root / split / "images"
        labels_dir = data_root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected split folders at {images_dir} and {labels_dir}"
            )

    if args.model_type == "dynamicvis" and not Path(args.dynamicvis_config).is_file():
        raise FileNotFoundError(f"DynamicVis config not found: {args.dynamicvis_config}")

    if not Path(args.backbone_checkpoint).is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_checkpoint}")

    iou_thresholds = parse_iou_thresholds(args.iou_thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "predictions"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = build_model(args).to(device)

    if not args.train_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    if args.detector_checkpoint:
        load_detector_checkpoint(model, args.detector_checkpoint, device)

    if not args.eval_only:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        best_val_ap = -1.0
        best_ckpt_path = output_dir / "best_detector.pth"

        print("\n===== Training =====")
        for epoch in range(1, args.num_epochs + 1):
            avg_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                log_every=args.log_every,
            )
            print(f"Epoch {epoch} finished | Avg loss: {avg_loss:.4f}")

            val_metrics = evaluate_map(model, val_loader, device, iou_thresh=0.5)
            print(
                f"Validation mAP@0.50: {val_metrics['ap']:.4f} | "
                f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}"
            )

            if val_metrics["ap"] > best_val_ap:
                best_val_ap = val_metrics["ap"]
                save_detector_checkpoint(model, best_ckpt_path)
                print(f"Saved best detector checkpoint -> {best_ckpt_path}")

        final_ckpt_path = output_dir / "last_detector.pth"
        save_detector_checkpoint(model, final_ckpt_path)
        print(f"Saved last detector checkpoint -> {final_ckpt_path}")

        if best_ckpt_path.exists():
            load_detector_checkpoint(model, str(best_ckpt_path), device)

    print("\n===== Test Evaluation =====")
    multi_iou_results = evaluate_multi_iou(
        model=model,
        loader=test_loader,
        device=device,
        iou_thresholds=iou_thresholds,
    )

    results_path = output_dir / "metrics.txt"
    with results_path.open("w", encoding="utf-8") as f:
        for iou, metrics in multi_iou_results.items():
            line = (
                f"mAP@{iou:.2f}: {metrics['ap']:.6f}, "
                f"precision: {metrics['precision']:.6f}, "
                f"recall: {metrics['recall']:.6f}\n"
            )
            f.write(line)
    print(f"Saved metrics -> {results_path}")

    visualize_predictions(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=vis_dir,
        num_images=args.num_visualize,
        score_thresh=args.score_thresh,
    )
    print(f"Saved prediction visualizations -> {vis_dir}")


if __name__ == "__main__":
    main()
