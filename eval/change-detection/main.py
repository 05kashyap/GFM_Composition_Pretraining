#!/usr/bin/env python3
"""DynamicVis-backed change detection evaluation on LEVIR-CD.

This script adapts an existing Siamese CD pipeline to use the repository's
DynamicVis backbone as the feature extractor.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "eval" / "LEVIR CD"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "change_detection"
DEFAULT_BACKBONE_WEIGHTS = REPO_ROOT / "outputs" / "bovw_training_8262" / "epoch_20.pth"
DEFAULT_DYNVIS_CONFIG = (
    REPO_ROOT
    / "architectures"
    / "DynamicVis"
    / "configs_DynamicVis"
    / "LEVIR-CD"
    / "dynamicvis_b_2X_levircd_mamba.py"
)


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

        mmdet_models_mod.nlc_to_nchw = _nlc_to_nchw
        mmdet_models_mod.nchw_to_nlc = _nchw_to_nlc

        class _FPNStub(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("FPN requires full mmdet/mmcv; unsupported in lightweight setup.")

        mmdet_models_mod.FPN = _FPNStub

        def _bbox2roi_stub(*args, **kwargs):
            raise RuntimeError("bbox2roi requires full mmdet; unsupported in lightweight setup.")

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


class LEVIRCDDataset(Dataset):
    """LEVIR-CD dataset from pre-extracted patches."""

    def __init__(
        self,
        patches_a: List[np.ndarray],
        patches_b: List[np.ndarray],
        patches_label: List[np.ndarray],
        augment: bool = False,
        mean: List[float] | Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: List[float] | Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.patches_a = patches_a
        self.patches_b = patches_b
        self.patches_label = patches_label
        self.augment = augment
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.patches_a)

    def _augment(self, img_a: np.ndarray, img_b: np.ndarray, label: np.ndarray):
        if random.random() > 0.5:
            img_a = np.fliplr(img_a).copy()
            img_b = np.fliplr(img_b).copy()
            label = np.fliplr(label).copy()
        if random.random() > 0.5:
            img_a = np.flipud(img_a).copy()
            img_b = np.flipud(img_b).copy()
            label = np.flipud(label).copy()
        k = random.randint(0, 3)
        img_a = np.rot90(img_a, k).copy()
        img_b = np.rot90(img_b, k).copy()
        label = np.rot90(label, k).copy()
        return img_a, img_b, label

    def __getitem__(self, idx: int):
        img_a = self.patches_a[idx]
        img_b = self.patches_b[idx]
        label = self.patches_label[idx]

        if self.augment:
            img_a, img_b, label = self._augment(img_a, img_b, label)

        img_a = self.normalize(self.to_tensor(img_a))
        img_b = self.normalize(self.to_tensor(img_b))
        label = torch.from_numpy((label > 127).astype(np.float32)).unsqueeze(0)
        return img_a, img_b, label


def extract_patches(
    img_a_path: str,
    img_b_path: str,
    label_path: str,
    patch_size: int,
    stride: int,
):
    img_a = np.array(Image.open(img_a_path).convert("RGB"))
    img_b = np.array(Image.open(img_b_path).convert("RGB"))
    label = np.array(Image.open(label_path).convert("L"))

    h, w = img_a.shape[:2]
    patches_a, patches_b, patches_l = [], [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches_a.append(img_a[y : y + patch_size, x : x + patch_size])
            patches_b.append(img_b[y : y + patch_size, x : x + patch_size])
            patches_l.append(label[y : y + patch_size, x : x + patch_size])

    return patches_a, patches_b, patches_l


def collect_patches(dir_a: Path, dir_b: Path, dir_label: Path, patch_size: int, stride: int):
    filenames = sorted(
        f for f in os.listdir(dir_a) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    )

    all_a, all_b, all_l = [], [], []
    for fname in filenames:
        pa, pb, pl = extract_patches(
            str(dir_a / fname),
            str(dir_b / fname),
            str(dir_label / fname),
            patch_size=patch_size,
            stride=stride,
        )
        all_a.extend(pa)
        all_b.extend(pb)
        all_l.extend(pl)

    return all_a, all_b, all_l, len(filenames)


def build_datasets(args: argparse.Namespace):
    print("Building datasets using predefined LEVIR-CD splits...")

    train_a, train_b, train_l, train_n = collect_patches(
        Path(args.data_root) / "train" / "A",
        Path(args.data_root) / "train" / "B",
        Path(args.data_root) / "train" / "label",
        args.patch_size,
        args.stride,
    )
    val_a, val_b, val_l, val_n = collect_patches(
        Path(args.data_root) / "val" / "A",
        Path(args.data_root) / "val" / "B",
        Path(args.data_root) / "val" / "label",
        args.patch_size,
        args.stride,
    )
    test_a, test_b, test_l, test_n = collect_patches(
        Path(args.data_root) / "test" / "A",
        Path(args.data_root) / "test" / "B",
        Path(args.data_root) / "test" / "label",
        args.patch_size,
        args.stride,
    )

    print(f"  Train: {train_n} images -> {len(train_a)} patches")
    print(f"  Val:   {val_n} images -> {len(val_a)} patches")
    print(f"  Test:  {test_n} images -> {len(test_a)} patches")

    train_dataset = LEVIRCDDataset(train_a, train_b, train_l, augment=not args.no_augment)
    val_dataset = LEVIRCDDataset(val_a, val_b, val_l, augment=False)
    test_dataset = LEVIRCDDataset(test_a, test_b, test_l, augment=False)

    return train_dataset, val_dataset, test_dataset


def build_dataloaders(train_ds, val_ds, test_ds, args: argparse.Namespace):
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


class BackboneWrapper(nn.Module):
    """DynamicVis backbone wrapper returning 4 feature maps for the CD neck."""

    def __init__(self, dynamicvis_config: str, img_size: int, backbone_checkpoint: str | None = None):
        super().__init__()
        self.expected_img_size = int(img_size)

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
        # Force runtime img_size to match extracted patch resolution.
        # The upstream LEVIR-CD config uses 1024, but this pipeline may run
        # on 512x512 patches, which must align with positional embeddings.
        backbone_cfg["img_size"] = self.expected_img_size
        backbone_cfg.setdefault("in_channels", 3)
        backbone_cfg["out_indices"] = (0, 1, 2, 3)
        backbone_cfg["out_type"] = "featmap"

        self.backbone = DynamicVisBackbone(**backbone_cfg)
        self.out_channels = list(self.backbone.c_embed_dims)

        if backbone_checkpoint:
            self._load_backbone_weights(backbone_checkpoint)

    def _load_backbone_weights(self, checkpoint_path: str) -> None:
        checkpoint_path = str(checkpoint_path)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
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
            raise ValueError("Unexpected checkpoint format for DynamicVis weights.")

        expected_keys = set(self.backbone.state_dict().keys())

        normalized_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            normalized_state[nk] = v

        drop_tokens = (
            "decoder",
            "decode_head",
            "bbox_head",
            "mask_head",
            "roi_head",
            "neck",
            "auxiliary_head",
            "seg_head",
            "panoptic_head",
            "cls_head",
            "category_embedding",
        )
        drop_prefixes = (
            "head.",
            "aux_cls_head.",
            "loss_fn.",
            "optimizer.",
            "scheduler.",
            "data_preprocessor.",
            "ema_model.",
        )

        filtered: Dict[str, torch.Tensor] = {}
        dropped_non_backbone = 0
        for k, v in normalized_state.items():
            if k.startswith(drop_prefixes) or any(t in k for t in drop_tokens):
                dropped_non_backbone += 1
                continue

            if k.startswith("backbone."):
                candidate_key = k[len("backbone.") :]
            elif ".backbone." in k:
                candidate_key = k.split(".backbone.", 1)[1]
            else:
                candidate_key = k

            if candidate_key in expected_keys:
                filtered[candidate_key] = v

        if not filtered:
            raise RuntimeError(
                "No DynamicVis backbone weights were found in checkpoint. "
                "Expected backbone.* keys or direct backbone state_dict keys."
            )

        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        loaded = len(filtered) - len(unexpected)
        print(
            "DynamicVis backbone load summary: "
            f"checkpoint_keys={len(state_dict)}, retained_backbone_keys={len(filtered)}, "
            f"loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}, "
            f"dropped_non_backbone={dropped_non_backbone}"
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        if h != self.expected_img_size or w != self.expected_img_size:
            raise ValueError(
                "Input resolution does not match DynamicVis backbone img_size: "
                f"got ({h}, {w}), expected "
                f"({self.expected_img_size}, {self.expected_img_size}). "
                "Set --patch-size to the same value used to build this model."
            )
        return self.backbone(x)


class FPNNeck(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels_list]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        lat = [l(f) for l, f in zip(self.lateral, features)]
        fused = lat[-1]
        for i in range(len(lat) - 2, -1, -1):
            fused = F.interpolate(fused, size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + lat[i]
        return self.out_conv(fused)


class ChangeDetectionHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.pred = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(feat_a - feat_b)
        x = self.mlp(diff)
        x = self.upsample(x)
        x = self.pred(x)
        return x


class ChangeDetectionModel(nn.Module):
    def __init__(
        self,
        dynamicvis_config: str,
        img_size: int,
        fpn_out_channels: int,
        backbone_checkpoint: str | None,
    ):
        super().__init__()
        self.backbone = BackboneWrapper(
            dynamicvis_config=dynamicvis_config,
            img_size=img_size,
            backbone_checkpoint=backbone_checkpoint,
        )
        self.fpn = FPNNeck(self.backbone.out_channels, out_channels=fpn_out_channels)
        self.head = ChangeDetectionHead(in_channels=fpn_out_channels)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        feats_a = self.backbone(img_a)
        feats_b = self.backbone(img_b)
        fused_a = self.fpn(feats_a)
        fused_b = self.fpn(feats_b)
        return self.head(fused_a, fused_b)


def set_backbone_trainable(model: ChangeDetectionModel, trainable: bool) -> None:
    """Enable/disable gradient updates for the DynamicVis backbone."""
    for p in model.backbone.parameters():
        p.requires_grad = trainable


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = 1 - (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_metrics(preds_bin: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    preds_bin = preds_bin.float()
    targets = targets.float()

    tp = (preds_bin * targets).sum().item()
    fp = (preds_bin * (1 - targets)).sum().item()
    fn = ((1 - preds_bin) * targets).sum().item()

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    iou = tp / (tp + fp + fn + smooth)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
    train_backbone: bool,
):
    model.train()
    if not train_backbone:
        # Keep backbone in eval mode when frozen to avoid training-time
        # stochastic routing branches inside DynamicVis.
        model.backbone.eval()
    total_loss = 0.0
    all_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for i, (img_a, img_b, label) in enumerate(loader):
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(img_a, img_b)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).float()
            m = compute_metrics(preds_bin, label)

        if i % log_every == 0:
            print(f"Batch [{i}/{len(loader)}] | Loss: {loss.item():.4f} | F1: {m['f1']:.4f}")

        for k in all_metrics:
            all_metrics[k] += m[k]

    n = max(len(loader), 1)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for img_a, img_b, label in loader:
        img_a = img_a.to(device, non_blocking=True)
        img_b = img_b.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        logits = model(img_a, img_b)
        loss = criterion(logits, label)
        total_loss += loss.item()

        preds_bin = (torch.sigmoid(logits) > 0.5).float()
        m = compute_metrics(preds_bin, label)
        for k in all_metrics:
            all_metrics[k] += m[k]

    n = max(len(loader), 1)
    return total_loss / n, {k: v / n for k, v in all_metrics.items()}


def plot_history(history: Dict[str, List[float]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="coral")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_f1"], label="Train F1", color="steelblue")
    axes[1].plot(history["val_f1"], label="Val F1", color="coral")
    axes[1].set_title("F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


@torch.no_grad()
def visualize_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mean: List[float],
    std: List[float],
    num_samples: int,
    save_path: Path,
) -> None:
    model.eval()

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    img_a_batch, img_b_batch, label_batch = next(iter(test_loader))
    img_a_batch = img_a_batch[:num_samples].to(device)
    img_b_batch = img_b_batch[:num_samples].to(device)
    label_batch = label_batch[:num_samples]

    logits = model(img_a_batch, img_b_batch)
    preds_bin = (torch.sigmoid(logits) > 0.5).cpu().float()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["Image A (Before)", "Image B (After)", "Ground Truth", "Prediction"]

    for i in range(num_samples):
        a = (img_a_batch[i].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        b = (img_b_batch[i].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        gt = label_batch[i, 0].numpy()
        pred = preds_bin[i, 0].numpy()

        for j, (img, cmap) in enumerate(zip([a, b, gt, pred], [None, None, "gray", "gray"])):
            axes[i][j].imshow(img, cmap=cmap)
            axes[i][j].axis("off")
            if i == 0:
                axes[i][j].set_title(titles[j], fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved predictions to {save_path}")


def train(args: argparse.Namespace, device: torch.device):
    train_ds, val_ds, test_ds = build_datasets(args)
    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, args)

    print(f"Train patches: {len(train_ds)}")
    print(f"Val patches:   {len(val_ds)}")
    print(f"Test patches:  {len(test_ds)}")

    model = ChangeDetectionModel(
        dynamicvis_config=args.dynamicvis_config,
        img_size=args.patch_size,
        fpn_out_channels=args.fpn_out_channels,
        backbone_checkpoint=args.backbone_checkpoint,
    ).to(device)

    set_backbone_trainable(model, trainable=args.train_backbone)
    print(f"Backbone trainable: {args.train_backbone}")

    criterion = BCEDiceLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)

    # DynamicVis decay-based routing expects these values in MMEngine MessageHub.
    message_hub = None
    try:
        from mmengine import MessageHub

        message_hub = MessageHub.get_current_instance()
        message_hub.update_info("max_epochs", float(args.num_epochs))
        message_hub.update_info("epoch", 0.0)
    except Exception as e:
        print(f"Warning: could not initialize MMEngine MessageHub metadata: {e}")

    best_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    print("=" * 60)
    print("Starting change detection training...")
    print("=" * 60)

    for epoch in range(1, args.num_epochs + 1):
        if message_hub is not None:
            # DynamicVis reads epoch/max_epochs during forward for decay schedule.
            message_hub.update_info("epoch", float(epoch - 1))

        train_loss, train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            log_every=args.log_every,
            train_backbone=args.train_backbone,
        )
        val_loss, val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_m["f1"])
        history["val_f1"].append(val_m["f1"])

        print(
            f"Epoch [{epoch:03d}/{args.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} F1: {train_m['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_m['f1']:.4f} IoU: {val_m['iou']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_f1,
                    "val_metrics": val_m,
                    "args": vars(args),
                },
                args.cd_checkpoint_path,
            )
            print(f"  Saved best model: {args.cd_checkpoint_path} (Val F1: {best_f1:.4f})")

    print(f"Training complete. Best Val F1: {best_f1:.4f}")
    return model, history, test_loader


@torch.no_grad()
def evaluate_test(model: nn.Module, test_loader: DataLoader, device: torch.device):
    criterion = BCEDiceLoss()
    test_loss, test_m = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Precision: {test_m['precision']:.4f}")
    print(f"Recall:    {test_m['recall']:.4f}")
    print(f"F1 Score:  {test_m['f1']:.4f}")
    print(f"IoU:       {test_m['iou']:.4f}")
    print("=" * 60)
    return test_m


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DynamicVis change detection on LEVIR-CD")

    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--dynamicvis-config", type=str, default=str(DEFAULT_DYNVIS_CONFIG))
    parser.add_argument("--backbone-checkpoint", type=str, default=str(DEFAULT_BACKBONE_WEIGHTS))

    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)

    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fpn-out-channels", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Train DynamicVis backbone along with FPN/head. Default is frozen backbone.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cd-checkpoint-path", type=str, default="")
    parser.add_argument("--num-visualize", type=int, default=4)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.output_dir = str(Path(args.output_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.cd_checkpoint_path:
        args.cd_checkpoint_path = str(Path(args.output_dir) / "best_cd_model.pth")

    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not Path(args.dynamicvis_config).exists():
        raise FileNotFoundError(f"DynamicVis config not found: {args.dynamicvis_config}")

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.eval_only:
        _, _, test_ds = build_datasets(args)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        model = ChangeDetectionModel(
            dynamicvis_config=args.dynamicvis_config,
            img_size=args.patch_size,
            fpn_out_channels=args.fpn_out_channels,
            backbone_checkpoint=args.backbone_checkpoint,
        ).to(device)

        ckpt = torch.load(args.cd_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(
            f"Loaded CD checkpoint from epoch {ckpt.get('epoch', 'n/a')} "
            f"(Val F1: {ckpt.get('val_f1', -1):.4f})"
        )

        evaluate_test(model, test_loader, device)
        visualize_predictions(
            model,
            test_loader,
            device,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_samples=args.num_visualize,
            save_path=Path(args.output_dir) / "predictions.png",
        )
        return

    model, history, test_loader = train(args, device)

    plot_history(history, Path(args.output_dir) / "training_curves.png")

    print("\nLoading best CD checkpoint for test evaluation...")
    checkpoint = torch.load(args.cd_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(Val F1: {checkpoint['val_f1']:.4f})"
    )

    evaluate_test(model, test_loader, device)
    visualize_predictions(
        model,
        test_loader,
        device,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        num_samples=args.num_visualize,
        save_path=Path(args.output_dir) / "predictions.png",
    )


if __name__ == "__main__":
    main()
