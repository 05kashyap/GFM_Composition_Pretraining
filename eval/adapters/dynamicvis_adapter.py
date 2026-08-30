"""DynamicVis model adapter (encoder-only)."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_dynamicvis_stubs() -> None:
    """Install minimal stubs to avoid importing full mmdet/mmseg stacks."""
    if "mmdet" not in sys.modules:
        mmdet_mod = types.ModuleType("mmdet")
        mmdet_models_mod = types.ModuleType("mmdet.models")
        mmdet_structures_mod = types.ModuleType("mmdet.structures")
        mmdet_structures_bbox_mod = types.ModuleType("mmdet.structures.bbox")

        def _nlc_to_nchw(x: torch.Tensor, hw_shape):
            h, w = hw_shape
            b, n, c = x.shape
            if n != h * w:
                raise ValueError(f"nlc_to_nchw: N={n} must equal H*W={h*w}")
            return x.transpose(1, 2).contiguous().view(b, c, h, w)

        def _nchw_to_nlc(x: torch.Tensor):
            b, c, h, w = x.shape
            return x.view(b, c, h * w).transpose(1, 2).contiguous()

        mmdet_models_mod.nlc_to_nchw = _nlc_to_nchw
        mmdet_models_mod.nchw_to_nlc = _nchw_to_nlc

        class _FPNStub(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                raise RuntimeError("FPN requires full mmdet/mmcv; not supported in lightweight setup.")

        mmdet_models_mod.FPN = _FPNStub

        def _bbox2roi_stub(*args, **kwargs):
            raise RuntimeError("bbox2roi requires full mmdet; not supported in lightweight setup.")

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


def _extract_backbone_cfg(cfg: Any) -> Dict[str, Any]:
    model_cfg = cfg.get("model", None) if hasattr(cfg, "get") else getattr(cfg, "model", None)
    if isinstance(model_cfg, dict):
        return dict(model_cfg.get("backbone", {}))
    return {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_dynamicvis_root() -> Path:
    dynamicvis_root_env = os.environ.get("DYNAMICVIS_ROOT", "")
    if dynamicvis_root_env:
        return Path(dynamicvis_root_env).expanduser()
    return _repo_root() / "architectures" / "DynamicVis"


def _report_mamba_fast_path_status() -> None:
    missing = []
    if importlib.util.find_spec("mamba_ssm") is None:
        missing.append("mamba-ssm")
    if importlib.util.find_spec("causal_conv1d") is None:
        missing.append("causal-conv1d")

    if not missing:
        return

    repo_root = _repo_root()
    local_wheel = repo_root / (
        "causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
    )
    wheel_hint = f" Local wheel: {local_wheel}." if local_wheel.exists() else ""

    warnings.warn(
        "DynamicVis fast path is unavailable because dependencies are missing: "
        f"{', '.join(missing)}. Install the matching CUDA/Torch wheels from "
        "architectures/DynamicVis/README.md to enable the fast path." + wheel_hint,
        stacklevel=2,
    )


class DynamicVisEncoder(nn.Module):
    """Encoder wrapper for DynamicVis backbone (retrieval embeddings only)."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        embedding_dim: int = 768,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim

        _report_mamba_fast_path_status()

        dynamicvis_root = _resolve_dynamicvis_root()

        if not dynamicvis_root.exists():
            raise FileNotFoundError(
                f"DynamicVis repo not found at: {dynamicvis_root}. "
                "Set DYNAMICVIS_ROOT or clone into architectures/DynamicVis."
            )

        if str(dynamicvis_root) not in sys.path:
            sys.path.insert(0, str(dynamicvis_root))

        # DynamicVis repo includes its own mmdet/mmseg/mmpretrain packages,
        # so we don't need stubs. Only install stubs if those packages are missing.
        has_bundled_mmdet = (dynamicvis_root / "mmdet" / "__init__.py").exists()
        if not has_bundled_mmdet:
            _install_dynamicvis_stubs()

        from mmengine import Config
        from dynamicvis.models.models import DynamicVisBackbone  # type: ignore

        cfg = Config.fromfile(config_path)
        backbone_cfg = _extract_backbone_cfg(cfg)
        backbone_cfg.pop("type", None)

        backbone_cfg.setdefault("arch", "b")
        backbone_cfg.setdefault("img_size", int(cfg.get("img_size", 224)))
        backbone_cfg.setdefault("in_channels", 3)
        backbone_cfg["out_indices"] = (3,)
        backbone_cfg["out_type"] = "avg_featmap"

        self.backbone = DynamicVisBackbone(**backbone_cfg)
        feature_dim = self.backbone.c_embed_dims[-1]

        if self.embedding_dim != feature_dim:
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )
        else:
            self.projection_head = nn.Identity()

        self._load_checkpoint(model_path)
        self.to(self.device)

    def _load_checkpoint(self, model_path: str) -> None:
        ckpt = torch.load(model_path, map_location="cpu")
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
            raise ValueError("Unexpected checkpoint format for DynamicVis model.")

        expected_keys = set(self.backbone.state_dict().keys())

        # Normalize common wrappers from distributed and trainer checkpoints.
        normalized_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
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
                candidate_key = k[len("backbone."):]
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
            "DynamicVis checkpoint load summary: "
            f"total_keys={len(state_dict)}, normalized_keys={len(normalized_state)}, "
            f"retained_backbone_keys={len(filtered)}, loaded={loaded}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"dropped_non_backbone={dropped_non_backbone}"
        )
        if missing:
            print(f"Warning: missing keys when loading DynamicVis checkpoint: {missing[:10]}")
        if unexpected:
            print(f"Warning: unexpected keys when loading DynamicVis checkpoint: {unexpected[:10]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            x = feats[-1]
        else:
            x = feats
        x = self.projection_head(x)
        return F.normalize(x, p=2, dim=1)
