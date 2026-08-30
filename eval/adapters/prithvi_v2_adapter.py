import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from pathlib import Path

from architectures.Prithvi_v2.prithvi_mae_v2 import PrithviMAE
#TODO: Make img_size, in_chans configurable

class PrithviEncoderV2(nn.Module):
    """Encoder wrapper for Prithvi v2 foundation models."""

    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 512,
        device: Optional[torch.device] = None,
        use_multi_scale: bool = False,
        layer_indices: Optional[List[int]] = None,
        model_config: Optional[Dict] = None,
        img_size: int = 224,
        in_chans: int = 4,
    ):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.use_multi_scale = use_multi_scale

        default_config: Dict = {
            'img_size': img_size,
            'patch_size': (1, 16, 16),
            'in_chans': in_chans,
            'num_frames': 1,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'decoder_embed_dim': 512,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
            'mlp_ratio': 4.0,
            'coords_encoding': None,
            'coords_scale_learn': False,
            'drop_path': 0.0,
            'mask_ratio': 0.0,
        }
        if model_config:
            default_config.update(model_config)

        state_dict = self._load_raw_state_dict(model_path)
        self.config = self._build_model_config(default_config, state_dict)

        self.backbone = PrithviMAE(**self.config)

        if layer_indices is None:
            self.layer_indices = [-4, -3, -2, -1]
        else:
            self.layer_indices = layer_indices
        print(f"✓ Using layer indices for multi-scale: {self.layer_indices}")
        self._load_checkpoint(state_dict)

        self.feature_dim = self.backbone.encoder.embed_dim

        if self.use_multi_scale:
            num_layers = len(self.layer_indices)
            if num_layers == 0:
                raise ValueError("layer_indices must contain at least one index when use_multi_scale=True.")
            per_layer_dim = max(1, self.embedding_dim // num_layers)

            self.multi_scale_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feature_dim, per_layer_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(per_layer_dim, per_layer_dim)
                ) for _ in range(num_layers)
            ])

            self.fusion = nn.Sequential(
                nn.Linear(per_layer_dim * num_layers, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
        else:
            if self.embedding_dim != self.feature_dim:
                self.projection_head = nn.Sequential(
                    nn.Linear(self.feature_dim, self.embedding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.embedding_dim, self.embedding_dim)
                )
            else:
                self.projection_head = nn.Identity()

        self.to(self.device)

    def _load_raw_state_dict(self, model_path: str) -> Dict[str, torch.Tensor]:
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Prithvi v2 checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint.get('model', checkpoint)
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("Unexpected checkpoint format for Prithvi v2 model.")

        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        return state_dict

    # def _build_model_config(self, base_config: Dict, state_dict: Dict[str, torch.Tensor]) -> Dict:
    #     config = dict(base_config)

    #     patch_key = next((k for k in state_dict if k.endswith('patch_embed.proj.weight')), None)
    #     if patch_key:
    #         weight = state_dict[patch_key]
    #         config['embed_dim'] = int(weight.shape[0])
    #         config['in_chans'] = min(int(weight.shape[1]), config.get('in_chans', int(weight.shape[1])))
    #         patch_size = tuple(int(v) for v in weight.shape[2:5])
    #         if len(patch_size) == 3:
    #             config['patch_size'] = patch_size

    #     block_indices = {int(k.split('.')[2])
    #                      for k in state_dict
    #                      if k.startswith('encoder.blocks.') and k.split('.')[2].isdigit()}
    #     if block_indices:
    #         config['depth'] = max(block_indices) + 1

    #     pos_key = next((k for k in state_dict if k.endswith('pos_embed')), None)
    #     if pos_key:
    #         pos = state_dict[pos_key]
    #         if pos.dim() == 3 and pos.shape[1] > 1:
    #             spatial_tokens = pos.shape[1] - 1
    #             patch_t, patch_h, patch_w = config['patch_size']
    #             candidate_frames = 1
    #             grid_side = None
    #             for frames in range(1, 33):
    #                 if spatial_tokens % frames != 0:
    #                     continue
    #                 tokens_per_frame = spatial_tokens // frames
    #                 side = int(round(tokens_per_frame ** 0.5))
    #                 if side * side == tokens_per_frame:
    #                     candidate_frames = frames
    #                     grid_side = side
    #                     break
    #             if grid_side:
    #                 config['num_frames'] = candidate_frames
    #                 config['img_size'] = grid_side * patch_h

    #     return config


    def _build_model_config(self, base_config: Dict, state_dict: Dict[str, torch.Tensor]) -> Dict:
        config = dict(base_config)

        patch_key = next((k for k in state_dict if k.endswith('patch_embed.proj.weight')), None)
        if patch_key:
            weight = state_dict[patch_key]
            config['embed_dim'] = int(weight.shape[0])
            config['in_chans'] = int(base_config.get('in_chans', weight.shape[1]))
            patch_size = tuple(int(v) for v in weight.shape[2:5])
            if len(patch_size) == 3:
                config['patch_size'] = patch_size

        block_indices = {int(k.split('.')[2])
                        for k in state_dict
                        if k.startswith('encoder.blocks.') and k.split('.')[2].isdigit()}
        if block_indices:
            config['depth'] = max(block_indices) + 1

        pos_key = next((k for k in state_dict if k.endswith('pos_embed')), None)
        if pos_key:
            pos = state_dict[pos_key]
            if pos.dim() == 3 and pos.shape[1] > 1:
                spatial_tokens = pos.shape[1] - 1
                patch_t, patch_h, patch_w = config['patch_size']
                candidate_frames = 1
                grid_side = None
                for frames in range(1, 33):
                    if spatial_tokens % frames != 0:
                        continue
                    tokens_per_frame = spatial_tokens // frames
                    side = int(round(tokens_per_frame ** 0.5))
                    if side * side == tokens_per_frame:
                        candidate_frames = frames
                        grid_side = side
                        break
                if grid_side:
                    config['num_frames'] = candidate_frames
                    config['img_size'] = grid_side * patch_h

        return config

    # def _load_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
    #     patch_weight_keys = [k for k in state_dict if k.endswith('patch_embed.proj.weight')]
    #     for key in patch_weight_keys:
    #         weight = state_dict[key]
    #         if weight.shape[1] > self.config['in_chans']:
    #             trimmed = weight[:, :self.config['in_chans'], ...].clone()
    #             trimmed *= weight.shape[1] / self.config['in_chans']
    #             state_dict[key] = trimmed

    #     pos_embed_keys = [k for k in list(state_dict.keys()) if k.endswith('pos_embed')]
    #     expected_tokens = self.backbone.encoder.patch_embed.num_patches + 1
    #     for key in pos_embed_keys:
    #         pos = state_dict[key]
    #         if pos.dim() != 3:
    #             continue
    #         if pos.shape[1] != expected_tokens:
    #             print(
    #                 f"Warning: dropping positional embedding '{key}' "
    #                 f"(checkpoint tokens={pos.shape[1]}, expected={expected_tokens}). "
    #                 "Using deterministic sin-cos initialization instead."
    #             )
    #             del state_dict[key]

    #     decoder_keys = [k for k in list(state_dict.keys()) if k.startswith('decoder.')]
    #     if decoder_keys:
    #         print(f"Info: dropping {len(decoder_keys)} decoder weights (encoder-only inference).")
    #         for key in decoder_keys:
    #             del state_dict[key]
                
    #     missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
    #     if missing:
    #         print(f"Warning: missing keys when loading Prithvi v2 checkpoint: {missing[:10]}")
    #     if unexpected:
    #         print(f"Warning: unexpected keys when loading Prithvi v2 checkpoint: {unexpected[:10]}")


    def _load_checkpoint(self, state_dict: Dict[str, torch.Tensor]) -> None:
        patch_weight_keys = [k for k in state_dict if k.endswith('patch_embed.proj.weight')]
        for key in patch_weight_keys:
            weight = state_dict[key]
            if weight.shape[1] > self.config['in_chans']:
                # Select first channels to match requested input
                trimmed = weight[:, :self.config['in_chans'], ...].clone()
                # Optional: Scale weights to preserve magnitude (can experiment with this)
                trimmed *= weight.shape[1] / self.config['in_chans']
                state_dict[key] = trimmed
            elif weight.shape[1] < self.config['in_chans']:
                # If checkpoint has fewer channels (unlikely), pad with zeros
                padding = torch.zeros(
                    weight.shape[0], 
                    self.config['in_chans'] - weight.shape[1], 
                    *weight.shape[2:], 
                    device=weight.device
                )
                state_dict[key] = torch.cat([weight, padding], dim=1)

        pos_embed_keys = [k for k in list(state_dict.keys()) if k.endswith('pos_embed')]
        expected_tokens = self.backbone.encoder.patch_embed.num_patches + 1
        for key in pos_embed_keys:
            pos = state_dict[key]
            if pos.dim() != 3:
                continue
            if pos.shape[1] != expected_tokens:
                print(
                    f"Warning: dropping positional embedding '{key}' "
                    f"(checkpoint tokens={pos.shape[1]}, expected={expected_tokens}). "
                    "Using deterministic sin-cos initialization instead."
                )
                del state_dict[key]

        decoder_keys = [k for k in list(state_dict.keys()) if k.startswith('decoder.')]
        if decoder_keys:
            print(f"Info: dropping {len(decoder_keys)} decoder weights (encoder-only inference).")
            for key in decoder_keys:
                del state_dict[key]
                
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading Prithvi v2 checkpoint: {missing[:10]}")
        if unexpected:
            print(f"Warning: unexpected keys when loading Prithvi v2 checkpoint: {unexpected[:10]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)

        if self.use_multi_scale:
            features = self.backbone.forward_features(x)
            multi_scale_embeddings = []
            total_layers = len(features)
            for proj, layer_idx in zip(self.multi_scale_projections, self.layer_indices):
                idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
                if idx < 0 or idx >= total_layers:
                    raise IndexError(f"Layer index {layer_idx} is out of bounds for feature map list of length {total_layers}.")
                layer_features = features[idx]
                patch_tokens = layer_features[:, 1:, :]
                pooled = patch_tokens.mean(dim=1)
                multi_scale_embeddings.append(proj(pooled))

            concatenated = torch.cat(multi_scale_embeddings, dim=1)
            embeddings = self.fusion(concatenated)
        else:
            features = self.backbone.forward_features(x)[-1]
            patch_tokens = features[:, 1:, :]
            pooled_tokens = patch_tokens.mean(dim=1)
            embeddings = self.projection_head(pooled_tokens)

        return F.normalize(embeddings, p=2, dim=1)

    def _tokens_to_feature_map(self, layer_features: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
        patch_tokens = layer_features[:, 1:, :]
        batch_size, num_tokens, channels = patch_tokens.shape

        if len(input_shape) != 5:
            raise ValueError(f"Expected 5D input shape for Prithvi features, got {tuple(input_shape)}")

        _, _, time, height, width = input_shape
        patch_t, patch_h, patch_w = self.config['patch_size']
        num_frames = int(self.config.get('num_frames', 1))

        grid_h = max(1, height // patch_h)
        grid_w = max(1, width // patch_w)
        expected_tokens = grid_h * grid_w * max(1, time // max(1, patch_t))

        if expected_tokens != num_tokens:
            if num_tokens % max(1, num_frames) == 0:
                tokens_per_frame = num_tokens // max(1, num_frames)
                side = int(round(tokens_per_frame ** 0.5))
                if side * side * max(1, num_frames) == num_tokens:
                    grid_h = side
                    grid_w = side
                else:
                    raise ValueError(
                        f"Unable to reshape Prithvi tokens into a feature map: tokens={num_tokens}, "
                        f"expected={expected_tokens}, input_shape={tuple(input_shape)}, "
                        f"patch_size={self.config['patch_size']}"
                    )
            else:
                raise ValueError(
                    f"Unable to reshape Prithvi tokens into a feature map: tokens={num_tokens}, "
                    f"expected={expected_tokens}, input_shape={tuple(input_shape)}, "
                    f"patch_size={self.config['patch_size']}"
                )

        return patch_tokens.transpose(1, 2).contiguous().view(batch_size, channels, grid_h, grid_w)

    def forward_feature_maps(
        self,
        x: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        pyramid_scales: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        if x.dim() == 4:
            x = x.unsqueeze(2)

        features = self.backbone.forward_features(x)
        selected_indices = layer_indices if layer_indices is not None else self.layer_indices
        if not selected_indices:
            raise ValueError("layer_indices must contain at least one index.")

        maps: List[torch.Tensor] = []
        total_layers = len(features)
        for layer_idx in selected_indices:
            idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
            if idx < 0 or idx >= total_layers:
                raise IndexError(
                    f"Layer index {layer_idx} is out of bounds for feature map list of length {total_layers}."
                )
            maps.append(self._tokens_to_feature_map(features[idx], x.shape))

        if pyramid_scales is None:
            pyramid_scales = [2 ** i for i in range(len(maps))]

        pyramid_maps: List[torch.Tensor] = []
        base_h, base_w = maps[0].shape[-2:]
        for feat_map, scale in zip(maps, pyramid_scales):
            target_h = max(1, base_h // max(1, scale))
            target_w = max(1, base_w // max(1, scale))
            if feat_map.shape[-2:] != (target_h, target_w):
                feat_map = F.adaptive_avg_pool2d(feat_map, (target_h, target_w))
            pyramid_maps.append(feat_map)

        return pyramid_maps

    def get_embedding_dim(self) -> int:
        return self.embedding_dim