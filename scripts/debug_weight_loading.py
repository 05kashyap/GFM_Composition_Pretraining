#!/usr/bin/env python3
"""
Diagnostic script for debugging pretrained backbone weight loading.

Usage:
    python scripts/debug_weight_loading.py weights/pretrain_dynamicvis_b_bf16_mamba_best_single-label_f1-score_epoch_170.pth
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_checkpoint(path: str) -> dict:
    """Load checkpoint and extract state dict, handling all common formats."""
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {path}")
    print(f"{'='*60}")

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    # Determine checkpoint format
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            print(f"Checkpoint format: wrapped in 'state_dict' key")
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            print(f"Checkpoint format: wrapped in 'model' key")
            state_dict = ckpt['model']
        elif 'backbone' in ckpt:
            print(f"Checkpoint format: wrapped in 'backbone' key")
            state_dict = ckpt['backbone']
        else:
            # Check if keys look like model parameters
            sample_key = next(iter(ckpt.keys()), '')
            if '.' in sample_key or sample_key.endswith('weight') or sample_key.endswith('bias'):
                print(f"Checkpoint format: raw state_dict")
                state_dict = ckpt
            else:
                print(f"Checkpoint format: unknown dict with keys: {list(ckpt.keys())[:5]}...")
                # Try to find state_dict-like nested dict
                for k, v in ckpt.items():
                    if isinstance(v, dict) and len(v) > 10:
                        sample = next(iter(v.keys()), '')
                        if '.' in sample:
                            print(f"  Found nested state_dict in key '{k}'")
                            state_dict = v
                            break
                else:
                    state_dict = ckpt
    else:
        print(f"Checkpoint format: not a dict, type={type(ckpt)}")
        state_dict = ckpt

    print(f"State dict has {len(state_dict)} keys")
    return state_dict


def print_keys(title: str, keys: list, shapes: dict = None, n: int = 20):
    """Print first n keys with optional shapes."""
    print(f"\n{title} (first {min(n, len(keys))} of {len(keys)}):")
    print("-" * 60)
    for k in sorted(keys)[:n]:
        if shapes:
            print(f"  {k}: {list(shapes[k])}")
        else:
            print(f"  {k}")


def get_common_prefixes(keys: list) -> list:
    """Find common prefixes in keys."""
    prefixes = Counter()
    for k in keys:
        parts = k.split('.')
        if len(parts) > 1:
            prefixes[parts[0] + '.'] += 1
    return [p for p, c in prefixes.most_common(5) if c > len(keys) * 0.1]


def try_transformations(ckpt_keys: set, model_keys: set) -> list:
    """Try different key transformations and return results."""
    results = []

    # 1. No transformation
    matches = ckpt_keys & model_keys
    results.append(('No transformation', matches, ckpt_keys, model_keys))

    # 2. Strip 'backbone.' from checkpoint keys
    transformed = {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k
                   for k in ckpt_keys}
    matches = transformed & model_keys
    results.append(("Strip 'backbone.' from checkpoint", matches, transformed, model_keys))

    # 3. Add 'backbone.' prefix to checkpoint keys
    transformed = {'backbone.' + k for k in ckpt_keys}
    matches = transformed & model_keys
    results.append(("Add 'backbone.' to checkpoint", matches, transformed, model_keys))

    # 4. Strip common prefixes from checkpoint
    common_prefixes = get_common_prefixes(list(ckpt_keys))
    for prefix in common_prefixes:
        transformed = {k.replace(prefix, '', 1) if k.startswith(prefix) else k
                       for k in ckpt_keys}
        matches = transformed & model_keys
        results.append((f"Strip '{prefix}' from checkpoint", matches, transformed, model_keys))

    # 5. Try for backbone submodule (checkpoint keys -> model.backbone keys)
    # Model backbone keys without 'backbone.' prefix
    backbone_keys_bare = {k.replace('backbone.', '', 1) for k in model_keys
                          if k.startswith('backbone.')}
    matches = ckpt_keys & backbone_keys_bare
    results.append(("Match checkpoint to model.backbone (stripped)", matches, ckpt_keys, backbone_keys_bare))

    return results


def build_model():
    """Build CompositionAwareDynamicVis with use_dynamicvis_keys=True."""
    print(f"\n{'='*60}")
    print("Building model: CompositionAwareDynamicVis(use_dynamicvis_keys=True)")
    print(f"{'='*60}")

    # Import dynamicvis to register backbone
    import dynamicvis

    # Import after path setup
    from models.composition_head import CompositionAwareDynamicVis

    model = CompositionAwareDynamicVis(
        backbone=dict(
            type='DynamicVisBackbone',
            arch='b',
            path_type='forward_reverse_mean',
            sampling_scale=dict(type='fixed', val=0.1),
            global_token_cfg=dict(pos='head', num=-1),
            is_softmax_on_x=True,
            img_size=512,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            spatial_token_keep_ratios=[8, 4, 2, 1],
            out_indices=(3,),
            out_type='avg_featmap',
        ),
        head=dict(
            type='CompositionHead',
            in_channels=768,
            proj_dim=256,
            hidden_dim=512,
            loss_type='mse',
            standardise_targets=True,
            tau=0.5,
            lambda_comp=0,
            lambda_cosine=0,
            lambda_var=0,
            lambda_cov=0,
            var_gamma=1.0,
            lambda_contrast=0,
            lambda_smooth=0,
            lambda_cls=0.5,
            lambda_slot_contrast=0.5,
            lambda_slot_var=0.25,
            slot_var_gamma=1.0,
            slot_contrast_tau=0.1,
        ),
        num_classes=63,
        num_queries=16,
        slot_dim=256,
        patch_dim=768,
        use_dynamicvis_keys=True,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Debug weight loading")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--verify", action="store_true", help="Verify loading after diagnosis")
    args = parser.parse_args()

    # Load checkpoint
    ckpt_state_dict = load_checkpoint(args.checkpoint)
    ckpt_keys = set(ckpt_state_dict.keys())
    ckpt_shapes = {k: v.shape for k, v in ckpt_state_dict.items()}

    print_keys("Checkpoint keys", list(ckpt_keys), ckpt_shapes, n=20)

    # Build model
    model = build_model()

    # Get backbone state dict
    backbone_sd = model.backbone.state_dict()
    backbone_keys = set(backbone_sd.keys())
    backbone_shapes = {k: v.shape for k, v in backbone_sd.items()}

    print_keys("model.backbone.state_dict() keys", list(backbone_keys), backbone_shapes, n=20)

    # Get full model state dict
    full_sd = model.state_dict()
    full_keys = set(full_sd.keys())
    full_shapes = {k: v.shape for k, v in full_sd.items()}

    print_keys("model.state_dict() keys (full model)", list(full_keys), full_shapes, n=20)

    # Try transformations against backbone keys
    print(f"\n{'='*60}")
    print("Trying transformations (checkpoint → model.backbone)")
    print(f"{'='*60}")

    results = try_transformations(ckpt_keys, backbone_keys)

    best_matches = 0
    best_result = None

    for name, matches, transformed, target in results:
        n_match = len(matches)
        n_target = len(target) if len(target) > 0 else 1  # Avoid div by zero
        print(f"\n{name}:")
        print(f"  Matched: {n_match}/{len(target)} ({100*n_match/n_target:.1f}%)")
        if n_match > best_matches:
            best_matches = n_match
            best_result = (name, matches, transformed, target)

    # Report best result
    if best_result:
        name, matches, transformed, target = best_result
        missing = target - transformed
        unexpected = transformed - target

        print(f"\n{'='*60}")
        print(f"BEST TRANSFORMATION: {name}")
        print(f"{'='*60}")
        print(f"Matched keys:    {len(matches)}")
        print(f"Missing keys:    {len(missing)} (in model but not checkpoint)")
        print(f"Unexpected keys: {len(unexpected)} (in checkpoint but not model)")

        # Show example matched pairs
        print(f"\n5 example matched keys:")
        matched_list = sorted(matches)[:5]
        for k in matched_list:
            # Find original checkpoint key
            if name == "No transformation":
                orig = k
            elif name.startswith("Strip"):
                prefix = name.split("'")[1]
                orig = prefix + k if (prefix + k) in ckpt_keys else k
            elif name.startswith("Add"):
                orig = k.replace('backbone.', '', 1)
            else:
                orig = k
            print(f"  checkpoint: {orig}")
            print(f"  model:      {k}")
            print()

        print(f"\n5 example missing keys (in model backbone, not in checkpoint):")
        for k in sorted(missing)[:5]:
            print(f"  {k}: {list(backbone_shapes.get(k, []))}")

        print(f"\n5 example unexpected keys (in checkpoint, not in model backbone):")
        for k in sorted(unexpected)[:5]:
            print(f"  {k}")

    # Verification
    if args.verify:
        print(f"\n{'='*60}")
        print("VERIFICATION: Testing weight loading")
        print(f"{'='*60}")

        # Apply the best transformation
        if best_result and best_matches > 0:
            name, matches, transformed, target = best_result

            # Build transformation function
            if name == "No transformation":
                def transform(k): return k
            elif "Strip 'backbone.'" in name:
                def transform(k): return k.replace('backbone.', '', 1) if k.startswith('backbone.') else k
            elif "Add 'backbone.'" in name:
                def transform(k): return 'backbone.' + k
            elif "Strip" in name:
                prefix = name.split("'")[1]
                def transform(k): return k.replace(prefix, '', 1) if k.startswith(prefix) else k
            else:
                def transform(k): return k

            # Transform checkpoint keys
            transformed_dict = {transform(k): v for k, v in ckpt_state_dict.items()}

            # Load into backbone
            missing, unexpected = model.backbone.load_state_dict(transformed_dict, strict=False)

            print(f"Loaded with strict=False")
            print(f"Missing:    {len(missing)}")
            print(f"Unexpected: {len(unexpected)}")

            # Verify 3 random parameters match
            import random
            matched_keys = list(set(transformed_dict.keys()) & set(backbone_sd.keys()))
            if len(matched_keys) >= 3:
                verify_keys = random.sample(matched_keys, 3)
                all_match = True
                for k in verify_keys:
                    ckpt_val = transformed_dict[k]
                    model_val = model.backbone.state_dict()[k]
                    if not torch.allclose(ckpt_val, model_val, atol=1e-6):
                        print(f"MISMATCH: {k}")
                        all_match = False
                    else:
                        print(f"MATCH: {k} ✓")

                if all_match:
                    print(f"\nVERIFIED: weights loaded correctly ✓")
                else:
                    print(f"\nFAILED: values do not match")
            else:
                print(f"Not enough matched keys to verify")
        else:
            print("No successful transformation found to verify")


if __name__ == "__main__":
    main()
