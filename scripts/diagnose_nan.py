#!/usr/bin/env python3
"""Diagnose WHY DINOv3 embeddings are all NaN.

Loads the model on CPU, runs a small synthetic input, and traces NaN
propagation through each layer to pinpoint exactly where it occurs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ".")

import torch
import numpy as np
from pathlib import Path

# Import directly — the module is on sys.path now
import scripts.viz_fmow_patch_embed_cluster_dinov3 as mod

WEIGHTS = "weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

def check_tensor(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()
    total = t.numel()
    if nan_count > 0 or inf_count > 0:
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
              f"NaN={nan_count}/{total} Inf={inf_count}/{total} "
              f"PROBLEM!")
    else:
        print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
              f"range=[{t.min().item():.4f}, {t.max().item():.4f}] OK")


def main():
    device = "cpu"
    print(f"=== DINOv3 NaN Diagnostic (device={device}) ===\n")

    # 1. Load model
    print("1. Building model...")
    model = mod.DinoV3SatViTL16(
        patch_size=16, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4.0, num_register_tokens=4,
    )
    model.eval()

    print("2. Loading weights...")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd_raw = mod._extract_checkpoint_state_dict(ckpt)
    sd = {mod._strip_known_prefixes(k): v for k, v in sd_raw.items()
          if isinstance(v, torch.Tensor)}

    # Check for NaN in the weights themselves
    nan_weight_keys = []
    for k, v in sd.items():
        if torch.isnan(v).any():
            nan_weight_keys.append(k)
    if nan_weight_keys:
        print(f"  WARNING: {len(nan_weight_keys)} checkpoint tensors contain NaN!")
        for k in nan_weight_keys[:10]:
            check_tensor(f"    ckpt[{k}]", sd[k])
    else:
        print(f"  Checkpoint: {len(sd)} tensors, no NaN in weights.")

    incompatible = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in incompatible.missing_keys if "bias_mask" not in k]
    real_unexpected = [k for k in incompatible.unexpected_keys if "bias_mask" not in k]
    print(f"  Missing (non-bias_mask): {len(real_missing)}")
    print(f"  Unexpected (non-bias_mask): {len(real_unexpected)}")
    if real_missing:
        print(f"    Missing: {real_missing[:10]}")
    if real_unexpected:
        print(f"    Unexpected: {real_unexpected[:10]}")

    # Check model params after loading
    nan_params = [(n, p) for n, p in model.named_parameters() if torch.isnan(p).any()]
    nan_buffers = [(n, b) for n, b in model.named_buffers() if torch.isnan(b).any()]
    print(f"  Model params with NaN: {len(nan_params)}")
    print(f"  Model buffers with NaN: {len(nan_buffers)}")
    if nan_params:
        for n, p in nan_params[:5]:
            check_tensor(f"    param[{n}]", p.data)

    # 2. Create synthetic input (128x128 patch, 1 image)
    print("\n3. Running forward pass with synthetic input...")
    # Simulate what _embed_batch does
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Random image in [0,1]
    x = torch.rand(1, 3, 128, 128)
    x_norm = (x - mean) / std
    check_tensor("Input (normalized)", x_norm)

    with torch.inference_mode():
        # Step through the model manually
        tokens = model.patch_embed(x_norm)
        check_tensor("After patch_embed", tokens)

        h = 128 // 16  # = 8
        w = 128 // 16  # = 8
        B = 1
        num_prefix = 1 + model.num_register_tokens  # 5

        cls = model.cls_token.expand(B, -1, -1)
        reg = model.storage_tokens.expand(B, -1, -1)
        check_tensor("cls_token", cls)
        check_tensor("storage_tokens", reg)

        tokens = torch.cat([cls, reg, tokens], dim=1)
        check_tensor("Tokens (cls+reg+patches)", tokens)

        rope_cos, rope_sin = model.rope_embed.get_cos_sin(
            h, w, device=tokens.device, dtype=tokens.dtype,
        )
        check_tensor("RoPE cos", rope_cos)
        check_tensor("RoPE sin", rope_sin)

        # Run through each block and check for NaN propagation
        print("\n4. Per-block forward (checking NaN propagation):")
        for i, blk in enumerate(model.blocks):
            tokens = blk(tokens, rope_cos=rope_cos, rope_sin=rope_sin,
                        num_prefix=num_prefix)
            has_nan = torch.isnan(tokens).any().item()
            has_inf = torch.isinf(tokens).any().item()
            if has_nan or has_inf:
                nan_frac = torch.isnan(tokens).sum().item() / tokens.numel()
                print(f"  Block {i:2d}: NaN={has_nan} Inf={has_inf} "
                      f"NaN%={100*nan_frac:.1f}% "
                      f"range=[{tokens[~torch.isnan(tokens)].min().item():.4f}, "
                      f"{tokens[~torch.isnan(tokens)].max().item():.4f}] "
                      f"<-- FIRST NaN BLOCK!" if i == 0 or not has_nan else "")
                if has_nan and i < 2:  # Detailed debug for first NaN block
                    # Check attention internals
                    x_in = tokens.clone()
                    x_in[torch.isnan(x_in)] = 0  # Reset for retry
                    normed = blk.norm1(x_in)
                    check_tensor(f"    Block {i} norm1 output", normed)
                    attn_out = blk.attn(normed, rope_cos=rope_cos, rope_sin=rope_sin,
                                       num_prefix=num_prefix)
                    check_tensor(f"    Block {i} attn output", attn_out)
            else:
                mn = tokens.min().item()
                mx = tokens.max().item()
                print(f"  Block {i:2d}: OK  range=[{mn:.4f}, {mx:.4f}]")

        tokens_final = model.norm(tokens)
        check_tensor("After final norm", tokens_final)

        cls_out = tokens_final[:, 0]
        patch_out = tokens_final[:, num_prefix:]
        check_tensor("cls_out", cls_out)
        check_tensor("patch_out", patch_out)

        # Pool (cls_avg)
        avg = patch_out.mean(dim=1)
        emb = torch.cat([cls_out, avg], dim=-1)
        check_tensor("Final embedding (cls_avg)", emb)

    # 5. Check float16 conversion
    print("\n5. Float16 conversion check:")
    if not torch.isnan(emb).any():
        emb_np = emb.numpy().astype(np.float32)
        check_tensor("As float32 numpy->torch", torch.from_numpy(emb_np))
        emb_f16 = emb_np.astype(np.float16)
        nan_after_f16 = np.isnan(emb_f16).sum()
        inf_after_f16 = np.isinf(emb_f16).sum()
        print(f"  After float16 cast: NaN={nan_after_f16} Inf={inf_after_f16} "
              f"total={emb_f16.size}")
        if nan_after_f16 > 0 or inf_after_f16 > 0:
            print(f"  Float32 range: [{emb_np.min():.4f}, {emb_np.max():.4f}]")
            print(f"  Values > 65504 (fp16 max): {(np.abs(emb_np) > 65504).sum()}")
    else:
        print("  Skipped (embedding already NaN)")

    # 6. Also test with autocast (fp16) if on GPU
    if device == "cpu":
        print("\n6. Testing with manual fp16 forward (simulating autocast):")
        with torch.inference_mode():
            x_f16 = x_norm.half()
            model_f16 = model.half()
            try:
                cls_out_f16, patch_out_f16 = model_f16(x_f16)
                check_tensor("fp16 cls_out", cls_out_f16)
                check_tensor("fp16 patch_out", patch_out_f16)
            except Exception as e:
                print(f"  fp16 forward failed: {e}")

    print("\n=== Diagnostic complete ===")


if __name__ == "__main__":
    main()
