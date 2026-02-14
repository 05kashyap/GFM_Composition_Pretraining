#!/usr/bin/env python3
"""Scan cached DINOv3 embeddings for NaN values and report statistics."""

import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import os

CACHE_DIR = Path("outputs/preprocess_cache_dinov3")

def main():
    files = sorted(CACHE_DIR.glob("*.npz"))
    total_files = len(files)
    print(f"Total cached .npz files: {total_files}")

    nan_files = []           # files with any NaN
    corrupt_files = []       # files that can't be loaded
    total_embeddings = 0     # total number of embedding vectors
    total_nan_embeddings = 0 # embedding vectors with at least one NaN
    total_elements = 0       # total scalar elements
    total_nan_elements = 0   # NaN scalar elements
    all_nan_files = []       # files where ALL embeddings are NaN
    embed_dims = set()

    for i, f in enumerate(files):
        if i % 10000 == 0 and i > 0:
            print(f"  Scanned {i}/{total_files} files... "
                  f"({len(nan_files)} with NaN so far)", flush=True)
        try:
            data = np.load(f, allow_pickle=False)
            emb = data["emb"]
            embed_dims.add(emb.shape)
        except Exception as e:
            corrupt_files.append((f.name, str(e)))
            continue

        n_vectors = emb.shape[0] if emb.ndim > 1 else 1
        n_elements = emb.size
        nan_mask = np.isnan(emb)
        n_nan_elements = int(nan_mask.sum())
        
        total_embeddings += n_vectors
        total_elements += n_elements

        if n_nan_elements > 0:
            total_nan_elements += n_nan_elements
            # Count vectors (rows) with any NaN
            if emb.ndim > 1:
                nan_rows = int(nan_mask.any(axis=1).sum())
                all_nan_rows = int(nan_mask.all(axis=1).sum())
            else:
                nan_rows = 1
                all_nan_rows = 1 if nan_mask.all() else 0
            
            total_nan_embeddings += nan_rows
            is_all_nan = (n_nan_elements == n_elements)
            
            nan_files.append({
                "file": f.name,
                "shape": emb.shape,
                "dtype": str(emb.dtype),
                "nan_elements": n_nan_elements,
                "total_elements": n_elements,
                "nan_rows": nan_rows,
                "all_nan_rows": all_nan_rows,
                "total_rows": n_vectors,
                "nan_frac": n_nan_elements / n_elements,
                "all_nan": is_all_nan,
            })
            if is_all_nan:
                all_nan_files.append(f.name)

            # Try to extract image path
            try:
                img_path = str(data.get("image_path", "unknown"))
            except:
                img_path = "unknown"

    # ---- Report ----
    print("\n" + "=" * 70)
    print("NaN INVESTIGATION REPORT")
    print("=" * 70)

    print(f"\nTotal cache files:       {total_files:,}")
    print(f"Corrupt/unloadable:      {len(corrupt_files):,}")
    print(f"Files with NaN:          {len(nan_files):,}  "
          f"({100*len(nan_files)/total_files:.2f}%)")
    print(f"Files fully NaN:         {len(all_nan_files):,}")
    print(f"Embedding shapes seen:   {embed_dims}")

    print(f"\nTotal embedding vectors: {total_embeddings:,}")
    print(f"Vectors with any NaN:    {total_nan_embeddings:,}  "
          f"({100*total_nan_embeddings/max(total_embeddings,1):.4f}%)")

    print(f"\nTotal scalar elements:   {total_elements:,}")
    print(f"NaN scalar elements:     {total_nan_elements:,}  "
          f"({100*total_nan_elements/max(total_elements,1):.6f}%)")

    if corrupt_files:
        print(f"\n--- Corrupt files (first 10) ---")
        for name, err in corrupt_files[:10]:
            print(f"  {name}: {err}")

    if nan_files:
        print(f"\n--- Files with NaN (first 20) ---")
        print(f"{'File':<50} {'Shape':>12} {'NaN rows':>10} {'Total rows':>10} {'NaN frac':>10} {'AllNaN':>6}")
        for info in nan_files[:20]:
            print(f"{info['file']:<50} {str(info['shape']):>12} "
                  f"{info['nan_rows']:>10} {info['total_rows']:>10} "
                  f"{info['nan_frac']:>10.4f} {'YES' if info['all_nan'] else 'no':>6}")
        
        if len(nan_files) > 20:
            print(f"  ... and {len(nan_files)-20} more files with NaN")

        # Distribution of NaN fraction
        fracs = [info["nan_frac"] for info in nan_files]
        print(f"\n--- NaN fraction distribution across affected files ---")
        print(f"  Min:    {min(fracs):.6f}")
        print(f"  Median: {sorted(fracs)[len(fracs)//2]:.6f}")
        print(f"  Max:    {max(fracs):.6f}")
        print(f"  Mean:   {sum(fracs)/len(fracs):.6f}")

        # Check: are NaN values Inf-related? Sample a few
        print(f"\n--- Sample NaN file value ranges ---")
        for info in nan_files[:5]:
            fpath = CACHE_DIR / info["file"]
            emb = np.load(fpath, allow_pickle=False)["emb"]
            finite = emb[np.isfinite(emb)]
            inf_count = int(np.isinf(emb).sum())
            if finite.size > 0:
                print(f"  {info['file']}: finite range [{finite.min():.4f}, {finite.max():.4f}], "
                      f"inf={inf_count}, nan={info['nan_elements']}/{info['total_elements']}")
            else:
                print(f"  {info['file']}: ALL values are NaN/Inf, inf={inf_count}")

    print("\n" + "=" * 70)
    if len(nan_files) == 0:
        print("No NaN found in cached embeddings!")
        print("The NaN may be introduced during fp16->fp32 load or PCA assembly.")
    else:
        clean_vectors = total_embeddings - total_nan_embeddings
        print(f"SUMMARY: {total_nan_embeddings:,} NaN vectors out of "
              f"{total_embeddings:,} total ({100*total_nan_embeddings/max(total_embeddings,1):.4f}%)")
        print(f"Clean vectors available: {clean_vectors:,}")
        print(f"\nSimple fix: filter out NaN rows before PCA.")
    print("=" * 70)


if __name__ == "__main__":
    main()
