"""
K-Sweep Utility Functions

Handles CBIR log parsing, metric extraction, silhouette scoring integration,
and summary file management for the BoVW K-sweep orchestration.
"""

import re
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def parse_cbir_log(log_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse CBIR evaluation log file and extract AID metrics.

    Extracts Recall@1/5/10 and mAP@1/5/10 from the eval/cbir/main.py output.
    Handles both stratified k-fold (with mean ± std) and fixed split formats.

    Args:
        log_path: Path to the CBIR evaluation log file.

    Returns:
        Dictionary with metrics, or None if parsing fails.

    Example output:
        {
            "recall_at_1": 0.123,
            "recall_at_5": 0.456,
            "recall_at_10": 0.789,
            "map_at_1": 0.111,
            "map_at_5": 0.222,
            "map_at_10": 0.333,
            "recall_at_1_std": 0.01,
            "map_at_1_std": 0.02
        }
    """
    try:
        with open(log_path, "r") as f:
            content = f.read()

        metrics = {}

        # Pattern 1: "Recall@5: 0.1234 ± 0.0123 | mAP@5: 0.5678 ± 0.0456" (k-fold with std)
        kfold_pattern = (
            r"Recall@(\d+):\s+([0-9.]+)\s+±\s+([0-9.]+).*?"
            r"mAP@\1:\s+([0-9.]+)\s+±\s+([0-9.]+)"
        )
        for match in re.finditer(kfold_pattern, content):
            k_val = match.group(1)
            recall_mean = float(match.group(2))
            recall_std = float(match.group(3))
            map_mean = float(match.group(4))
            map_std = float(match.group(5))

            metrics[f"recall_at_{k_val}"] = recall_mean
            metrics[f"recall_at_{k_val}_std"] = recall_std
            metrics[f"map_at_{k_val}"] = map_mean
            metrics[f"map_at_{k_val}_std"] = map_std

        # Pattern 2: "Recall@5: 0.1234 | mAP@5: 0.5678" (fixed split, no std)
        if not metrics:
            fixed_pattern = r"Recall@(\d+):\s+([0-9.]+).*?mAP@\1:\s+([0-9.]+)"
            for match in re.finditer(fixed_pattern, content):
                k_val = match.group(1)
                recall_score = float(match.group(2))
                map_score = float(match.group(3))

                metrics[f"recall_at_{k_val}"] = recall_score
                metrics[f"map_at_{k_val}"] = map_score

        if not metrics:
            print(f"Warning: No CBIR metrics found in log {log_path}")
            return None

        return metrics

    except Exception as e:
        print(f"Error parsing CBIR log {log_path}: {e}")
        return None


def compute_silhouette_from_histograms(
    centroids: np.ndarray,
    histograms: np.ndarray,
    seed: int = 42,
) -> Optional[float]:
    """
    Compute silhouette score from centroids and histogram assignments.

    Uses argmax of histogram soft-assignments as cluster labels,
    then computes silhouette score against the centroid embeddings.

    Args:
        centroids: (K, D) cluster centroids
        histograms: (N, K) soft-assignment histograms
        seed: Random seed for sampling

    Returns:
        Silhouette score, or None if invalid.
    """
    try:
        from scripts.pipeline_utils import silhouette_optional
    except ImportError:
        print("Warning: Could not import silhouette_optional from pipeline_utils")
        return None

    if centroids.shape[0] < 2 or histograms.shape[0] < 3:
        print(f"Warning: Insufficient data for silhouette (K={centroids.shape[0]}, N={histograms.shape[0]})")
        return None

    # Use hard assignments (argmax)
    labels = np.argmax(histograms, axis=1)

    # Compute silhouette on centroid embeddings with assignment labels
    score = silhouette_optional(centroids, labels, seed=seed)
    return score


def append_to_summary_file(
    summary_file: str,
    k: int,
    silhouette_score: Optional[float],
    cbir_metrics: Optional[Dict[str, float]],
) -> bool:
    """
    Append or update a result row in the summary JSON file.

    Args:
        summary_file: Path to JSON summary file.
        k: Vocabulary size.
        silhouette_score: Silhouette score, or None.
        cbir_metrics: Dictionary of CBIR metrics, or None.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Load existing summary
        with open(summary_file, "r") as f:
            summary = json.load(f)

        result = {
            "k": int(k),
            "silhouette_score": silhouette_score,
            "cbir_metrics": cbir_metrics or {},
        }

        # Add K to metadata if not present
        if int(k) not in summary.get("sweep_metadata", {}).get("k_values", []):
            if "sweep_metadata" not in summary:
                summary["sweep_metadata"] = {}
            if "k_values" not in summary["sweep_metadata"]:
                summary["sweep_metadata"]["k_values"] = []
            summary["sweep_metadata"]["k_values"].append(int(k))

        # Replace or append result
        existing_idx = next(
            (i for i, r in enumerate(summary.get("results", [])) if r.get("k") == int(k)),
            None,
        )
        if existing_idx is not None:
            summary["results"][existing_idx] = result
        else:
            if "results" not in summary:
                summary["results"] = []
            summary["results"].append(result)

        # Write back
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return True

    except Exception as e:
        print(f"Error appending to summary file {summary_file}: {e}")
        return False


def generate_summary_csv(json_file: str, csv_file: str) -> bool:
    """
    Convert summary JSON to CSV for easier viewing.

    Args:
        json_file: Path to JSON summary file.
        csv_file: Path to output CSV file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import csv

        with open(json_file, "r") as f:
            summary = json.load(f)

        results = summary.get("results", [])
        if not results:
            print(f"No results to export from {json_file}")
            return False

        # Collect all unique metric keys
        all_cbir_keys = set()
        for result in results:
            if result.get("cbir_metrics"):
                all_cbir_keys.update(result["cbir_metrics"].keys())

        cbir_keys = sorted(all_cbir_keys)

        # Write CSV
        fieldnames = ["K", "Silhouette Score"] + cbir_keys
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in sorted(results, key=lambda r: r["k"]):
                row = {
                    "K": result["k"],
                    "Silhouette Score": result.get("silhouette_score", ""),
                }
                if result.get("cbir_metrics"):
                    row.update(result["cbir_metrics"])
                writer.writerow(row)

        print(f"✓ Summary CSV written to: {csv_file}")
        return True

    except Exception as e:
        print(f"Error generating CSV {csv_file}: {e}")
        return False


def print_summary_table(json_file: str) -> None:
    """
    Print a formatted table of sweep results.

    Args:
        json_file: Path to JSON summary file.
    """
    try:
        import json

        with open(json_file, "r") as f:
            summary = json.load(f)

        results = sorted(summary.get("results", []), key=lambda r: r["k"])

        if not results:
            print("No results to display.")
            return

        print("\n" + "=" * 120)
        print(f"{'K':>5} {'Silhouette':>12} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'mAP@1':>8} {'mAP@5':>8} {'mAP@10':>8}")
        print("=" * 120)

        for result in results:
            k = result.get("k", "?")
            sil = result.get("silhouette_score")
            cbir = result.get("cbir_metrics", {})

            sil_str = f"{sil:.4f}" if sil is not None else "N/A"
            r1_str = f"{cbir.get('recall_at_1', 0):.4f}"
            r5_str = f"{cbir.get('recall_at_5', 0):.4f}"
            r10_str = f"{cbir.get('recall_at_10', 0):.4f}"
            m1_str = f"{cbir.get('map_at_1', 0):.4f}"
            m5_str = f"{cbir.get('map_at_5', 0):.4f}"
            m10_str = f"{cbir.get('map_at_10', 0):.4f}"

            print(f"{k:>5} {sil_str:>12} {r1_str:>8} {r5_str:>8} {r10_str:>8} {m1_str:>8} {m5_str:>8} {m10_str:>8}")

        print("=" * 120 + "\n")

    except Exception as e:
        print(f"Error printing summary: {e}")
