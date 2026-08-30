#!/usr/bin/env python3
"""
Post-sweep analysis for BoVW K-sweep results.

Generates summary tables, CSV exports, and basic visualizations from the
consolidated results JSON file.

Usage:
    python3 scripts/analyze_k_sweep.py
    python3 scripts/analyze_k_sweep.py --summary-file outputs/bovw_k_sweep/k_sweep_results.json
    python3 scripts/analyze_k_sweep.py --export-csv
    python3 scripts/analyze_k_sweep.py --plot
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.k_sweep_utilities import (
    generate_summary_csv,
    print_summary_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BoVW K-sweep results"
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default="outputs/bovw_k_sweep/k_sweep_results.json",
        help="Path to sweep results JSON file",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bovw_k_sweep",
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    summary_file = Path(args.summary_file)
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    with open(summary_file, "r") as f:
        summary = json.load(f)

    print(f"\nLoaded summary from: {summary_file}")
    print(f"Results for K values: {summary.get('sweep_metadata', {}).get('k_values', [])}")

    # Print table
    print_summary_table(str(summary_file))

    # Export CSV
    if args.export_csv:
        csv_file = output_dir / "k_sweep_results.csv"
        if generate_summary_csv(str(summary_file), str(csv_file)):
            print(f"✓ CSV exported to: {csv_file}")

    # Generate plots
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            results = sorted(
                summary.get("results", []),
                key=lambda r: r["k"]
            )

            if not results:
                print("No results to plot.")
                return 0

            k_values = [r["k"] for r in results]
            silhouette_scores = [
                r.get("silhouette_score") for r in results
            ]
            recall_at_1 = [
                r.get("cbir_metrics", {}).get("recall_at_1", 0) for r in results
            ]
            map_at_1 = [
                r.get("cbir_metrics", {}).get("map_at_1", 0) for r in results
            ]

            # Plot 1: Silhouette score vs K
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].plot(k_values, silhouette_scores, "o-", linewidth=2, markersize=8)
            axes[0].set_xlabel("Vocabulary Size (K)")
            axes[0].set_ylabel("Silhouette Score")
            axes[0].set_title("Silhouette Score vs K")
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Recall@1 vs K
            axes[1].plot(k_values, recall_at_1, "s-", linewidth=2, markersize=8, color="green")
            axes[1].set_xlabel("Vocabulary Size (K)")
            axes[1].set_ylabel("Recall@1")
            axes[1].set_title("CBIR Recall@1 vs K (AID)")
            axes[1].grid(True, alpha=0.3)

            # Plot 3: mAP@1 vs K
            axes[2].plot(k_values, map_at_1, "^-", linewidth=2, markersize=8, color="red")
            axes[2].set_xlabel("Vocabulary Size (K)")
            axes[2].set_ylabel("mAP@1")
            axes[2].set_title("CBIR mAP@1 vs K (AID)")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = output_dir / "k_sweep_plots.png"
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            print(f"✓ Plots saved to: {plot_file}")
            plt.close()

        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            print("Install with: pip install matplotlib")

    print(f"\n✓ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
