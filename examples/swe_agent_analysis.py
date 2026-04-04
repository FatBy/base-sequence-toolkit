#!/usr/bin/env python3
"""
End-to-end example: Analyze SWE-agent trajectories with the Base Sequence Toolkit.

This script demonstrates the full analysis pipeline:
  1. Load SWE-agent trajectories from HuggingFace
  2. Classify each trajectory into XEPV base sequences
  3. Run comprehensive analysis (distributions, transitions, patterns)
  4. Print a formatted report
  5. Optionally save detailed JSON results

Requirements:
    pip install base-sequence-toolkit[swe-agent]

Usage:
    python swe_agent_analysis.py                # Analyze 500 trajectories
    python swe_agent_analysis.py -n 2000        # Analyze 2000 trajectories
    python swe_agent_analysis.py -n 500 -o out/ # Save JSON results to out/
"""

import argparse
import json
from pathlib import Path

from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface
from base_sequence_toolkit.core.analyzer import (
    analyze_positions,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
    run_full_analysis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SWE-agent trajectories using XEPV base sequence framework"
    )
    parser.add_argument(
        "-n", "--max-records", type=int, default=500,
        help="Maximum number of trajectories to process (default: 500)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Directory to save detailed JSON results",
    )
    args = parser.parse_args()

    # ── Step 1: Load and classify ──
    print(f"Loading SWE-agent trajectories (max {args.max_records})...")
    print("(This downloads from HuggingFace on first run)\n")

    results = load_from_huggingface(max_records=args.max_records)
    print(f"Classified {len(results)} trajectories\n")

    # ── Step 2: Extract sequences and outcomes ──
    sequences = [r.base_sequence for r in results]
    outcomes = [r.resolved for r in results]

    # ── Step 3: Run full analysis ──
    report = run_full_analysis(
        sequences=sequences,
        outcomes=outcomes,
        min_pattern_count=max(len(results) // 20, 5),
    )

    # ── Step 4: Print report ──
    print(report.format_summary())

    # ── Step 5: Additional SWE-agent specific insights ──
    print("\n" + "=" * 70)
    print("SWE-AGENT SPECIFIC INSIGHTS")
    print("=" * 70)

    # Model breakdown
    from collections import Counter
    model_counts = Counter(r.model_name for r in results)
    print("\nModel distribution:")
    for model, count in model_counts.most_common():
        resolved = sum(1 for r in results if r.model_name == model and r.resolved)
        print(f"  {model}: {count} traces, {resolved}/{count} resolved ({100 * resolved / count:.0f}%)")

    # Sequence length analysis
    res_lengths = [r.step_count for r in results if r.resolved]
    unres_lengths = [r.step_count for r in results if not r.resolved]
    if res_lengths:
        print(f"\nResolved: avg steps={sum(res_lengths) / len(res_lengths):.1f}, "
              f"median={sorted(res_lengths)[len(res_lengths) // 2]}")
    if unres_lengths:
        print(f"Unresolved: avg steps={sum(unres_lengths) / len(unres_lengths):.1f}, "
              f"median={sorted(unres_lengths)[len(unres_lengths) // 2]}")

    # Sample sequences
    print("\nShortest resolved sequences:")
    resolved_sorted = sorted([r for r in results if r.resolved], key=lambda r: r.step_count)
    for r in resolved_sorted[:5]:
        print(f"  [{r.step_count:2d}] {r.base_sequence[:60]}  ({r.instance_id})")

    # ── Step 6: Save results ──
    if args.output_dir:
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Per-trajectory data
        records = [r.to_dict() for r in results]
        traj_file = out_path / "swe_xepv_trajectories.json"
        with open(traj_file, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nSaved {len(records)} trajectory records to {traj_file}")

        # Pattern analysis
        patterns = find_discriminative_patterns(
            [r.base_sequence for r in results if r.resolved],
            [r.base_sequence for r in results if not r.resolved],
        )
        pattern_data = [
            {
                "pattern": p.pattern,
                "lift": p.lift,
                "resolved_count": p.resolved_count,
                "unresolved_count": p.unresolved_count,
            }
            for p in patterns[:50]
        ]
        pattern_file = out_path / "swe_xepv_patterns.json"
        with open(pattern_file, "w") as f:
            json.dump(pattern_data, f, indent=2)
        print(f"Saved pattern analysis to {pattern_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
