"""
CLI entry point for base-sequence-toolkit.

Usage:
    bst-analyze swe-agent -n 500 -o results/
    bst-analyze duncrew /path/to/exec_traces/ -o results/
    bst-analyze json /path/to/data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_swe_agent(args: argparse.Namespace) -> None:
    from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface
    from base_sequence_toolkit.core.analyzer import run_full_analysis

    print(f"Loading SWE-agent trajectories (max {args.max_records})...")
    results = load_from_huggingface(max_records=args.max_records)
    print(f"Classified {len(results)} trajectories\n")

    sequences = [r.base_sequence for r in results]
    outcomes = [r.resolved for r in results]

    report = run_full_analysis(sequences, outcomes, min_pattern_count=max(len(results) // 20, 5))
    print(report.format_summary())

    if args.output_dir:
        _save_results(results, report, args.output_dir)


def cmd_duncrew(args: argparse.Namespace) -> None:
    from base_sequence_toolkit.adapters.duncrew import load_traces
    from base_sequence_toolkit.core.analyzer import run_full_analysis

    print(f"Loading DunCrew traces from {args.trace_dir}...")
    results = load_traces(args.trace_dir, max_samples=args.max_records)
    print(f"Loaded {len(results)} traces\n")

    sequences = [r.base_sequence for r in results]
    outcomes = [r.resolved for r in results]

    report = run_full_analysis(sequences, outcomes, min_pattern_count=max(len(results) // 20, 5))
    print(report.format_summary())

    if args.output_dir:
        _save_results(results, report, args.output_dir)


def cmd_json(args: argparse.Namespace) -> None:
    """Analyze a JSON file with pre-computed sequences.

    Expected format: list of {base_sequence: str, resolved: bool}
    """
    from base_sequence_toolkit.core.analyzer import run_full_analysis

    with open(args.json_file, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        sequences = [d["base_sequence"] for d in data]
        outcomes = [d["resolved"] for d in data]
    else:
        print("Error: JSON file must contain a list of objects with 'base_sequence' and 'resolved' fields.")
        sys.exit(1)

    print(f"Loaded {len(sequences)} sequences\n")
    report = run_full_analysis(sequences, outcomes, min_pattern_count=max(len(sequences) // 20, 5))
    print(report.format_summary())


def _save_results(results: list, report, output_dir: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = [r.to_dict() for r in results]
    traj_file = out_path / "trajectories.json"
    with open(traj_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records to {traj_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bst-analyze",
        description="XEPV Base Sequence Analysis Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Analysis source")

    # swe-agent subcommand
    swe_parser = subparsers.add_parser("swe-agent", help="Analyze SWE-agent trajectories from HuggingFace")
    swe_parser.add_argument("-n", "--max-records", type=int, default=500)
    swe_parser.add_argument("-o", "--output-dir", type=str, default=None)

    # duncrew subcommand
    dc_parser = subparsers.add_parser("duncrew", help="Analyze DunCrew execution traces")
    dc_parser.add_argument("trace_dir", type=str, help="Path to exec_traces directory")
    dc_parser.add_argument("-n", "--max-records", type=int, default=None)
    dc_parser.add_argument("-o", "--output-dir", type=str, default=None)

    # json subcommand
    json_parser = subparsers.add_parser("json", help="Analyze pre-computed JSON data")
    json_parser.add_argument("json_file", type=str, help="Path to JSON file")

    args = parser.parse_args()

    if args.command == "swe-agent":
        cmd_swe_agent(args)
    elif args.command == "duncrew":
        cmd_duncrew(args)
    elif args.command == "json":
        cmd_json(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
