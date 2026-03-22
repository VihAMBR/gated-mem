#!/usr/bin/env python3
"""
Unified experiment runner for the gated-memory project.

Every experiment follows the same three-phase pipeline:
  1. GENERATE — Build memories, retrieve, produce LLM answers
  2. EVALUATE — Score answers with BLEU, F1, and LLM judge
  3. REPORT  — Print per-category and overall scores

Usage:
    python run_experiment.py <experiment> [options]

Experiments:
    naive_baseline        Store every conversation turn (no filtering)
    surprise_gated        Store only "surprising" turns (cosine novelty gate)
    enhanced_gated        Surprise gate + temporal bypass + entity novelty bypass
    threshold_sweep       Run surprise_gated across multiple threshold configs
    compare               Print comparison table across all completed runs

Run `python run_experiment.py <experiment> --help` for experiment-specific options.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


RESULTS_DIR = Path("results")
DATASET = "dataset/locomo10.json"


def run_cmd(cmd: list[str], description: str):
    """Run a subprocess, streaming output. Exit on failure."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─'*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


# ─── Experiment: naive_baseline ──────────────────────────────────────────────

def cmd_naive_baseline(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = str(RESULTS_DIR / "naive_baseline_results.json")
    evals_file = str(RESULTS_DIR / "naive_baseline_evals.json")

    generate_args = [
        sys.executable, "run_naive_baseline.py",
        "--dataset", args.dataset,
        "--output", results_file,
        "--top_k", str(args.top_k),
    ]
    if args.max_convs:
        generate_args += ["--max_convs", str(args.max_convs)]

    run_cmd(generate_args, "Phase 1/3: Generate answers (embed → retrieve → LLM)")

    run_cmd([
        sys.executable, "evals.py",
        "--input_file", results_file,
        "--output_file", evals_file,
        "--max_workers", str(args.eval_workers),
    ], "Phase 2/3: Evaluate (BLEU + F1 + LLM Judge)")

    run_cmd([
        sys.executable, "generate_scores.py",
        "--input_path", evals_file,
    ], "Phase 3/3: Report scores")


# ─── Experiment: surprise_gated ──────────────────────────────────────────────

def cmd_surprise_gated(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"t{args.threshold}_{args.mode}_{args.metric}"
    results_file = str(RESULTS_DIR / f"gated_{tag}.json")
    evals_file = str(RESULTS_DIR / f"gated_{tag}_evals.json")

    generate_args = [
        sys.executable, "run_gated_baseline.py",
        "--dataset", args.dataset,
        "--output", results_file,
        "--top_k", str(args.top_k),
        "--threshold", str(args.threshold),
        "--mode", args.mode,
        "--metric", args.metric,
        "--warmup", str(args.warmup),
    ]
    if args.max_convs:
        generate_args += ["--max_convs", str(args.max_convs)]

    run_cmd(generate_args, f"Phase 1/3: Generate answers (gated, {tag})")

    run_cmd([
        sys.executable, "evals.py",
        "--input_file", results_file,
        "--output_file", evals_file,
        "--max_workers", str(args.eval_workers),
    ], "Phase 2/3: Evaluate (BLEU + F1 + LLM Judge)")

    run_cmd([
        sys.executable, "generate_scores.py",
        "--input_path", evals_file,
    ], "Phase 3/3: Report scores")


# ─── Experiment: enhanced_gated ───────────────────────────────────────────────

def cmd_enhanced_gated(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"t{args.threshold}_{args.mode}_{args.metric}"
    results_file = str(RESULTS_DIR / f"enhanced_{tag}.json")
    evals_file = str(RESULTS_DIR / f"enhanced_{tag}_evals.json")

    generate_args = [
        sys.executable, "run_enhanced_gated.py",
        "--dataset", args.dataset,
        "--output", results_file,
        "--top_k", str(args.top_k),
        "--threshold", str(args.threshold),
        "--mode", args.mode,
        "--metric", args.metric,
        "--warmup", str(args.warmup),
        "--answer_workers", str(args.answer_workers),
    ]
    if args.max_convs:
        generate_args += ["--max_convs", str(args.max_convs)]

    run_cmd(generate_args, f"Phase 1/3: Generate answers (enhanced gated, {tag})")

    run_cmd([
        sys.executable, "evals.py",
        "--input_file", results_file,
        "--output_file", evals_file,
        "--max_workers", str(args.eval_workers),
    ], "Phase 2/3: Evaluate (BLEU + F1 + LLM Judge)")

    run_cmd([
        sys.executable, "generate_scores.py",
        "--input_path", evals_file,
    ], "Phase 3/3: Report scores")


# ─── Experiment: threshold_sweep ─────────────────────────────────────────────

SWEEP_CONFIGS = [
    {"threshold": 0.1, "mode": "fixed", "metric": "nearest_neighbor"},
    {"threshold": 0.2, "mode": "fixed", "metric": "nearest_neighbor"},
    {"threshold": 0.3, "mode": "fixed", "metric": "nearest_neighbor"},
    {"threshold": 0.4, "mode": "fixed", "metric": "nearest_neighbor"},
    {"threshold": 0.5, "mode": "fixed", "metric": "nearest_neighbor"},
    {"threshold": 0.3, "mode": "fixed", "metric": "centroid"},
    {"threshold": 0.5, "mode": "fixed", "metric": "centroid"},
    {"threshold": 0.3, "mode": "adaptive", "metric": "nearest_neighbor"},
    {"threshold": 0.5, "mode": "adaptive", "metric": "nearest_neighbor"},
    {"threshold": 0.7, "mode": "adaptive", "metric": "nearest_neighbor"},
]


def cmd_threshold_sweep(args):
    configs = SWEEP_CONFIGS
    if args.quick:
        configs = [
            {"threshold": 0.2, "mode": "fixed", "metric": "nearest_neighbor"},
            {"threshold": 0.3, "mode": "fixed", "metric": "nearest_neighbor"},
            {"threshold": 0.4, "mode": "fixed", "metric": "nearest_neighbor"},
            {"threshold": 0.5, "mode": "adaptive", "metric": "nearest_neighbor"},
        ]

    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'═'*60}")
        print(f"  Sweep [{i}/{total}]: threshold={cfg['threshold']}, mode={cfg['mode']}, metric={cfg['metric']}")
        print(f"{'═'*60}")

        tag = f"t{cfg['threshold']}_{cfg['mode']}_{cfg['metric']}"
        results_file = str(RESULTS_DIR / f"gated_{tag}.json")
        evals_file = str(RESULTS_DIR / f"gated_{tag}_evals.json")

        generate_args = [
            sys.executable, "run_gated_baseline.py",
            "--dataset", args.dataset,
            "--output", results_file,
            "--top_k", str(args.top_k),
            "--threshold", str(cfg["threshold"]),
            "--mode", cfg["mode"],
            "--metric", cfg["metric"],
        ]
        if args.max_convs:
            generate_args += ["--max_convs", str(args.max_convs)]

        run_cmd(generate_args, "Generate answers")

        run_cmd([
            sys.executable, "evals.py",
            "--input_file", results_file,
            "--output_file", evals_file,
            "--max_workers", str(args.eval_workers),
        ], "Evaluate")

    run_cmd([sys.executable, "analyze_results.py"], "Comparison report")


# ─── Experiment: lme (LongMemEval) ───────────────────────────────────────────

LME_DATASET = "LongMemEval/data/longmemeval_s_cleaned.json"


def cmd_lme(args):
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = str(RESULTS_DIR / f"lme_{args.mode}.json")
    scored_file = str(RESULTS_DIR / f"lme_{args.mode}_scored.json")

    generate_args = [
        sys.executable, "run_longmemeval.py",
        "--dataset", args.lme_dataset,
        "--output", results_file,
        "--top_k", str(args.top_k),
        "--mode", args.mode,
        "--answer_workers", str(args.answer_workers),
    ]
    if args.mode != "naive":
        generate_args += [
            "--threshold", str(args.threshold),
            "--gate_mode", args.gate_mode,
        ]
    if args.max_instances:
        generate_args += ["--max_instances", str(args.max_instances)]
    if args.mode == "neuroplastic":
        if not getattr(args, "enable_ltp", True):
            generate_args.append("--no_ltp")
        if not getattr(args, "enable_associations", True):
            generate_args.append("--no_associations")
        if not getattr(args, "enable_inhibition", True):
            generate_args.append("--no_inhibition")
        if not getattr(args, "enable_consolidation", True):
            generate_args.append("--no_consolidation")

    run_cmd(generate_args, f"Phase 1/2: Generate answers (LME, {args.mode})")

    run_cmd([
        sys.executable, "eval_longmemeval.py",
        "--results", results_file,
        "--output", scored_file,
        "--max_workers", str(args.eval_workers),
    ], "Phase 2/2: Judge with LongMemEval prompts")


# ─── Experiment: compare ─────────────────────────────────────────────────────

def cmd_compare(args):
    run_cmd([sys.executable, "analyze_results.py"], "Comparison across all runs")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def add_common_args(parser):
    parser.add_argument("--dataset", default=DATASET, help="Path to LoCoMo dataset JSON")
    parser.add_argument("--top_k", type=int, default=30, help="Number of memories to retrieve per query")
    parser.add_argument("--max_convs", type=int, default=None, help="Limit to first N conversations (for quick testing)")
    parser.add_argument("--eval_workers", type=int, default=2, help="Parallel workers for LLM judge (keep low to avoid rate limits)")


def main():
    parser = argparse.ArgumentParser(
        description="Run memory system experiments on LoCoMo benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="experiment", help="Experiment to run")

    # naive_baseline
    p = subparsers.add_parser("naive_baseline", help="Store every turn, retrieve by similarity")
    add_common_args(p)
    p.set_defaults(func=cmd_naive_baseline)

    # surprise_gated
    p = subparsers.add_parser("surprise_gated", help="Store only surprising turns")
    add_common_args(p)
    p.add_argument("--threshold", type=float, default=0.3, help="Surprise threshold (fixed: absolute, adaptive: percentile)")
    p.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed", help="Threshold mode")
    p.add_argument("--metric", choices=["nearest_neighbor", "centroid"], default="nearest_neighbor", help="Surprise metric")
    p.add_argument("--warmup", type=int, default=3, help="Store first N messages unconditionally")
    p.set_defaults(func=cmd_surprise_gated)

    # enhanced_gated
    p = subparsers.add_parser("enhanced_gated", help="Surprise gate + temporal bypass + entity novelty")
    add_common_args(p)
    p.add_argument("--threshold", type=float, default=0.3, help="Surprise threshold")
    p.add_argument("--mode", choices=["fixed", "adaptive"], default="fixed", help="Threshold mode")
    p.add_argument("--metric", choices=["nearest_neighbor", "centroid"], default="nearest_neighbor", help="Surprise metric")
    p.add_argument("--warmup", type=int, default=3, help="Store first N messages unconditionally")
    p.add_argument("--answer_workers", type=int, default=4, help="Parallel workers for question answering in generate phase")
    p.set_defaults(func=cmd_enhanced_gated)

    # threshold_sweep
    p = subparsers.add_parser("threshold_sweep", help="Run gated encoder across multiple configs")
    add_common_args(p)
    p.add_argument("--quick", action="store_true", help="Run only 4 key configs instead of full 10")
    p.set_defaults(func=cmd_threshold_sweep)

    # lme (LongMemEval)
    p = subparsers.add_parser("lme", help="Run LongMemEval benchmark (naive/enhanced/inhibition/neuroplastic)")
    p.add_argument("--mode", choices=["naive", "enhanced", "inhibition", "neuroplastic"], default="naive")
    p.add_argument("--lme_dataset", default=LME_DATASET, help="Path to LongMemEval JSON")
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--gate_mode", default="fixed")
    p.add_argument("--max_instances", type=int, default=None, help="Limit instances (for testing)")
    p.add_argument("--answer_workers", type=int, default=4)
    p.add_argument("--eval_workers", type=int, default=4)
    p.add_argument("--no_ltp", action="store_true", help="Disable LTP/LTD (neuroplastic mode)")
    p.add_argument("--no_associations", action="store_true", help="Disable associations (neuroplastic mode)")
    p.add_argument("--no_inhibition", action="store_true", help="Disable inhibition (neuroplastic mode)")
    p.add_argument("--no_consolidation", action="store_true", help="Disable consolidation (neuroplastic mode)")
    p.set_defaults(func=cmd_lme, enable_ltp=True, enable_associations=True,
                   enable_inhibition=True, enable_consolidation=True)

    # compare
    p = subparsers.add_parser("compare", help="Print comparison table of all completed runs")
    p.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if not args.experiment:
        parser.print_help()
        print("\nAvailable experiments:")
        print("  naive_baseline     Store every turn, retrieve by similarity")
        print("  surprise_gated     Store only surprising turns (with gating)")
        print("  enhanced_gated     Surprise gate + temporal bypass + entity novelty")
        print("  threshold_sweep    Run gated encoder across multiple configs")
        print("  lme                Run LongMemEval benchmark (naive/enhanced/inhibition)")
        print("  compare            Compare results from all completed runs")
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
