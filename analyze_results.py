"""
Analyze and compare results across all runs (baseline + gated variants).
Produces a summary table of scores and storage efficiency.

Usage:
    python analyze_results.py
"""

import json
import os
import pandas as pd
from pathlib import Path

CATEGORIES = {1: "single_hop", 2: "temporal", 3: "multi_hop", 4: "open_domain"}
RESULTS_DIR = Path("results")


def load_eval_scores(eval_path: str) -> dict:
    with open(eval_path) as f:
        data = json.load(f)

    all_items = []
    for key in data:
        if key == "gate_info":
            continue
        all_items.extend(data[key])

    if not all_items:
        return {}

    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    scores = {"overall_llm": df["llm_score"].mean()}
    for cat_id, cat_name in CATEGORIES.items():
        cat_df = df[df["category"] == cat_id]
        if len(cat_df) > 0:
            scores[cat_name] = cat_df["llm_score"].mean()
        else:
            scores[cat_name] = None

    scores["num_questions"] = len(df)
    return scores


def load_gate_info(results_path: str) -> dict | None:
    with open(results_path) as f:
        data = json.load(f)

    if "gate_info" not in data:
        return None

    gate_info = data["gate_info"]
    total_seen = sum(
        g["speaker_a"]["total_seen"] + g["speaker_b"]["total_seen"]
        for g in gate_info.values()
    )
    total_stored = sum(
        g["speaker_a"]["total_stored"] + g["speaker_b"]["total_stored"]
        for g in gate_info.values()
    )

    return {
        "total_seen": total_seen,
        "total_stored": total_stored,
        "compression": 1.0 - total_stored / total_seen if total_seen > 0 else 0,
    }


def main():
    rows = []

    # Baseline
    baseline_eval = RESULTS_DIR / "naive_baseline_evals.json"
    if baseline_eval.exists():
        scores = load_eval_scores(str(baseline_eval))
        rows.append({
            "run": "naive_baseline",
            "threshold": "-",
            "mode": "-",
            "metric": "-",
            "memories_stored": "ALL",
            "compression": "0%",
            **{k: f"{v:.1%}" if isinstance(v, float) else v for k, v in scores.items()},
        })

    # Gated runs (surprise-gated and enhanced)
    eval_files = sorted(RESULTS_DIR.glob("gated_t*_evals.json")) + sorted(RESULTS_DIR.glob("enhanced_t*_evals.json"))
    for results_file in eval_files:
        name = results_file.stem.replace("_evals", "")
        source_file = RESULTS_DIR / f"{name}.json"

        scores = load_eval_scores(str(results_file))
        gate = load_gate_info(str(source_file)) if source_file.exists() else None

        if name.startswith("enhanced_t"):
            prefix = "enhanced_t"
        else:
            prefix = "gated_t"
        parts = name.replace(prefix, "").split("_", 2)
        threshold = parts[0] if parts else "?"
        mode = parts[1] if len(parts) > 1 else "?"
        metric = parts[2] if len(parts) > 2 else "?"

        row = {
            "run": name,
            "threshold": threshold,
            "mode": mode,
            "metric": metric,
        }

        if gate:
            row["memories_stored"] = f"{gate['total_stored']}/{gate['total_seen']}"
            row["compression"] = f"{gate['compression']:.1%}"
        else:
            row["memories_stored"] = "?"
            row["compression"] = "?"

        for k, v in scores.items():
            row[k] = f"{v:.1%}" if isinstance(v, float) else v

        rows.append(row)

    if not rows:
        print("No results found. Run the benchmark first.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  COMPARISON: Naive Baseline vs Surprise-Gated Variants")
    print("=" * 100)

    display_cols = ["run", "threshold", "mode", "metric", "memories_stored",
                    "compression", "overall_llm", "single_hop", "temporal",
                    "multi_hop", "open_domain"]
    display_cols = [c for c in display_cols if c in df.columns]

    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("  KEY QUESTION: Does storing less hurt or help retrieval quality?")
    print("=" * 100)

    if len(rows) > 1:
        baseline_score = rows[0].get("overall_llm", "?")
        print(f"\n  Baseline overall LLM score: {baseline_score}")
        for row in rows[1:]:
            diff_str = ""
            try:
                b = float(baseline_score.strip("%")) / 100
                g = float(row["overall_llm"].strip("%")) / 100
                diff = g - b
                diff_str = f" ({diff:+.1%} vs baseline)"
            except (ValueError, KeyError):
                pass
            print(f"  {row['run']}: {row.get('overall_llm', '?')} @ {row.get('compression', '?')} compression{diff_str}")


if __name__ == "__main__":
    main()
