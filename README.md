# Surprise-Gated Memory for Long-Term Conversational Agents

This project investigates whether **selective memory storage** can match or beat **brute-force storage** in long-term conversational AI. The core hypothesis: human memory doesn't record everything — it stores what's *surprising*. Can we apply the same principle to LLM memory systems?

We evaluate all experiments on the [LoCoMo benchmark](https://github.com/memodb-io/memobase), which tests how well a memory system answers questions about multi-session conversations across four categories: single-hop, temporal, multi-hop, and open-domain.

## Setup

```bash
# Clone and install
git clone https://github.com/yourusername/gated-mem.git
cd gated-mem
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Requirements:** Python 3.10+, an OpenAI API key (for GPT-4o-mini).

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run (~80MB).

## Quick Start

```bash
# Verify the pipeline works (5 questions, ~30 seconds)
python test_quick.py

# Run an experiment on 2 conversations (~5 min + ~10 min eval)
python run_experiment.py naive_baseline --max_convs 2

# Run a gated experiment
python run_experiment.py surprise_gated --threshold 0.3 --max_convs 2

# Compare all completed runs
python run_experiment.py compare
```

---

## Experiments

### Experiment 1: Naive Baseline

**Question:** What score does a simple "store everything, retrieve by similarity" system get on LoCoMo?

**How it works:**

1. Every conversation turn becomes a memory: `"<timestamp> | <Speaker>: <text>"`
2. Each memory is embedded with `all-MiniLM-L6-v2` (384-dim, L2-normalized)
3. Embeddings are stored in a FAISS `IndexFlatIP` index (brute-force cosine similarity)
4. At query time, top-30 most similar memories are retrieved per speaker
5. Retrieved memories + question are passed to GPT-4o-mini using Mem0's exact answer prompt
6. Answers are scored by BLEU, F1, and an LLM judge (GPT-4o-mini, Mem0's judge prompt)

**Run:**

```bash
# Full benchmark (10 conversations, ~1540 questions, ~60 min + eval time)
python run_experiment.py naive_baseline

# Quick test (2 conversations, ~233 questions, ~5 min + eval time)
python run_experiment.py naive_baseline --max_convs 2
```

**Expected output:** `results/naive_baseline_results.json` and `results/naive_baseline_evals.json`

**Expected scores:** ~60-66% overall LLM judge accuracy. This is the control — every other experiment is measured against this.

**Key files:**
- `run_naive_baseline.py` — The `NaiveMemoryBaseline` class
- `metrics/llm_judge.py` — LLM judge scoring
- `metrics/utils.py` — BLEU and F1 calculation

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_k` | 30 | Number of memories retrieved per speaker per question |
| `--max_convs` | None | Limit to first N conversations |
| `--dataset` | `dataset/locomo10.json` | Path to LoCoMo dataset |
| `--eval_workers` | 2 | Parallel threads for LLM judge (keep low to avoid rate limits) |

**Design decisions:**
- We use Mem0's exact answer prompt and judge prompt for apples-to-apples comparison with published benchmarks. Different prompts can swing scores by 20+ points.
- Separate memory banks per speaker, matching how Mem0/Memobase organize memories.
- `top_k=30` is high — it costs more tokens but ensures the relevant memory is likely in the context. Lower values trade recall for cost.
- Category 5 (adversarial) questions are skipped, matching benchmark conventions.

---

### Experiment 2: Surprise-Gated Memory

**Question:** If we only store messages that are *surprising* (novel relative to existing memories), does retrieval quality hold up while using less storage?

**How it works:**

Same pipeline as the naive baseline, except before storing a turn, it passes through a `SurpriseGatedEncoder`:

1. Embed the incoming message
2. Compare against all previously stored memory embeddings
3. Compute a **surprise score**: how different is this from what we've already seen?
4. If surprise ≥ threshold → **store it**
5. If surprise < threshold → **skip it** (considered redundant)

Each speaker gets their own independent gate — what's surprising for speaker A depends on A's history, not B's.

**Surprise metrics:**
- **`nearest_neighbor`**: surprise = `1 - max(cosine_similarity)` to any stored memory. Measures novelty against the single closest existing memory.
- **`centroid`**: surprise = `1 - cosine_similarity(embedding, mean_of_all_stored)`. Measures novelty against the "average topic" of all stored memories.

**Threshold modes:**
- **`fixed`**: A static value (e.g., 0.3). Anything below this similarity-distance is filtered out.
- **`adaptive`**: The threshold is the Nth percentile of all surprise scores seen so far. For example, `threshold=0.5` means "only store messages more surprising than the median." This auto-calibrates to each conversation's natural variability.

**Run:**

```bash
# Single configuration
python run_experiment.py surprise_gated \
    --threshold 0.3 \
    --mode fixed \
    --metric nearest_neighbor \
    --max_convs 2

# Different configurations
python run_experiment.py surprise_gated --threshold 0.2 --mode fixed --metric nearest_neighbor
python run_experiment.py surprise_gated --threshold 0.4 --mode fixed --metric centroid
python run_experiment.py surprise_gated --threshold 0.5 --mode adaptive --metric nearest_neighbor
```

**Expected output:** `results/gated_t{threshold}_{mode}_{metric}.json` and corresponding `_evals.json`

**Key files:**
- `surprise_gated_encoder.py` — The `SurpriseGatedEncoder` class and `GateStats`
- `run_gated_baseline.py` — The `GatedMemorySystem` class (integrates gating into the pipeline)

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.3 | Surprise cutoff. Fixed mode: absolute (0=store all, 1=store none). Adaptive mode: percentile (0.5=median). |
| `--mode` | `fixed` | `fixed` or `adaptive` |
| `--metric` | `nearest_neighbor` | `nearest_neighbor` or `centroid` |
| `--warmup` | 3 | Always store the first N messages unconditionally (gate needs some data before it can compare) |
| `--top_k` | 30 | Memories retrieved per query |
| `--max_convs` | None | Limit conversations |

**What the gate output looks like:**

```
Gate: A stored 88/211 (58% compressed), B stored 82/208 (61% compressed)
```

This means speaker A said 211 things, but only 88 were surprising enough to keep.

---

### Experiment 3: Threshold Sweep

**Question:** How does retrieval quality degrade as we increase the surprise threshold (store less)?

This is a batch runner that executes Experiment 2 across multiple configurations and produces a comparison table.

**Run:**

```bash
# Full sweep (10 configurations)
python run_experiment.py threshold_sweep --max_convs 2

# Quick sweep (4 key configurations)
python run_experiment.py threshold_sweep --quick --max_convs 2
```

**Configurations tested:**

| # | Threshold | Mode | Metric | What it tests |
|---|-----------|------|--------|---------------|
| 1 | 0.1 | fixed | nearest_neighbor | Very permissive — almost everything stored |
| 2 | 0.2 | fixed | nearest_neighbor | Light filtering |
| 3 | 0.3 | fixed | nearest_neighbor | Moderate filtering |
| 4 | 0.4 | fixed | nearest_neighbor | Aggressive filtering |
| 5 | 0.5 | fixed | nearest_neighbor | Very aggressive |
| 6 | 0.3 | fixed | centroid | Centroid vs NN comparison |
| 7 | 0.5 | fixed | centroid | Centroid at high threshold |
| 8 | 0.3 | adaptive | nearest_neighbor | Adaptive low percentile |
| 9 | 0.5 | adaptive | nearest_neighbor | Adaptive median |
| 10 | 0.7 | adaptive | nearest_neighbor | Adaptive high percentile |

**Expected output:** One `results/gated_*_evals.json` per configuration, plus a comparison table printed at the end.

---

### Experiment 4: Compare

**Question:** How do all completed experiments compare to each other?

Reads all `*_evals.json` files in `results/` and prints a comparison table with LLM judge scores per category and compression ratios.

```bash
python run_experiment.py compare
```

No API calls — this just reads existing result files.

---

## Results (2-conversation subset)

| Config | Compression | LLM Score | vs Baseline |
|--------|------------|-----------|-------------|
| Naive Baseline | 0% | **66.1%** | — |
| t=0.2 fixed NN | 48.2% | 56.2% | -14.9% |
| t=0.3 fixed NN | 78.4% | 37.8% | -42.8% |
| t=0.4 fixed NN | 91.2% | 28.8% | -56.5% |
| t=0.5 adaptive NN | 58.0% | 52.8% | -20.1% |

**Key findings:**
- The quality/compression tradeoff is steep. At t=0.2, we store half the messages but lose ~15% accuracy.
- **Temporal questions are hit hardest** (74.6% → 17.5% at t=0.4). Routine daily updates with timestamps get filtered as "not surprising," but temporal questions depend on exactly those updates.
- **Multi-hop questions are surprisingly robust** (61.5% across all configs). Distinctive facts that multi-hop questions need are inherently "surprising" enough to pass the gate.
- **Adaptive thresholding** lands between t=0.2 and t=0.3 in both compression and quality — a reasonable middle ground but doesn't outperform a well-tuned fixed threshold.

---

## Project Structure

```
gated-mem/
├── run_experiment.py          # Unified entry point for all experiments
├── run_naive_baseline.py      # Experiment 1: naive baseline system
├── run_gated_baseline.py      # Experiment 2: surprise-gated system
├── surprise_gated_encoder.py  # Core: SurpriseGatedEncoder class
├── evals.py                   # Evaluation pipeline (BLEU + F1 + LLM judge)
├── generate_scores.py         # Per-category score aggregation
├── analyze_results.py         # Cross-experiment comparison table
├── test_quick.py              # 5-question smoke test
├── prompts.py                 # Prompt variants (Mem0, graph, Zep)
├── metrics/
│   ├── llm_judge.py           # GPT-4o-mini judge (CORRECT/WRONG)
│   └── utils.py               # BLEU, F1 calculations
├── dataset/
│   └── locomo10.json          # LoCoMo benchmark (10 conversations)
├── results/                   # Output directory (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Pipeline Architecture

```
dataset/locomo10.json
         │
         ├──► run_naive_baseline.py ──────────────────────► results/*_results.json
         │         (store all turns)                              │
         │                                                        │
         ├──► run_gated_baseline.py ──────────────────────► results/gated_*.json
         │         │                                              │
         │         └── surprise_gated_encoder.py                  │
         │              (filter by novelty)                        │
         │                                                        ▼
         │                                                   evals.py
         │                                              (BLEU + F1 + LLM Judge)
         │                                                        │
         │                                                        ▼
         │                                              results/*_evals.json
         │                                                        │
         │                                            ┌───────────┴───────────┐
         │                                            ▼                       ▼
         │                                    generate_scores.py      analyze_results.py
         │                                    (per-category)          (cross-experiment)
```

## Running Individual Components

If you need finer control, you can run each phase separately:

```bash
# Phase 1: Generate answers only
python run_naive_baseline.py --dataset dataset/locomo10.json --output results/my_run.json

# Phase 2: Evaluate answers only
python evals.py --input_file results/my_run.json --output_file results/my_run_evals.json --max_workers 2

# Phase 3: Print scores only
python generate_scores.py --input_path results/my_run_evals.json
```

Both `run_naive_baseline.py` and `run_gated_baseline.py` support **resume** — if interrupted, re-run the same command and it will skip already-completed conversations.

## Rate Limits

OpenAI API rate limits are the main bottleneck. Both the answer generation and LLM judge phases call GPT-4o-mini. The code has built-in exponential backoff (up to 60s wait, 8 retries). If you're hitting limits:

- Use `--max_convs 2` to test with fewer conversations
- Use `--eval_workers 1` (default is 2) to reduce judge parallelism
- Expect ~1-2 questions/second under normal conditions, dropping during rate limits
