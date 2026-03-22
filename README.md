# Surprise-Gated Memory for Long-Term Conversational Agents

This project investigates whether **selective memory storage** can match or beat **brute-force storage** in long-term conversational AI. The core hypothesis: human memory doesn't record everything — it stores what's *surprising*. Can we apply the same principle to LLM memory systems?

We evaluate all experiments on the [LoCoMo benchmark](https://github.com/memodb-io/memobase), which tests how well a memory system answers questions about multi-session conversations across four categories: single-hop, temporal, multi-hop, and open-domain.

## Setup

```bash
# Clone and install
git clone https://github.com/VihAMBR/gated-mem.git
cd gated-mem
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

```bash
# Download spaCy model (needed for enhanced gated experiments)
python -m spacy download en_core_web_sm
```

**Requirements:** Python 3.10+, an OpenAI API key (for GPT-4o-mini).

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run (~80MB). The spaCy model (`en_core_web_sm`) must be downloaded manually (see above).

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

### Experiment 3: Enhanced Gated Memory

**Question:** Can we recover temporal question accuracy (destroyed by pure surprise-gating) while keeping the storage savings?

**How it works:**

Same surprise-gating pipeline, but with two bypass mechanisms that let critical messages skip the gate:

1. **Temporal bypass** — A regex-based detector catches date patterns (`May 7, 2023`, `last Monday`, `three months ago`), time expressions (`at 3pm`, `in the morning`), and temporal keywords (`started`, `began`, `moved`, `changed`, `recently`, `just`, `ago`, `since`). Any message triggering the detector is stored unconditionally.

2. **Entity novelty bypass** — spaCy's `en_core_web_sm` model extracts named entities (PERSON, ORG, GPE, LOC, etc.) per speaker. A running set of seen entities is maintained. When a message introduces an entity that speaker hasn't mentioned before, it's stored regardless of surprise score.

The storage decision becomes:

```
store = (surprise > threshold) OR has_temporal_markers OR has_novel_entities
```

Every stored memory is a `MemoryRecord` with metadata fields for future experiments:
- `retrieval_weight` (1.0), `retrieval_count` (0), `last_retrieved` (null)
- `associations` (empty), `inhibited_by` (null), `inhibition_weight` (0.0)
- `created_at`, `surprise_score`, `temporal_salience`, `entity_novelty`

Retrieval uses weighted scoring: `cosine_similarity * retrieval_weight * (1 - inhibition_weight)`. Since weights are initialized to neutral values, this is currently equivalent to raw cosine — but the plumbing is ready for decay, interference, and consolidation experiments.

**Run:**

```bash
python run_experiment.py enhanced_gated \
    --threshold 0.2 \
    --mode fixed \
    --metric nearest_neighbor \
    --max_convs 2
```

**Expected output:** `results/enhanced_t{threshold}_{mode}_{metric}.json` and `_evals.json`

**Key files:**
- `enhanced_gated_encoder.py` — `EnhancedGatedEncoder`, `TemporalDetector`, `EntityTracker`, `MemoryRecord`
- `run_enhanced_gated.py` — `EnhancedMemorySystem` with weighted retrieval

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 0.3 | Surprise cutoff (same as Experiment 2) |
| `--mode` | `fixed` | `fixed` or `adaptive` |
| `--metric` | `nearest_neighbor` | `nearest_neighbor` or `centroid` |
| `--warmup` | 3 | Always store first N messages |
| `--top_k` | 30 | Memories retrieved per query |
| `--max_convs` | None | Limit conversations |

**What the gate output looks like:**

```
Gate A: 143/211 stored (32% compressed) — surprise=102, temporal=46, entity=21, warmup=3
Gate B: 133/208 stored (36% compressed) — surprise=113, temporal=37, entity=20, warmup=3
```

This shows the bypass breakdown: of 143 stored messages for speaker A, 102 passed the surprise gate, 46 were saved by temporal detection, and 21 by entity novelty (categories overlap — a message can trigger multiple pathways).

---

### Experiment 4: Threshold Sweep

**Question:** How does retrieval quality degrade as we increase the surprise threshold (store less)?

This is a batch runner that executes the surprise-gated experiment across multiple configurations and produces a comparison table.

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

### Experiment 5: Compare

**Question:** How do all completed experiments compare to each other?

Reads all `*_evals.json` files in `results/` and prints a comparison table with LLM judge scores per category and compression ratios.

```bash
python run_experiment.py compare
```

No API calls — this just reads existing result files.

---

## Results (Full 10-conversation LoCoMo benchmark, 1540 questions)

| Config | Memories | Compression | Overall | Single-hop | Temporal | Multi-hop | Open-domain | vs Baseline |
|--------|----------|------------|---------|------------|----------|-----------|-------------|-------------|
| Naive Baseline | 5882/5882 | 0% | **62.0%** | 54.3% | 60.7% | 44.8% | 67.1% | — |
| Surprise t=0.2 | 3808/5882 | 35.3% | 57.9% | 55.3% | 49.8% | 43.8% | 63.5% | -4.1% |
| **Enhanced t=0.2** | **3700/5314** | **30.4%** | **61.6%** | **57.1%** | **59.5%** | **44.8%** | **65.9%** | **-0.4%** |

**Key findings:**

- **Enhanced gating matches the baseline while storing 30% less data.** The temporal and entity bypasses recover the information that pure surprise-gating destroys. At 61.6% vs 62.0%, the difference is 0.4% — within noise on 1540 questions.

- **Temporal questions nearly fully recovered** (49.8% → 59.5%, vs baseline 60.7%). Pure surprise-gating at t=0.2 dropped temporal by 11 points because routine timestamped updates look "unsurprising." The temporal bypass reclaims most of that loss.

- **Single-hop actually improves** (54.3% → 57.1%). The entity novelty bypass captures factual mentions that single-hop questions target, which pure cosine similarity sometimes misses.

- **Multi-hop holds steady** at 44.8%, exactly matching baseline. Distinctive facts that multi-hop questions chain together are inherently "surprising" enough to pass the gate even without bypasses.

- **The quality/compression tradeoff is steep for pure surprise-gating** (57.9% at 35% compression). But the enhanced approach breaks this tradeoff — it compresses 30% while losing only 0.4% quality, compared to 4.1% loss from pure surprise gating at similar compression.

- **Bypass breakdown across all conversations**: 965 messages saved by temporal detection, 463 by entity novelty (out of ~5300 total messages). These bypasses are why the enhanced encoder recovers temporal and single-hop accuracy without sacrificing compression.

---

## Project Structure

```
gated-mem/
├── run_experiment.py            # Unified entry point for all experiments
├── run_naive_baseline.py        # Experiment 1: naive baseline system
├── run_gated_baseline.py        # Experiment 2: surprise-gated system
├── run_enhanced_gated.py        # Experiment 3: enhanced gated system
├── surprise_gated_encoder.py    # Core: SurpriseGatedEncoder class
├── enhanced_gated_encoder.py    # Enhanced: temporal bypass + entity novelty + MemoryRecord
├── evals.py                     # Evaluation pipeline (BLEU + F1 + LLM judge)
├── generate_scores.py           # Per-category score aggregation
├── analyze_results.py           # Cross-experiment comparison table
├── test_quick.py                # 5-question smoke test
├── prompts.py                   # Prompt variants (Mem0, graph, Zep)
├── metrics/
│   ├── llm_judge.py             # GPT-4o-mini judge (CORRECT/WRONG)
│   └── utils.py                 # BLEU, F1 calculations
├── dataset/
│   └── locomo10.json            # LoCoMo benchmark (10 conversations)
├── results/                     # Benchmark results (committed)
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
         │                                                        │
         ├──► run_enhanced_gated.py ──────────────────────► results/enhanced_*.json
         │         │                                              │
         │         └── enhanced_gated_encoder.py                  │
         │              (surprise + temporal + entity)             │
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
