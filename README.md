# Neuroplastic Memory for Long-Term Conversational Agents

A biologically-inspired memory system for LLM agents. Instead of storing everything and retrieving by static similarity, this system **selectively encodes** what matters and **reorganizes itself through use** — memories that prove useful get stronger, outdated beliefs get suppressed, and frequently co-retrieved facts become linked.

> **Status:** Core encoding experiments complete on both benchmarks. Neuroplastic mechanism runs in progress on LongMemEval. Results tables will be updated as runs finish.

## Results Summary

### RQ1: Does selective encoding work?

| System | Compression | LoCoMo | LongMemEval |
|--------|-------------|--------|-------------|
| A: Naive (store all) | 0% | 62.0% | 72.4% |
| B: Surprise-gated (t=0.2) | 35% | 57.9% | *running* |
| C: Multi-signal gated (t=0.2) | 30% | 61.6% | *running* |

Multi-signal gating matches the naive baseline within 0.4% on LoCoMo while storing 30% less. Pure surprise-gating loses 4.1% — the temporal and entity bypasses recover what it destroys.

### RQ2: Does neuroplasticity improve memory quality?

| System | LME Overall | LME-KU | LME-TR | LME-MR |
|--------|-------------|--------|--------|--------|
| C: Multi-signal (base) | *running* | | | |
| C + Inhibition | *running* | | | |
| C + All (Neuroplastic) | *running* | | | |

*Runs in progress. Hypothesis: inhibition improves knowledge-update accuracy; consolidation and associations improve multi-session reasoning.*

## Context: How Other Systems Score

### LoCoMo Benchmark

| System | Overall | Ingestion Cost |
|--------|---------|----------------|
| Mem0 | ~61% | LLM calls per turn |
| Zep | ~58-75% | Graph construction |
| Memobase | ~76% | LLM summarization |
| **gated-mem (multi-signal)** | **62%** | **Embedding only** |

Our multi-signal encoder matches Mem0 while using zero LLM calls during ingestion. Memobase and higher-scoring systems use expensive LLM summarization at every turn.

### LongMemEval Benchmark

| System | Overall | Source |
|--------|---------|--------|
| GPT-4o (full context) | ~60-64% | LongMemEval paper |
| ReadAgent | ~55% | LongMemEval paper |
| **gated-mem (naive)** | **72.4%** | This repo |

Our naive baseline already outperforms GPT-4o with full context window on LongMemEval_S. This validates the retrieve-then-answer architecture over brute-force context stuffing.

---

## Research Questions

### RQ1: Selective Encoding

**Does filtering messages at encoding time preserve retrieval quality while reducing storage?**

Three memory systems, each tested on both benchmarks:

**System A: Naive Baseline.** Store every conversation turn. No filtering. The control group.

- Every turn becomes a memory: `"<timestamp> | <Speaker>: <text>"`
- Embedded with `all-MiniLM-L6-v2` (384-dim), indexed in FAISS (`IndexFlatIP`)
- Top-30 most similar memories retrieved per query, passed to GPT-4o-mini
- LoCoMo uses Mem0's exact answer/judge prompts; LongMemEval uses its official per-type judge prompts

**System B: Surprise-Gated.** Only store messages that exceed a surprise threshold.

- Surprise = `1 - max(cosine_similarity)` to any existing stored memory
- If surprise >= threshold, store. Otherwise, skip.
- Each speaker gets their own gate — novelty is relative to that speaker's history
- Warmup: first 3 messages stored unconditionally

**System C: Multi-Signal Gated.** Surprise gate + temporal bypass + entity novelty bypass.

```
store = (surprise > threshold) OR has_temporal_markers OR has_novel_entities
```

- Temporal bypass: regex detects dates, time expressions, temporal keywords (`started`, `moved`, `ago`, `since`)
- Entity bypass: spaCy NER tracks seen entities per speaker; new entities bypass the gate
- Every stored memory is a `MemoryRecord` with metadata for plasticity experiments

#### LoCoMo Results (10 conversations, 1540 questions)

| System | Memories | Compression | Overall | Single-hop | Temporal | Multi-hop | Open-domain |
|--------|----------|-------------|---------|------------|----------|-----------|-------------|
| A: Naive | 5882 | 0% | **62.0%** | 54.3% | 60.7% | 44.8% | 67.1% |
| B: Surprise (t=0.2) | 3808 | 35% | 57.9% | 55.3% | 49.8% | 43.8% | 63.5% |
| **C: Multi-signal (t=0.2)** | **3700** | **30%** | **61.6%** | **57.1%** | **59.5%** | **44.8%** | **65.9%** |

- Multi-signal gating loses only 0.4% overall while storing 30% less data
- Temporal questions recover from 49.8% to 59.5% (baseline: 60.7%) thanks to the temporal bypass
- Single-hop improves from 54.3% to 57.1% via entity novelty bypass
- Multi-hop holds at 44.8% — distinctive facts are inherently "surprising" enough to pass any gate

#### LongMemEval Results (500 instances, ~115K tokens each)

| System | IE-User | IE-Asst | IE-Pref | MR | TR | KU | Overall |
|--------|---------|---------|---------|-----|-----|-----|---------|
| A: Naive | 95.7% | 98.2% | 36.7% | 54.9% | 67.7% | 84.6% | **72.4%** |
| B: Surprise (t=0.2) | *running* | | | | | | |
| C: Multi-signal (t=0.2) | *running* | | | | | | |

#### Threshold Sensitivity (LoCoMo)

The surprise threshold controls the compression/quality tradeoff. Fixed-mode nearest-neighbor results:

| Threshold | Compression | Overall | Delta |
|-----------|-------------|---------|-------|
| 0.2 | 35% | 57.9% | -4.1% |
| 0.3 | ~45% | ~56% | ~-6% |
| 0.4 | ~55% | ~53% | ~-9% |
| 0.5 (adaptive) | ~50% | ~55% | ~-7% |

The quality/compression tradeoff is steep for pure surprise-gating. Multi-signal gating breaks this tradeoff.

---

### RQ2: Neuroplasticity

**Can a memory system that reorganizes itself through use outperform static memory?**

Four biologically-inspired mechanisms, each independently toggleable, all building on System C (multi-signal gated):

**Mechanism 1: Retrieval Strengthening & Decay (LTP/LTD)**

When a memory is retrieved for a question answered correctly, its `retrieval_weight` gets boosted (`*= 1.05`). Periodically, all memories decay (`*= 0.99`, floored at 0.1). Over many questions, useful memories float to the top of retrieval results and noise sinks. Scoring: `cosine_similarity * retrieval_weight * (1 - inhibition_weight)`.

**Mechanism 2: Associative Linking (Hebbian Learning)**

Memories co-retrieved for the same question get their co-retrieval count incremented. After 3+ co-retrievals, a strong link forms. During retrieval, one-hop expansion surfaces associated memories that FAISS alone wouldn't return. Bounded to top-10 expansion to limit compute.

**Mechanism 3: Belief Revision through Inhibition**

When a newer memory has high embedding similarity (>0.85) to an older one from a different session, the older one gets inhibited (`inhibition_weight += 0.7`, capped at 0.95). Inhibited memories are suppressed, not deleted — queries containing past-state indicators ("used to", "originally", "before") temporarily reduce inhibition by 70%. Detection uses FAISS kNN (top-10 neighbors) instead of O(n^2) brute-force.

**Mechanism 4: Memory Consolidation**

Periodic offline pass: merge near-duplicates (cosine > 0.92, keep higher-weight survivor), apply extra decay to never-retrieved memories, and generate centroid-based abstract summary memories from clusters of 3+ similar memories. Abstractions get elevated retrieval weight (1.5x) to surface patterns over individual episodes.

#### Ablation Table

| System | LME Overall | LME-KU | LME-TR | LME-MR |
|--------|-------------|--------|--------|--------|
| C: Multi-signal (base) | *running* | | | |
| C + Inhibition only | *running* | | | |
| C + All (Neuroplastic) | *running* | | | |

*Ablation runs in progress. Each mechanism can be independently disabled: `--no_ltp`, `--no_associations`, `--no_inhibition`, `--no_consolidation`.*

---

## Architecture

```
Message Stream
    │
    ▼
┌─────────────────────────┐
│  ENCODING                │
│  • Surprise gate         │
│  • Temporal bypass       │
│  • Entity novelty bypass │
│  • Belief revision       │
│    (inhibit superseded)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MEMORY STORE            │
│  MemoryRecord objects    │
│  with plasticity state:  │
│  • retrieval_weight      │
│  • inhibition_weight     │
│  • associations          │
│  • retrieval_count       │
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────┐  ┌───────────────┐
│ RETRIEVAL│  │ CONSOLIDATION │
│ • FAISS  │  │ (periodic)    │
│ • Weight │  │ • Merge dupes │
│ • Assoc  │  │ • Decay       │
│   expand │  │ • Abstract    │
│ • Inhib  │  │               │
│   aware  │  │               │
└────┬─────┘  └───────────────┘
     │
     ▼
┌─────────────────────────┐
│  FEEDBACK                │
│  • Strengthen if CORRECT │
│  • Update co-retrieval   │
│    graph                 │
└─────────────────────────┘
```

---

## Reproduction

### Setup

```bash
git clone https://github.com/VihAMBR/gated-mem.git
cd gated-mem
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Download LongMemEval dataset
mkdir -p LongMemEval/data
curl -sL -o LongMemEval/data/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
```

**Requirements:** Python 3.10+, OpenAI API key (for GPT-4o-mini). The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run.

### Quick Test

```bash
python test_quick.py  # 5 questions, ~30 seconds
```

### RQ1: Encoding Experiments

```bash
# LoCoMo
python run_experiment.py naive_baseline                                          # System A
python run_experiment.py surprise_gated --threshold 0.2                          # System B
python run_experiment.py enhanced_gated --threshold 0.2                          # System C
python run_experiment.py threshold_sweep --quick                                 # Sensitivity analysis

# LongMemEval
python run_experiment.py lme --mode naive                                        # System A
python run_experiment.py lme --mode enhanced                                     # System C
```

### RQ2: Plasticity Experiments

```bash
# Full neuroplastic (all 4 mechanisms)
python run_experiment.py lme --mode neuroplastic

# Ablations (disable one mechanism at a time)
python run_experiment.py lme --mode neuroplastic --no_ltp
python run_experiment.py lme --mode neuroplastic --no_associations
python run_experiment.py lme --mode neuroplastic --no_inhibition
python run_experiment.py lme --mode neuroplastic --no_consolidation

# Inhibition only (no LTP/associations/consolidation)
python run_experiment.py lme --mode inhibition
```

### Utilities

```bash
python run_experiment.py compare                                                 # Cross-experiment comparison table

# Run phases individually
python run_naive_baseline.py --output results/my_run.json                        # Generate only
python evals.py --input_file results/my_run.json --output_file results/evals.json # Evaluate only
python generate_scores.py --input_path results/evals.json                        # Score only
```

All runners support **resume** — if interrupted, re-run the same command and it skips completed work.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_k` | 30 | Memories retrieved per query |
| `--threshold` | 0.2-0.3 | Surprise cutoff (fixed: absolute, adaptive: percentile) |
| `--mode` | `fixed` | `fixed` or `adaptive` threshold mode |
| `--metric` | `nearest_neighbor` | `nearest_neighbor` or `centroid` surprise metric |
| `--warmup` | 3 | Always store first N messages unconditionally |
| `--max_convs` | None | Limit LoCoMo conversations (for testing) |
| `--max_instances` | None | Limit LongMemEval instances (for testing) |
| `--eval_workers` | 2-4 | Parallel threads for LLM judge |
| `--answer_workers` | 4 | Parallel threads for answer generation |

### Rate Limits

OpenAI API rate limits are the main bottleneck. The code has built-in exponential backoff (up to 60s, 8 retries). Use `--max_convs 2` or `--max_instances 10` for quick tests.

---

## Project Structure

```
gated-mem/
├── run_experiment.py            # Unified CLI for all experiments
├── run_naive_baseline.py        # System A: naive baseline (LoCoMo)
├── run_gated_baseline.py        # System B: surprise-gated (LoCoMo)
├── run_enhanced_gated.py        # System C: multi-signal gated (LoCoMo)
├── run_longmemeval.py           # Systems A/B/C + neuroplastic (LongMemEval)
├── eval_longmemeval.py          # LongMemEval judge (official per-type prompts)
├── surprise_gated_encoder.py    # Surprise gate: SurpriseGatedEncoder
├── enhanced_gated_encoder.py    # Multi-signal: TemporalDetector, EntityTracker, MemoryRecord
├── neuroplastic_memory.py       # Plasticity: LTP, Associations, Inhibition, Consolidation
├── evals.py                     # LoCoMo evaluation (BLEU + F1 + LLM judge)
├── generate_scores.py           # Per-category score aggregation
├── analyze_results.py           # Cross-experiment comparison table
├── test_quick.py                # 5-question smoke test
├── prompts.py                   # Answer prompt variants
├── metrics/
│   ├── llm_judge.py             # GPT-4o-mini judge
│   └── utils.py                 # BLEU, F1 calculations
├── dataset/
│   └── locomo10.json            # LoCoMo benchmark (10 conversations)
├── LongMemEval/                 # LongMemEval repo + data (gitignored)
├── results/                     # All benchmark results (committed)
├── requirements.txt
├── .env.example
└── .gitignore
```
