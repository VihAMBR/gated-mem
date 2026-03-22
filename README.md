# Neuroplastic Memory for Long-Term Conversational Agents

This project builds a **biologically-inspired memory system** for LLM agents. Starting from a naive "store everything" baseline, we progressively add neuroplastic mechanisms: surprise-gating, temporal/entity bypass, belief revision through inhibition, memory consolidation, and retrieval strengthening.

We evaluate on two benchmarks:
- **[LoCoMo](https://github.com/memodb-io/memobase)** — 10 multi-session conversations, 1540 questions across single-hop, temporal, multi-hop, and open-domain categories
- **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** (ICLR 2025) — 500 evaluation instances testing information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention

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
# Download spaCy model (needed for enhanced/neuroplastic experiments)
python -m spacy download en_core_web_sm

# Download LongMemEval dataset (needed for LME experiments)
mkdir -p LongMemEval/data
curl -sL -o LongMemEval/data/longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
```

**Requirements:** Python 3.10+, an OpenAI API key (for GPT-4o-mini).

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run (~80MB). The spaCy model (`en_core_web_sm`) must be downloaded manually (see above).

## Quick Start

```bash
# Verify the pipeline works (5 questions, ~30 seconds)
python test_quick.py

# LoCoMo experiments
python run_experiment.py naive_baseline --max_convs 2
python run_experiment.py surprise_gated --threshold 0.3 --max_convs 2
python run_experiment.py enhanced_gated --threshold 0.2 --max_convs 2

# LongMemEval experiments
python run_experiment.py lme --mode naive --max_instances 10
python run_experiment.py lme --mode neuroplastic --max_instances 10

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

### Experiment 6: LongMemEval Benchmark

**Question:** How does our memory system perform on a different benchmark with different question types — especially knowledge updates and temporal reasoning?

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) is a 500-instance benchmark where each instance has its own chat history (~40 sessions, ~500 turns, ~115K tokens). Unlike LoCoMo (two friends chatting), LongMemEval uses user-assistant format. It tests six categories:

- **IE-User / IE-Asst / IE-Pref**: Information extraction from user/assistant/preference messages
- **MR**: Multi-session reasoning (connecting facts across sessions)
- **TR**: Temporal reasoning (date arithmetic, sequence questions)
- **KU**: Knowledge updates (has the user's information changed?)

The adapter (`run_longmemeval.py`) feeds LongMemEval's format through our existing pipeline with speaker-role separation.

**Run:**

```bash
# Naive baseline (500 instances, ~4 hours)
python run_experiment.py lme --mode naive

# Multi-signal gated (surprise + temporal + entity bypass)
python run_experiment.py lme --mode enhanced

# With belief revision inhibition
python run_experiment.py lme --mode inhibition

# Full neuroplastic (all 4 mechanisms)
python run_experiment.py lme --mode neuroplastic

# Quick test (10 instances, ~5 min)
python run_experiment.py lme --mode naive --max_instances 10
```

**Key files:**
- `run_longmemeval.py` — LongMemEval adapter and runner
- `eval_longmemeval.py` — LongMemEval judge (uses their official per-type prompts)

---

### Experiment 7: Neuroplastic Memory

**Question:** Can a memory system that reorganizes itself through use — like a brain — outperform static memory?

Four biologically-inspired plasticity mechanisms, each with its own `--enable/--no` flag:

**Mechanism 1: Retrieval Strengthening & Decay (LTP/LTD)**
- Memories retrieved for correctly answered questions get a weight boost (`retrieval_weight *= 1.05`)
- All memories decay gently over time (`retrieval_weight *= 0.99`), floored at 0.1
- Emergent behavior: frequently useful memories float to the top; noise sinks

**Mechanism 2: Associative Linking (Hebbian Learning)**
- Memories co-retrieved for the same question get their co-retrieval count incremented
- After 3+ co-retrievals, a strong link forms
- During retrieval, one-hop expansion surfaces linked memories FAISS wouldn't return
- Emergent behavior: multi-hop reasoning improves as the system discovers which facts go together

**Mechanism 3: Belief Revision through Inhibition**
- When a newer memory has high embedding similarity (>0.85) to an older one from a different session, the older one gets inhibited (`inhibition_weight = 0.7`)
- Inhibited memories are suppressed but not deleted — queries about past states ("where did you used to live?") temporarily reduce inhibition
- Directly tests LongMemEval's knowledge-update category

**Mechanism 4: Memory Consolidation (Sleep-time Reorganization)**
- Merge near-duplicate memories (cosine > 0.92): keep the stronger one, inhibit the weaker
- Extra decay for never-retrieved memories
- Generate centroid-based abstract "summary" memories from clusters of 3+ similar memories
- Emergent behavior: episodic memories → semantic patterns

**Run:**

```bash
# All mechanisms enabled (default)
python run_experiment.py lme --mode neuroplastic

# Ablations
python run_experiment.py lme --mode neuroplastic --no_ltp
python run_experiment.py lme --mode neuroplastic --no_associations
python run_experiment.py lme --mode neuroplastic --no_inhibition
python run_experiment.py lme --mode neuroplastic --no_consolidation
```

**Key files:**
- `neuroplastic_memory.py` — `RetrievalPlasticity`, `AssociationGraph`, `BeliefRevisionDetector`, `ConsolidationEngine`, `NeuroplasticMemory`

**Architecture:**

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

## Results

### LoCoMo Benchmark (10 conversations, 1540 questions)

| Config | Memories | Compression | Overall | Single-hop | Temporal | Multi-hop | Open-domain | vs Baseline |
|--------|----------|------------|---------|------------|----------|-----------|-------------|-------------|
| Naive Baseline | 5882/5882 | 0% | **62.0%** | 54.3% | 60.7% | 44.8% | 67.1% | — |
| Surprise t=0.2 | 3808/5882 | 35.3% | 57.9% | 55.3% | 49.8% | 43.8% | 63.5% | -4.1% |
| **Enhanced t=0.2** | **3700/5314** | **30.4%** | **61.6%** | **57.1%** | **59.5%** | **44.8%** | **65.9%** | **-0.4%** |

### LongMemEval Benchmark (LongMemEval_S, 500 instances, ~115K tokens each)

| Config | IE-User | IE-Asst | IE-Pref | MR | TR | KU | Overall |
|--------|---------|---------|---------|-----|-----|-----|---------|
| Naive (store all) | 95.7% | 98.2% | 36.7% | 54.9% | 67.7% | 84.6% | **72.4%** |
| Enhanced gated | *running* | | | | | | |
| Inhibition-only | *running* | | | | | | |
| **Neuroplastic** | *running* | | | | | | |

*Enhanced, inhibition, and neuroplastic runs are in progress. Results will be updated when complete.*

### Key Findings

**LoCoMo:**
- **Enhanced gating matches the baseline while storing 30% less data.** At 61.6% vs 62.0%, the difference is 0.4% — within noise on 1540 questions.
- **Temporal questions nearly fully recovered** (49.8% → 59.5%, vs baseline 60.7%). The temporal bypass reclaims what pure surprise-gating destroys.
- **Single-hop actually improves** (54.3% → 57.1%). Entity novelty bypass captures factual mentions that cosine similarity sometimes misses.

**LongMemEval:**
- **Naive baseline achieves 72.4% overall** on LongMemEval_S. Strong on information extraction (95-98% for user/assistant) but weaker on multi-session reasoning (54.9%) and preferences (36.7%).
- **Knowledge updates score 84.6%** — already strong without inhibition. The hypothesis is that belief revision via inhibition will push this higher.
- **Temporal reasoning at 67.7%** — the enhanced gated encoder's temporal bypass should help here.

---

## Project Structure

```
gated-mem/
├── run_experiment.py            # Unified entry point for all experiments
├── run_naive_baseline.py        # Experiment 1: naive baseline (LoCoMo)
├── run_gated_baseline.py        # Experiment 2: surprise-gated (LoCoMo)
├── run_enhanced_gated.py        # Experiment 3: enhanced gated (LoCoMo)
├── run_longmemeval.py           # Experiment 6: LongMemEval adapter & runner
├── eval_longmemeval.py          # LongMemEval judge (official prompts per type)
├── surprise_gated_encoder.py    # Core: SurpriseGatedEncoder class
├── enhanced_gated_encoder.py    # Enhanced: temporal/entity bypass + MemoryRecord
├── neuroplastic_memory.py       # Neuroplastic: LTP/LTD, associations, inhibition, consolidation
├── evals.py                     # LoCoMo evaluation pipeline (BLEU + F1 + LLM judge)
├── generate_scores.py           # Per-category score aggregation
├── analyze_results.py           # Cross-experiment comparison table
├── test_quick.py                # 5-question smoke test
├── prompts.py                   # Prompt variants (Mem0, graph, Zep)
├── metrics/
│   ├── llm_judge.py             # GPT-4o-mini judge (CORRECT/WRONG)
│   └── utils.py                 # BLEU, F1 calculations
├── dataset/
│   └── locomo10.json            # LoCoMo benchmark (10 conversations)
├── LongMemEval/                 # LongMemEval repo + data (gitignored, see Setup)
├── results/                     # Benchmark results (committed)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Pipeline Architecture

```
LoCoMo (dataset/locomo10.json)          LongMemEval (LongMemEval/data/*.json)
         │                                        │
         ├──► run_naive_baseline.py                ├──► run_longmemeval.py --mode naive
         ├──► run_gated_baseline.py                ├──► run_longmemeval.py --mode enhanced
         ├──► run_enhanced_gated.py                ├──► run_longmemeval.py --mode inhibition
         │                                         ├──► run_longmemeval.py --mode neuroplastic
         │                                         │
         │    Shared modules:                      │
         │    ├── surprise_gated_encoder.py         │
         │    ├── enhanced_gated_encoder.py         │
         │    └── neuroplastic_memory.py            │
         │                                         │
         ▼                                         ▼
    results/*.json                          results/lme_*.json
         │                                         │
         ▼                                         ▼
    evals.py (BLEU + F1 + LLM Judge)       eval_longmemeval.py (official prompts)
         │                                         │
         ▼                                         ▼
    results/*_evals.json                    results/lme_*_scored.json
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
