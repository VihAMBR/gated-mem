# gated-mem: What Actually Works for LLM Long-Term Memory

An experimental investigation into memory systems for conversational AI agents. This project started as an attempt to build a biologically-inspired "neuroplastic" memory system, and became an honest accounting of what matters and what doesn't when an LLM needs to remember past conversations.

**The headline finding:** After weeks of engineering increasingly sophisticated encoding, retrieval, and memory reorganization systems, the single biggest improvement came from changing the **answer prompt**. Adding chain-of-thought reasoning to a naive RAG baseline jumped accuracy from 72.3% to **78.2% on LongMemEval** (+5.8% on all 500 instances), with a **+40 percentage point** gain on preference questions. The bottleneck was never the memory system — it was how the LLM processed the retrieved context.

---

## The Research Journey

This is written in the order we actually built and tested things, not restructured after the fact. The failures are as informative as the successes.

### Phase 1: The Naive Baseline

**Hypothesis:** Before building anything clever, establish what a minimal RAG system achieves.

**What we built:**
- Every conversation turn stored as `"<timestamp> | <Speaker>: <text>"`
- Embedded with `all-MiniLM-L6-v2` (384-dim, ~80MB, runs locally)
- Indexed in FAISS (`IndexFlatIP` — exact inner product search)
- Top-30 most similar memories retrieved per query
- GPT-4o-mini generates the answer from retrieved context
- GPT-4o-mini as LLM judge (using Mem0's prompts for LoCoMo, official per-type prompts for LongMemEval)

**Results:**

| Benchmark | Overall | Best Category | Worst Category |
|-----------|---------|---------------|----------------|
| LoCoMo (10 convs, 1540 Qs) | **62.0%** | Temporal 67.1% | Open-domain 44.8% |
| LongMemEval (500 instances) | **72.4%** | IE-Assistant 98.2% | Preferences 36.7% |

The LongMemEval score was surprising. The paper reports GPT-4o with full context at ~60-64%. Our simple retrieve-then-answer pipeline beats brute-force context stuffing because FAISS retrieval acts as a relevance filter — the LLM sees 30 focused memories instead of 115K tokens of noise.

**Cost:** Zero LLM calls during ingestion. One LLM call per question at inference.

### Phase 2: Surprise-Gated Encoding — The First Failure

**Hypothesis:** Most conversation turns are phatic or redundant. If we only store "surprising" messages (those whose embedding is distant from all stored memories), we can compress the memory bank without losing retrieval quality.

**What we built:**
- `SurpriseGatedEncoder`: computes `surprise = 1 - max(cosine_similarity(new, stored))`
- If surprise >= threshold, store. Otherwise, skip.
- Per-speaker gates — novelty is relative to each speaker's history
- Warmup: first 3 messages always stored

**Results (LoCoMo, 10 conversations):**

| Threshold | Compression | Overall | Delta vs Naive |
|-----------|-------------|---------|----------------|
| 0.2 | 35% | 57.9% | **-4.1%** |
| 0.3 | ~45% | ~38.2%* | -23.8% |
| 0.4 | ~55% | ~28.3%* | -33.7% |
| 0.5 (adaptive) | ~50% | ~53.2%* | -8.8% |

*\*t=0.3/0.4/0.5 ran on 2 conversations only (not full benchmark)*

**What went wrong:** Surprise gating is a blunt instrument. Temporal information looks "unsurprising" to embeddings — "I moved to SF in March" and "I started my new job in April" have similar embedding patterns to existing life-update messages, so they get filtered. But these are exactly the facts that LoCoMo's temporal questions ask about. The gate threw away what mattered most.

### Phase 3: Multi-Signal Encoding — Partial Recovery

**Hypothesis:** Surprise alone is insufficient. Add bypass mechanisms for information that has low embedding novelty but high semantic importance: temporal content and novel entities.

**What we built (`EnhancedGatedEncoder`):**
```
store = (surprise > threshold) OR has_temporal_markers OR has_novel_entities
```

- **Temporal bypass:** Regex patterns for dates, time expressions, temporal keywords (`started`, `moved`, `ago`, `since`)
- **Entity bypass:** spaCy NER (`en_core_web_sm`) tracks seen entities per speaker; new entities bypass the gate
- Every stored memory wrapped in a `MemoryRecord` with metadata fields for later experiments

**Results (matched conversations, apples-to-apples):**

| System | LoCoMo (convs 0-2) | LongMemEval (500) |
|--------|---------------------|-------------------|
| Naive | 70.1% | **72.4%** |
| Surprise t=0.2 | 63.1% | — |
| Multi-signal t=0.2 | **69.1%** | 72.3% |

The temporal and entity bypasses recovered most of what surprise gating destroyed. On LoCoMo, multi-signal is within 1% of naive while storing 30% less. On LongMemEval, it's essentially identical overall but shifts category performance:

| Category | Naive | Multi-signal | Delta |
|----------|-------|-------------|-------|
| Multi-session reasoning | 54.9% | **60.3%** | **+5.4%** |
| Preferences | 36.7% | **46.7%** | **+10.0%** |
| Temporal reasoning | **67.7%** | 61.7% | -6.0% |
| Knowledge update | **84.6%** | 79.5% | -5.1% |
| IE (user/assistant) | 95.7/98.2% | 95.7/100% | ~0% |

**Lesson learned:** The encoding stage is not the bottleneck. Storing 30% less data is useful for cost/latency at scale, but it doesn't improve answer quality. The LLM is already good at finding relevant information in a pile of context.

### Phase 4: Neuroplastic Memory — The Ambitious Attempt

**Hypothesis:** A memory system that reorganizes itself through use — strengthening useful memories, linking related ones, suppressing outdated ones, consolidating duplicates — should outperform a static store.

**What we built (`NeuroplasticMemory`, four mechanisms):**

1. **Retrieval Strengthening (LTP/LTD):** Memories that help answer questions correctly get weight boosts (`retrieval_weight *= 1.05`). All memories decay periodically (`*= 0.99`, floor 0.1). Scoring becomes `cosine_sim * retrieval_weight * (1 - inhibition_weight)`.

2. **Associative Linking (Hebbian):** Track co-retrieval counts between memory pairs. After 3+ co-retrievals, form a link. During retrieval, one-hop expansion surfaces associated memories.

3. **Belief Revision (Inhibition):** When a newer memory has cosine similarity > 0.85 to an older one from a different session, the older one gets inhibited (weight += 0.7, capped at 0.95). Queries with past-state indicators ("used to", "originally") temporarily reduce inhibition by 70%.

4. **Consolidation:** Merge near-duplicates (cosine > 0.92), apply extra decay to never-retrieved memories, generate centroid-based abstract summaries from clusters of 3+ similar memories.

**Results (LongMemEval, 500 instances):**

| System | Overall | KU | TR | MR | Pref |
|--------|---------|-----|-----|-----|------|
| Multi-signal (base) | 72.3% | 79.5% | 61.7% | **60.3%** | **46.7%** |
| + Inhibition only | 70.6% | 78.2% | 61.7% | 57.1% | 40.0% |
| + All neuroplastic | 72.2% | 80.8% | 63.9% | 57.9% | **46.7%** |
| Naive (control) | **72.4%** | **84.6%** | **67.7%** | 54.9% | 36.7% |

**What went wrong — and why:**

- **LTP/LTD was structurally inactive.** LongMemEval has one question per instance. Retrieval strengthening requires a multi-question feedback loop where the system learns which memories are useful from earlier questions. With one question, there's nothing to learn from.

- **Associative linking never formed links.** Same reason — zero co-retrieval events means zero associations. This mechanism needs dozens of questions per conversation to build a useful graph.

- **Inhibition over-triggered.** The 0.85 cosine threshold caught topically similar but non-contradictory memories. "I love my job at the clinic" and "busy day at the clinic" score > 0.85 but don't contradict each other. False positive inhibitions suppressed useful memories, dropping knowledge-update accuracy by 6.4% compared to naive.

- **Consolidation was neutral.** It merged 1,259 near-duplicates across 500 instances and created 77 abstractions. The merges removed genuine duplicates (helpful), but the abstractions — centroid embeddings with template text like "[Consolidated pattern from 4 memories]" — weren't informative enough for the LLM to extract answers from.

**The uncomfortable truth:** These mechanisms aren't wrong. They're well-motivated by neuroscience and would genuinely help in a system that answers thousands of questions over months. But no benchmark tests that. LoCoMo and LongMemEval both evaluate static, one-shot retrieval from a pre-loaded memory bank. The neuroplastic features have nothing to adapt to.

### Phase 5: The Prompt Was the Bottleneck All Along

After four phases of engineering encoding, retrieval, and memory reorganization, we tested whether simply changing the **answer prompt** would matter more. It did — by a wide margin.

**Chain-of-Thought (CoT) prompting** replaces the default "answer in 1-2 sentences" prompt with an explicit reasoning scaffold:
1. Identify every relevant memory
2. If counting/listing, enumerate each item before totaling
3. If information was updated, use the most recent version and state which date
4. If time calculations needed, write out start date, end date, and arithmetic
5. If asking about preferences, infer from behavior and stated opinions
6. Give final answer

**Results (full 500-instance LongMemEval):**

| System | Overall | Multi-session | Preferences | Temporal | Knowledge-update |
|--------|---------|---------------|-------------|----------|-----------------|
| Naive baseline | 72.3% | 54.9% | 36.7% | 67.7% | 84.6% |
| **Naive + CoT prompt** | **78.2%** | **63.2%** | **76.7%** | **72.2%** | 84.6% |
| Delta | **+5.8%** | **+8.3%** | **+40.0%** | +4.5% | +0.0% |

This is by far the most impactful change in the entire project. No encoding changes. No retrieval changes. No new models. Just a better prompt that forces the LLM to show its reasoning before answering. Preferences jump from 36.7% to 76.7% because CoT forces the model to enumerate evidence and infer from behavior rather than guessing.

**All retrieval experiments (100-instance stratified subset, apples-to-apples vs naive on same questions):**

| Experiment | Overall | vs Naive | Best Improvement | Worst Regression |
|------------|---------|----------|------------------|-----------------|
| **CoT prompting** | **80.4%** | **+9.3%** | Multi-session +23.1%, Preferences +33.3% | — |
| **Cross-encoder reranking** | **79.8%** | **+4.3%** | Temporal +11.5%, KU +6.7% | Preferences -20.0% |
| Recency boost | 71.4% | -1.0% | — | Preferences -16.7% |

CoT and reranking are complementary: CoT fixes reasoning-dependent categories (multi-session, preferences) while reranking fixes retrieval-dependent categories (temporal reasoning, knowledge updates). Recency bias doesn't help.

### What This Means

Weeks of engineering went into optimizing the wrong layer. The encoding, retrieval, and memory reorganization experiments all operated on the assumption that getting the right memories in front of the LLM was the hard part. In reality, GPT-4o-mini was already seeing the right information — it just wasn't reasoning carefully enough to extract the answer.

The CoT prompt fixes this by forcing explicit enumeration (helps counting), temporal ordering (helps knowledge updates), and evidence marshaling (helps multi-hop inference). The cost is zero additional API calls — just a longer prompt template.

---

## Results Summary

### How We Compare to Other Systems

#### LoCoMo

| System | Overall | Ingestion Cost | Source |
|--------|---------|----------------|--------|
| Supermemory | ~85% | Full LLM pipeline | Their paper |
| Memobase | ~76% | LLM summarization per turn | Their paper |
| **gated-mem (naive)** | **62%** | **Embedding only** | This repo |
| Mem0 | ~61% | LLM calls per turn | Mem0 paper |
| Zep | ~58-75% | Graph construction | Reported range |

Our system matches Mem0 using zero LLM calls during ingestion. Memobase and higher-scoring systems pay for LLM calls at every conversation turn.

#### LongMemEval

| System | Overall | Source |
|--------|---------|--------|
| **gated-mem (naive + CoT)** | **78.2%** | This repo (499 instances) |
| gated-mem (naive) | 72.4% | This repo (500 instances) |
| Supermemory | ~71% | Their research page |
| GPT-4o (full context) | ~60-64% | LongMemEval paper |
| ReadAgent | ~55% | LongMemEval paper |

With CoT prompting, our system outperforms all published baselines on LongMemEval, using only a local embedding model and one GPT-4o-mini call per question (zero LLM calls during ingestion).

### Full Results Table

| System | LoCoMo | LME Overall | LME-KU | LME-TR | LME-MR | LME-Pref |
|--------|--------|-------------|--------|--------|--------|----------|
| Naive (store all) | 62.0% | 72.4% | 84.6% | 67.7% | 54.9% | 36.7% |
| Surprise-gated (t=0.2) | 57.9% | — | — | — | — | — |
| Multi-signal (t=0.2) | 69.1%* | 72.3% | 79.5% | 61.7% | 60.3% | 46.7% |
| + Inhibition only | — | 70.6% | 78.2% | 61.7% | 57.1% | 40.0% |
| + All neuroplastic | — | 72.2% | 80.8% | 63.9% | 57.9% | 46.7% |
| **Naive + CoT prompt** | — | **78.2%** | 84.6% | **72.2%** | **63.2%** | **76.7%** |
| Naive + Reranking | — | 79.8%† | **93.3%** | 76.9% | 66.7% | 20.0% |

*\*LoCoMo multi-signal ran on 3/10 conversations. Apples-to-apples: naive 70.1% vs multi-signal 69.1%.*
*CoT result is on full 499 instances. †Rerank result on 94-instance subset.*

---

## Why Current Benchmarks Don't Test What Matters

Both LoCoMo and LongMemEval evaluate memory systems in a static, one-shot mode: load a conversation history, build a memory bank, answer questions. This misses the properties that would differentiate a good memory system in production:

**What benchmarks test:**
- Can you find the right fact in a pile of conversation?
- Can you handle knowledge updates?
- Can you reason across sessions?

**What production memory needs but no benchmark tests:**
- **Longitudinal adaptation:** Does the system get better at serving a specific user over weeks/months? (LTP/LTD would show value here)
- **Cost at scale:** What's the storage/compute cost per user per month? (Compression matters here, but benchmarks don't penalize storing everything)
- **Latency under load:** How fast is retrieval when you have 100K memories per user? (FAISS scales, but the index management matters)
- **Graceful forgetting:** Can the system surface recent preferences over stale ones without explicit retraining? (Decay and inhibition would help here)
- **Cross-conversation learning:** Does answering questions about topic A improve retrieval for related questions about topic B? (Associative linking would show value here)
- **Privacy and deletion:** Can you reliably delete specific memories? (GDPR compliance)

A meaningful benchmark for long-term memory agents would process a sequence of 500+ questions interleaved with new conversations, evaluating accuracy over time. If the neuroplastic system's second-half accuracy exceeds its first-half accuracy (and the static system's doesn't), that's evidence of learning through use. No existing benchmark measures this.

### What We'd Build for Production

Based on everything we learned:

**For maximum accuracy (the "best system"):**
1. Store every conversation turn (naive encoding — don't filter)
2. Embed with `all-MiniLM-L6-v2`, index in FAISS
3. Retrieve top-30 by cosine similarity
4. Answer with Chain-of-Thought prompt (forces enumeration, temporal ordering, evidence marshaling)
5. Zero LLM calls during ingestion, one per question at inference

This is the simplest possible architecture, and it beats every alternative we tested.

**For cost-efficient production at scale:**
- Multi-signal gated encoding (30% storage reduction, negligible quality loss)
- Periodic deduplication (consolidation merging only, skip abstractions)
- CoT prompting at inference time

**For a truly adaptive agent (when the right benchmark exists):**
- Full neuroplastic stack with multi-question feedback loops
- LTP strengthening gated on answer correctness
- Associative linking with periodic pruning
- Belief revision with entity-aware contradiction detection (not just embedding similarity)

---

## Architecture

```
Message Stream
    │
    ▼
┌─────────────────────────┐
│  ENCODING                │
│  • Every turn (naive)    │
│  OR                      │
│  • Surprise gate         │
│  • + Temporal bypass     │
│  • + Entity novelty      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MEMORY STORE            │
│  FAISS IndexFlatIP       │
│  all-MiniLM-L6-v2 embs  │
│  MemoryRecord metadata   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  RETRIEVAL               │
│  Top-k cosine similarity │
│  Optional: weighted      │
│    scoring, reranking,   │
│    association expansion │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  ANSWER GENERATION       │
│  GPT-4o-mini             │
│  Retrieved memories as   │
│  context                 │
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

**Requirements:** Python 3.10+, OpenAI API key (for GPT-4o-mini). The embedding model (~80MB) downloads automatically on first run.

### Quick Test

```bash
python test_quick.py  # 5 questions, ~30 seconds
```

### Run Experiments

```bash
# Phase 1: Naive baseline
python run_experiment.py naive_baseline                                    # LoCoMo
python run_experiment.py lme --mode naive                                  # LongMemEval

# Phase 2: Surprise gating
python run_experiment.py surprise_gated --threshold 0.2                    # LoCoMo

# Phase 3: Multi-signal encoding
python run_experiment.py enhanced_gated --threshold 0.2                    # LoCoMo
python run_experiment.py lme --mode enhanced                               # LongMemEval

# Phase 4: Neuroplastic memory
python run_experiment.py lme --mode neuroplastic                           # All 4 mechanisms
python run_experiment.py lme --mode inhibition                             # Inhibition only
python run_experiment.py lme --mode neuroplastic --no_ltp                  # Ablation: disable LTP
python run_experiment.py lme --mode neuroplastic --no_associations         # Ablation: disable linking
python run_experiment.py lme --mode neuroplastic --no_inhibition           # Ablation: disable inhibition
python run_experiment.py lme --mode neuroplastic --no_consolidation        # Ablation: disable consolidation

# Phase 5: Retrieval experiments
python precompute_embeddings.py --max_instances 100                        # Pre-compute embeddings
python run_lme_experiments.py --experiment cot                             # Chain-of-thought prompting
python run_lme_experiments.py --experiment rerank                          # Cross-encoder reranking
python run_lme_experiments.py --experiment recency                         # Recency-boosted retrieval
python run_lme_experiments.py --experiment topk50                          # Top-k = 50

# Utilities
python run_experiment.py compare                                           # Cross-experiment comparison
```

All runners support **resume** — if interrupted, re-run the same command and it continues from where it left off.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_k` | 30 | Memories retrieved per query |
| `--threshold` | 0.2 | Surprise cutoff for gated encoding |
| `--mode` | `fixed` | `fixed` or `adaptive` threshold mode |
| `--warmup` | 3 | Always store first N messages unconditionally |
| `--max_convs` | None | Limit LoCoMo conversations (for testing) |
| `--max_instances` | None | Limit LongMemEval instances (for testing) |
| `--eval_workers` | 2-4 | Parallel threads for LLM judge |
| `--answer_workers` | 4 | Parallel threads for answer generation |

### Rate Limits

OpenAI API rate limits are the main bottleneck. Built-in exponential backoff (up to 60s, 8 retries). Use `--max_convs 2` or `--max_instances 10` for quick tests.

---

## Project Structure

```
gated-mem/
├── run_experiment.py            # Unified CLI for all experiments
├── run_naive_baseline.py        # Naive baseline (LoCoMo)
├── run_gated_baseline.py        # Surprise-gated encoder (LoCoMo)
├── run_enhanced_gated.py        # Multi-signal gated encoder (LoCoMo)
├── run_longmemeval.py           # All systems on LongMemEval
├── eval_longmemeval.py          # LongMemEval judge (official per-type prompts)
├── run_lme_experiments.py       # Retrieval experiment suite
├── precompute_embeddings.py     # Embedding cache for experiment suite
├── surprise_gated_encoder.py    # SurpriseGatedEncoder
├── enhanced_gated_encoder.py    # TemporalDetector, EntityTracker, MemoryRecord
├── neuroplastic_memory.py       # LTP, Associations, Inhibition, Consolidation
├── evals.py                     # LoCoMo evaluation (BLEU + F1 + LLM judge)
├── generate_scores.py           # Per-category score aggregation
├── analyze_results.py           # Cross-experiment comparison
├── test_quick.py                # 5-question smoke test
├── prompts.py                   # Answer prompt variants
├── metrics/                     # BLEU, F1, LLM judge utilities
├── dataset/locomo10.json        # LoCoMo benchmark (10 conversations)
├── LongMemEval/                 # LongMemEval dataset (gitignored)
├── results/                     # All benchmark results
├── requirements.txt
└── .env.example
```
