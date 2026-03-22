#!/bin/bash
set -e

echo "============================================"
echo "  Naive Baseline — Full LoCoMo Benchmark"
echo "============================================"
echo ""

# Step 1: Run the baseline (generates answers)
echo "[Step 1/3] Running naive baseline (embed → retrieve → answer)..."
echo "  This will process 1,540 questions across 10 conversations."
echo "  Estimated time: 30-60 minutes (depends on OpenAI API speed)."
echo ""
python run_naive_baseline.py \
    --dataset dataset/locomo10.json \
    --output results/naive_baseline_results.json \
    --top_k 30

echo ""
echo "[Step 2/3] Evaluating answers (BLEU + F1 + LLM Judge)..."
echo "  This calls GPT-4o-mini for each question to judge correctness."
echo ""
python evals.py \
    --input_file results/naive_baseline_results.json \
    --output_file results/naive_baseline_evals.json

echo ""
echo "[Step 3/3] Generating final scores..."
echo ""
python generate_scores.py \
    --input_path results/naive_baseline_evals.json

echo ""
echo "============================================"
echo "  Done! Results saved to results/"
echo "============================================"
