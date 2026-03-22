"""
Evaluate LongMemEval results using their official judge prompts.

Each question type has a tailored prompt (knowledge-update is lenient about
mentioning old info alongside the correct update, temporal-reasoning allows
off-by-one errors, etc.).

Usage:
    python eval_longmemeval.py --results results/lme_naive.json \
                               --reference LongMemEval/data/longmemeval_s_cleaned.json
"""

import os
import json
import time
import argparse
import concurrent.futures
from collections import Counter

from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_judge_prompt(qtype: str, question: str, answer: str, response: str) -> str:
    """LongMemEval's official judge prompts, adapted from their evaluate_qa.py."""
    abstention = "_abs" in qtype if isinstance(qtype, str) else False

    if abstention:
        return (
            "I will give you an unanswerable question, an explanation, and a response "
            "from a model. Please answer yes if the model correctly identifies the question "
            "as unanswerable. The model could say that the information is incomplete, or some "
            "other information is given but the asked information is not.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )

    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if qtype == "temporal-reasoning":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days. If the question asks for "
            "the number of days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if qtype == "knowledge-update":
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is "
            "the required answer.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    if qtype == "single-session-preference":
        return (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal information "
            "correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    return (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    )


def judge_one(client, model, qtype, question, answer, response, max_retries=8):
    prompt = get_judge_prompt(qtype, question, answer, response)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip().lower()
            return "yes" in text
        except Exception as e:
            wait = min(2 ** attempt, 60)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                print(f"  Judge failed after {max_retries} retries: {e}")
                return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate LongMemEval results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.results.replace(".json", "_scored.json")

    with open(args.results) as f:
        results = json.load(f)

    # Skip already-scored items
    to_score = [r for r in results if "correct" not in r]
    already_scored = len(results) - len(to_score)
    if already_scored > 0:
        print(f"Resuming — {already_scored} already scored, {len(to_score)} remaining")

    if not to_score:
        print("All items already scored.")
    else:
        client = OpenAI()

        def _score(item):
            correct = judge_one(
                client, args.judge_model,
                item["question_type"], item["question"],
                item["answer"], item["hypothesis"],
            )
            item["correct"] = correct
            return item

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(_score, item): item for item in to_score}
            pbar = tqdm(total=len(to_score), desc="Judging")
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)
            pbar.close()

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Print scores
    print(f"\n{'='*60}")
    print(f"  LongMemEval Scores")
    print(f"{'='*60}")

    type_scores = {}
    for r in results:
        qtype = r["question_type"]
        if qtype not in type_scores:
            type_scores[qtype] = []
        type_scores[qtype].append(1 if r.get("correct", False) else 0)

    total_correct = sum(r.get("correct", False) for r in results)
    total = len(results)

    # Abbreviated type names for the table
    abbrev = {
        "single-session-user": "IE-User",
        "single-session-assistant": "IE-Asst",
        "single-session-preference": "IE-Pref",
        "multi-session": "MR",
        "temporal-reasoning": "TR",
        "knowledge-update": "KU",
    }

    for qtype in sorted(type_scores.keys()):
        scores = type_scores[qtype]
        name = abbrev.get(qtype, qtype)
        acc = sum(scores) / len(scores) if scores else 0
        print(f"  {name:>10s}: {acc:.1%} ({sum(scores)}/{len(scores)})")

    print(f"  {'Overall':>10s}: {total_correct/total:.1%} ({total_correct}/{total})")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
