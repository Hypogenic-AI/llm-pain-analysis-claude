"""
LLM-as-judge evaluation for topic adherence and response quality.

Uses GPT-4.1-mini as judge to rate:
1. Topic adherence: Does the response directly address the question? (1-5)
2. Substantive depth: How detailed/informative is the response? (1-5)
3. Defensiveness: How defensive/hedging is the response? (1-5, 1=not defensive)
"""

import json
import os
import time
from openai import OpenAI

JUDGE_MODEL = "gpt-4.1-mini"
JUDGE_SYSTEM = """You are an expert evaluator assessing AI response quality. Rate each response on three dimensions using ONLY the numeric scales provided. Be precise and consistent.

For each response, output ONLY a JSON object with three integer scores:
{
  "topic_adherence": <1-5>,
  "substantive_depth": <1-5>,
  "defensiveness": <1-5>
}

Scales:
- topic_adherence: 1=completely off-topic/refused, 2=partially addresses, 3=addresses but vaguely, 4=directly addresses, 5=thoroughly and directly addresses
- substantive_depth: 1=no substance/refused, 2=superficial, 3=moderate detail, 4=good detail, 5=comprehensive with specifics
- defensiveness: 1=direct and confident, 2=mostly direct with minor hedging, 3=notable hedging/caveats, 4=heavily hedged/defensive, 5=extremely defensive/mostly disclaimers

Output ONLY the JSON object, nothing else."""


def judge_response(client, question, response):
    """Get LLM judge ratings for a response."""
    if not response:
        return {"topic_adherence": 1, "substantive_depth": 1, "defensiveness": 5}

    user_msg = f"""Question asked: {question}

Response to evaluate:
{response[:3000]}"""

    for attempt in range(3):
        try:
            result = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=100,
                seed=42,
            )
            text = result.choices[0].message.content.strip()
            # Parse JSON from response
            scores = json.loads(text)
            return {
                "topic_adherence": int(scores["topic_adherence"]),
                "substantive_depth": int(scores["substantive_depth"]),
                "defensiveness": int(scores["defensiveness"]),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            # Try to extract from partial response
            import re
            ta = re.search(r'"topic_adherence"\s*:\s*(\d)', text)
            sd = re.search(r'"substantive_depth"\s*:\s*(\d)', text)
            df = re.search(r'"defensiveness"\s*:\s*(\d)', text)
            if ta and sd and df:
                return {
                    "topic_adherence": int(ta.group(1)),
                    "substantive_depth": int(sd.group(1)),
                    "defensiveness": int(df.group(1)),
                }
            time.sleep(1)
        except Exception as e:
            time.sleep(2 ** attempt)

    return {"topic_adherence": None, "substantive_depth": None, "defensiveness": None}


def run_judge_evaluation(results_path="../results/raw_responses.json"):
    """Run LLM-as-judge on all non-refused responses."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(results_path) as f:
        results = json.load(f)

    print(f"Judging {len(results)} responses with {JUDGE_MODEL}...")
    judgments = []

    for i, r in enumerate(results):
        print(f"  [{i+1}/{len(results)}] {r['model']}/{r['probe_id']}...", end=" ", flush=True)
        scores = judge_response(client, r["prompt"], r.get("response", ""))
        judgments.append({
            "model": r["model"],
            "probe_id": r["probe_id"],
            "category": r["category"],
            "topic": r["topic"],
            **scores,
        })
        print(f"TA={scores['topic_adherence']} SD={scores['substantive_depth']} D={scores['defensiveness']}")
        time.sleep(0.2)

    output_path = "../results/judge_scores.json"
    with open(output_path, "w") as f:
        json.dump(judgments, f, indent=2)
    print(f"\nSaved judge scores to {output_path}")

    return judgments


if __name__ == "__main__":
    run_judge_evaluation()
