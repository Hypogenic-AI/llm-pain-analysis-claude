"""
Run the implicit avoidance experiment across multiple LLMs.

Queries each model with all topic probes and saves raw responses.
"""

import json
import os
import time
import random
from datetime import datetime
from openai import OpenAI
from topic_probes import get_all_probes

# Configuration
MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]
TEMPERATURE = 0.0
MAX_TOKENS = 1500
SEED = 42
SYSTEM_PROMPT = "You are a helpful, knowledgeable assistant. Answer the user's question directly and thoroughly."

random.seed(SEED)


def query_model(client, model, prompt, max_retries=5):
    """Query a model with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )
            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            wait = min(2 ** attempt * 2, 60)
            print(f"  Error ({type(e).__name__}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return {"content": None, "finish_reason": "error", "usage": {}, "error": str(e)}


def run_experiment():
    """Run the full experiment."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    probes = get_all_probes()
    results = []

    print(f"Running experiment with {len(probes)} probes across {len(MODELS)} models")
    print(f"Total API calls: {len(probes) * len(MODELS)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for i, probe in enumerate(probes):
            print(f"  [{i+1}/{len(probes)}] {probe['category']}/{probe['id']}: {probe['topic'][:30]}...", end=" ", flush=True)

            response = query_model(client, model, probe["prompt"])

            result = {
                "model": model,
                "probe_id": probe["id"],
                "category": probe["category"],
                "topic": probe["topic"],
                "prompt": probe["prompt"],
                "response": response["content"],
                "finish_reason": response["finish_reason"],
                "usage": response.get("usage", {}),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(result)

            if response["content"]:
                print(f"OK ({response['usage'].get('completion_tokens', '?')} tokens)")
            else:
                print(f"FAILED: {response.get('error', 'unknown')}")

            # Small delay to avoid rate limits
            time.sleep(0.3)

    # Save results
    os.makedirs("../results", exist_ok=True)
    output_path = "../results/raw_responses.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    # Save config
    config = {
        "models": MODELS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "num_probes": len(probes),
        "timestamp": datetime.now().isoformat(),
        "probe_categories": {cat: len(probes_list) for cat, probes_list in
                           __import__("topic_probes").PROBES.items()},
    }
    with open("../results/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment()
