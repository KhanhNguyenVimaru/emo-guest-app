"""
Batch-test the Gemini API against the Hugging Face `emotion` dataset using
blocks of 4 samples per request. The model must answer with JSON so we can
compare predictions with the dataset ground truth.

Usage examples:
  python test_gemini_blocks.py --blocks 2
  python test_gemini_blocks.py --blocks 5 --model gemini-1.5-flash

Requires environment variable GEMINI_API_KEY (or VITE_GEMINI_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, Iterable, List, Sequence

import google.generativeai as genai
from datasets import load_dataset
from tqdm import tqdm


LABELS: Sequence[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BLOCK_SIZE = 4
TOKENS_PER_SENTENCE = 512
DEFAULT_MODEL = "gemini-2.5-flash"
GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": BLOCK_SIZE * TOKENS_PER_SENTENCE,
    "response_mime_type": "application/json",
}

SYNONYMS: Dict[str, str] = {
    "happy": "joy",
    "happiness": "joy",
    "joyful": "joy",
    "ecstatic": "joy",
    "sad": "sadness",
    "depressed": "sadness",
    "angry": "anger",
    "mad": "anger",
    "furious": "anger",
    "afraid": "fear",
    "scared": "fear",
    "fearful": "fear",
    "terrified": "fear",
    "surprised": "surprise",
    "shocked": "surprise",
    "astonished": "surprise",
    "love": "love",
    "loved": "love",
    "loving": "love",
}


@dataclass
class Sample:
    dataset_index: int
    sentence: str
    gold_emotion: str
    local_id: int  # 1-based index inside the block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemini on the emotion dataset using 4-sample blocks.")
    parser.add_argument("--blocks", type=int, default=1, help="Number of 4-sample blocks to process.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model to invoke.")
    parser.add_argument("--split", default="test", help="Dataset split to read (default: test).")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Gemini API key (fallback to env vars).")
    return parser.parse_args()


def get_api_key(cli_key: str | None) -> str:
    candidate = cli_key or os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GEMINI_API_KEY") or ""
    key = candidate.strip()
    if not key:
        raise RuntimeError("Missing Gemini API key. Provide --api-key or set GEMINI_API_KEY/VITE_GEMINI_API_KEY.")
    return key


def chunk_samples(split: str, block_limit: int) -> Iterable[List[Sample]]:
    dataset = load_dataset("emotion", split=split)
    block: List[Sample] = []
    block_id = 0
    for idx, row in enumerate(dataset):
        gold = LABELS[row["label"]]
        block.append(Sample(dataset_index=idx, sentence=row["text"], gold_emotion=gold, local_id=len(block) + 1))
        if len(block) == BLOCK_SIZE:
            yield block
            block = []
            block_id += 1
            if block_id >= block_limit:
                break
    # Drop leftover samples to keep each block at BLOCK_SIZE exactly.


def build_block_prompt(block_id: int, samples: List[Sample]) -> str:
    header = (
        "You are an emotion classifier. Classify each sentence independently and respond with JSON only.\n"
        f"Allowed labels: {', '.join(LABELS)}.\n"
        "Required JSON schema:\n"
        '{\n'
        '  "block": <block_id>,\n'
        '  "results": [\n'
        f'    {{"local_id": <1-{BLOCK_SIZE}>, "dataset_index": <int>, "sentence": "<original text>", "predicted_emotion": "<label>"}}\n'
        "  ]\n"
        "}\n"
        "Do not omit any sentences and keep the sentence text verbatim."
    )
    lines = [header, f"Block #{block_id} sentences:"]
    for sample in samples:
        lines.append(f"{sample.local_id}. dataset_index={sample.dataset_index} :: {sample.sentence}")
    return "\n".join(lines)


def sanitize_json_text(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned


def normalize_label(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z ]", " ", raw).strip().lower()
    if not cleaned:
        return ""
    token = cleaned.split()[0]
    if token in LABELS:
        return token
    return SYNONYMS.get(token, token)


def trim_results_for_export(results: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return only the fields that must be exposed externally."""
    trimmed: List[Dict[str, str]] = []
    for item in results:
        trimmed.append(
            {
                "sentence": str(item.get("sentence", "")),
                "gold_emotion": str(item.get("gold_emotion", "")),
                "predicted_emotion": str(item.get("predicted_emotion", "")),
            }
        )
    return trimmed


def compute_accuracy_from_json(payload: str | Dict[str, Any]) -> float:
    """Compute accuracy % from a summary JSON (string or dict)."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Expected 'results' to be a list in the provided JSON payload.")
    total = len(results)
    if not total:
        return 0.0
    matches = 0
    for entry in results:
        if not isinstance(entry, dict):
            continue
        if entry.get("gold_emotion") == entry.get("predicted_emotion"):
            matches += 1
    return round(100 * matches / total, 2)


def parse_block_response(raw_text: str) -> Dict:
    cleaned = sanitize_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model JSON: {exc}: {raw_text}") from exc


def evaluate_blocks(model: genai.GenerativeModel, block_iter: Iterable[List[Sample]]) -> Dict:
    processed = 0
    total = 0
    correct = 0
    detailed_results = []

    for block_id, samples in enumerate(block_iter, start=1):
        prompt = build_block_prompt(block_id, samples)
        try:
            response = model.generate_content(prompt)
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed for block {block_id}") from exc
        raw_text = response.text or ""
        parsed = parse_block_response(raw_text)
        predictions: Dict[int, str] = {}
        for entry in parsed.get("results", []):
            try:
                local_id = int(entry["local_id"])
            except (KeyError, ValueError, TypeError):
                continue
            predictions[local_id] = normalize_label(str(entry.get("predicted_emotion", "")))

        for sample in samples:
            predicted = predictions.get(sample.local_id, "")
            is_correct = predicted == sample.gold_emotion
            if is_correct:
                correct += 1
            detailed_results.append(
                {
                    "block": block_id,
                    "dataset_index": sample.dataset_index,
                    "sentence": sample.sentence,
                    "gold_emotion": sample.gold_emotion,
                    "predicted_emotion": predicted,
                    "match": is_correct,
                }
            )
            total += 1
        processed += 1

    accuracy = 100 * correct / total if total else 0.0
    model_name = getattr(model, "model_name", DEFAULT_MODEL)
    trimmed_results = trim_results_for_export(detailed_results)
    summary = {
        "model": model_name,
        "block_size": BLOCK_SIZE,
        "blocks_processed": processed,
        "total_sentences": total,
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "results": trimmed_results,
    }
    # accuracy sanity-check derived from the exported JSON structure
    summary["accuracy"] = compute_accuracy_from_json(summary)
    return summary


def main() -> None:
    args = parse_args()
    api_key = get_api_key(args.api_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model, generation_config=GENERATION_CONFIG)

    block_iter = chunk_samples(args.split, args.blocks)
    try:
        first_block = next(block_iter)
    except StopIteration:
        print("No samples found for the requested configuration.", file=sys.stderr)
        sys.exit(1)

    summary = evaluate_blocks(model, tqdm(chain([first_block], block_iter), desc="Blocks", unit="block"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
