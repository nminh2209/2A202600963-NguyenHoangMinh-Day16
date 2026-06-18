from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

def _difficulty(question_type: str, num_context: int) -> str:
    if question_type == "comparison" or num_context >= 10:
        return "hard"
    if num_context >= 6:
        return "medium"
    return "easy"

def convert_row(item: dict, index: int) -> dict:
    context = []
    titles = item["context"]["title"]
    sentences_list = item["context"]["sentences"]
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(sentences).strip()
        if text:
            context.append({"title": title, "text": text})
    return {
        "qid": f"hotpot_{index:04d}",
        "difficulty": _difficulty(item.get("type", "bridge"), len(context)),
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HotpotQA subset for Lab 16.")
    parser.add_argument("--count", type=int, default=55, help="Number of examples to export.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/hotpot_subset_55.json")
    args = parser.parse_args()

    parquet_path = hf_hub_download(
        repo_id="hotpot_qa",
        repo_type="dataset",
        filename="distractor/validation-00000-of-00001.parquet",
    )
    frame = pd.read_parquet(parquet_path)
    indices = list(range(len(frame)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[: args.count]

    examples = []
    for rank, row_index in enumerate(selected, start=1):
        row = frame.iloc[row_index].to_dict()
        examples.append(convert_row(row, rank))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {out_path}")

if __name__ == "__main__":
    main()
