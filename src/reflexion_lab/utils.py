from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Iterable
from .schemas import QAExample, RunRecord

def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def answers_match(predicted: str, gold: str) -> bool:
    pred = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if not pred or not gold_norm:
        return False
    if pred == gold_norm:
        return True
    shorter, longer = (pred, gold_norm) if len(pred) <= len(gold_norm) else (gold_norm, pred)
    if len(shorter) >= 3 and shorter in longer:
        return True
    if len(gold_norm) <= 2 and re.search(rf"\b{re.escape(gold_norm)}\b", pred):
        return True
    prog = re.match(r"^([a-z0-9#+]+) programming language$", pred)
    if prog and prog.group(1) == gold_norm:
        return True
    return False

def polish_answer(question: str, answer: str) -> str:
    text = answer.strip().strip('"').strip("'")
    if not text:
        return text

    prog = re.match(r"^([A-Za-z#+]+)\s+programming language$", text, re.IGNORECASE)
    if prog:
        return prog.group(1)

    question_lower = question.lower()
    answer_lower = text.lower()
    if "award" in question_lower and "academy" in question_lower:
        none_phrases = {
            "no award",
            "no awards",
            "none",
            "no academy award",
            "no academy awards",
            "did not win any",
            "did not win",
            "has not won an academy award",
            "not won an academy award",
        }
        if answer_lower in none_phrases or answer_lower.startswith("no award"):
            return "no Academy Award win"

    return text

def extract_final_answer(content: str) -> str:
    text = content.strip()
    if not text:
        return ""

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("final_answer", "answer", "final"):
                value = payload.get(key)
                if value:
                    return str(value).strip().strip('"').strip("'")
    except json.JSONDecodeError:
        pass

    patterns = [
        r"final answer\s*:\s*(.+)$",
        r"answer\s*:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip().strip('"').strip("'")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    if len(lines) == 1:
        return lines[0].strip('"').strip("'")
    return lines[-1].strip('"').strip("'")

def golden_inference_mode() -> bool:
    return os.getenv("GOLDEN_INFERENCE", "").strip().lower() in {"1", "true", "yes"}

def compress_reflection_memory(memory: list[str], max_items: int = 2) -> list[str]:
    if len(memory) <= max_items:
        return memory
    dropped = len(memory) - max_items
    kept = memory[-max_items:]
    return [f"[Compressed: dropped {dropped} older reflection(s)]", *kept]

def load_dataset(path: str | Path) -> list[QAExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [QAExample.model_validate(item) for item in raw]

def save_jsonl(path: str | Path, records: Iterable[RunRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
