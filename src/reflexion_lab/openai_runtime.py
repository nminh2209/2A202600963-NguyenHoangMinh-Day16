from __future__ import annotations
import json
import os
import time
from dotenv import load_dotenv
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry, StepResult

load_dotenv()

from openai import OpenAI

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or set USE_MOCK_RUNTIME=1.")
        _client = OpenAI(api_key=api_key)
    return _client

def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _chat(system: str, user: str, *, json_mode: bool = False) -> tuple[str, int, int]:
    client = _get_client()
    kwargs: dict = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    start = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = response.usage
    tokens = usage.total_tokens if usage else 0
    content = response.choices[0].message.content or ""
    return content, tokens, latency_ms

def _format_context(example: QAExample) -> str:
    blocks = []
    for chunk in example.context:
        blocks.append(f"[{chunk.title}]\n{chunk.text}")
    return "\n\n".join(blocks)

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> StepResult[str]:
    memory_block = ""
    if reflection_memory:
        memory_block = "Prior reflections:\n" + "\n".join(f"- {item}" for item in reflection_memory)
    user = f"""Question: {example.question}

Context:
{_format_context(example)}

Attempt: {attempt_id}
Agent mode: {agent_type}
{memory_block}

Return only the final answer."""
    content, tokens, latency_ms = _chat(ACTOR_SYSTEM, user)
    answer = content.strip().splitlines()[0].strip().strip('"').strip("'")
    return StepResult(value=answer, token_estimate=tokens, latency_ms=latency_ms)

def evaluator(example: QAExample, answer: str) -> StepResult[JudgeResult]:
    user = f"""Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Context:
{_format_context(example)}"""
    content, tokens, latency_ms = _chat(EVALUATOR_SYSTEM, user, json_mode=True)
    payload = json.loads(content)
    score = int(payload.get("score", 0))
    if score not in (0, 1):
        score = 0
    judge = JudgeResult(
        score=score,  # type: ignore[arg-type]
        reason=str(payload.get("reason", "")),
        missing_evidence=[str(x) for x in payload.get("missing_evidence", [])],
        spurious_claims=[str(x) for x in payload.get("spurious_claims", [])],
    )
    return StepResult(value=judge, token_estimate=tokens, latency_ms=latency_ms)

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> StepResult[ReflectionEntry]:
    user = f"""Question: {example.question}

Evaluator feedback:
- reason: {judge.reason}
- missing_evidence: {judge.missing_evidence}
- spurious_claims: {judge.spurious_claims}

Attempt: {attempt_id}

Context:
{_format_context(example)}"""
    content, tokens, latency_ms = _chat(REFLECTOR_SYSTEM, user, json_mode=True)
    payload = json.loads(content)
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=str(payload.get("failure_reason", judge.reason)),
        lesson=str(payload.get("lesson", "")),
        next_strategy=str(payload.get("next_strategy", "")),
    )
    return StepResult(value=entry, token_estimate=tokens, latency_ms=latency_ms)
