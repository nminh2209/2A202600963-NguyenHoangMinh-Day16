from __future__ import annotations
import json
import os
import time
from dotenv import load_dotenv
from .context_ranking import select_context
from .prompts import (
    ACTOR_REACT_SYSTEM,
    ACTOR_REFLEXION_SYSTEM,
    EVALUATOR_SYSTEM,
    PLANNER_SYSTEM,
    REFLECTOR_SYSTEM,
    SELF_EVALUATOR_SYSTEM,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry, StepResult
from .utils import extract_final_answer, golden_inference_mode, polish_answer

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

def _context_limit() -> int:
    return int(os.getenv("CONTEXT_TOP_K", "10"))

def _full_context_threshold() -> int:
    return int(os.getenv("CONTEXT_FULL_THRESHOLD", "12"))

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

def _format_context(example: QAExample, extra_titles: list[str] | None = None) -> str:
    chunks = select_context(
        example.question,
        example.context,
        top_k=_context_limit(),
        full_threshold=_full_context_threshold(),
        extra_titles=extra_titles,
    )
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in chunks)

def _parse_judge(content: str) -> JudgeResult:
    payload = json.loads(content)
    score = int(payload.get("score", 0))
    if score not in (0, 1):
        score = 0
    return JudgeResult(
        score=score,  # type: ignore[arg-type]
        reason=str(payload.get("reason", "")),
        missing_evidence=[str(item) for item in payload.get("missing_evidence", [])],
        spurious_claims=[str(item) for item in payload.get("spurious_claims", [])],
    )

def plan_question(
    example: QAExample,
    wrong_answers: list[str],
    reflection_memory: list[str],
) -> StepResult[str]:
    wrong_block = ""
    if wrong_answers:
        wrong_block = "Prior wrong answers:\n" + "\n".join(f"- {ans}" for ans in wrong_answers)
    memory_block = ""
    if reflection_memory:
        memory_block = "Reflections:\n" + "\n".join(f"- {item}" for item in reflection_memory)
    user = f"""Question: {example.question}

Context:
{_format_context(example)}

{wrong_block}
{memory_block}"""
    content, tokens, latency_ms = _chat(PLANNER_SYSTEM, user, json_mode=True)
    payload = json.loads(content)
    relevant_titles = [str(title) for title in payload.get("relevant_titles", [])]
    plan = (
        f"Type: {payload.get('question_type', 'bridge')}. "
        f"Relevant titles: {', '.join(relevant_titles)}. "
        f"Hops: {' | '.join(payload.get('hops', []))}. "
        f"Answer type: {payload.get('answer_type', 'entity')}."
    )
    return StepResult(value=plan, token_estimate=tokens, latency_ms=latency_ms)

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    *,
    plan: str = "",
    wrong_answers: list[str] | None = None,
    relevant_titles: list[str] | None = None,
) -> StepResult[str]:
    wrong_answers = wrong_answers or []
    is_reflexion = agent_type == "reflexion" and (attempt_id > 1 or wrong_answers or reflection_memory)
    memory_block = ""
    if reflection_memory:
        memory_block = "Prior reflections:\n" + "\n".join(f"- {item}" for item in reflection_memory)
    wrong_block = ""
    if wrong_answers:
        wrong_block = "Do NOT repeat these wrong answers:\n" + "\n".join(f"- {ans}" for ans in wrong_answers)
    plan_block = f"Plan:\n{plan}" if plan else ""

    context_text = _format_context(example, extra_titles=relevant_titles)
    user = f"""Question: {example.question}

Context:
{context_text}

Attempt: {attempt_id}
{plan_block}
{wrong_block}
{memory_block}"""

    if is_reflexion:
        content, tokens, latency_ms = _chat(ACTOR_REFLEXION_SYSTEM, user, json_mode=True)
        answer = extract_final_answer(content)
    else:
        content, tokens, latency_ms = _chat(ACTOR_REACT_SYSTEM, user, json_mode=False)
        answer = extract_final_answer(content)
    answer = polish_answer(example.question, answer)
    return StepResult(value=answer, token_estimate=tokens, latency_ms=latency_ms)

def evaluator(example: QAExample, answer: str) -> StepResult[JudgeResult]:
    context_text = _format_context(example)
    use_self_eval = golden_inference_mode() or not example.gold_answer.strip()
    if use_self_eval:
        user = f"""Question: {example.question}
Predicted answer: {answer}

Context:
{context_text}"""
        content, tokens, latency_ms = _chat(SELF_EVALUATOR_SYSTEM, user, json_mode=True)
    else:
        user = f"""Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Context:
{context_text}"""
        content, tokens, latency_ms = _chat(EVALUATOR_SYSTEM, user, json_mode=True)
    return StepResult(value=_parse_judge(content), token_estimate=tokens, latency_ms=latency_ms)

def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
    *,
    wrong_answers: list[str] | None = None,
    prior_strategies: list[str] | None = None,
) -> StepResult[ReflectionEntry]:
    wrong_answers = wrong_answers or []
    prior_strategies = prior_strategies or []
    user = f"""Question: {example.question}

Wrong answers so far:
{wrong_answers}

Evaluator feedback:
- reason: {judge.reason}
- missing_evidence: {judge.missing_evidence}
- spurious_claims: {judge.spurious_claims}

Prior strategies already tried (do not repeat):
{prior_strategies}

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
