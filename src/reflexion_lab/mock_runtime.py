from __future__ import annotations
from .schemas import JudgeResult, QAExample, ReflectionEntry, StepResult
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
    **_: object,
) -> StepResult[str]:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        answer = example.gold_answer
    elif agent_type == "react":
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    elif attempt_id == 1 and not reflection_memory:
        answer = FIRST_ATTEMPT_WRONG[example.qid]
    else:
        answer = example.gold_answer
    token_estimate = 320 + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0)
    latency_ms = 160 + (attempt_id * 40) + (90 if agent_type == "reflexion" else 0)
    return StepResult(value=answer, token_estimate=token_estimate, latency_ms=latency_ms)

def evaluator(example: QAExample, answer: str) -> StepResult[JudgeResult]:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        judge = JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    elif normalize_answer(answer) == "london":
        judge = JudgeResult(
            score=0,
            reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
            missing_evidence=["Need to identify the river that flows through London."],
            spurious_claims=[],
        )
    else:
        judge = JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        )
    return StepResult(value=judge, token_estimate=95, latency_ms=45)

def plan_question(example: QAExample, wrong_answers: list[str], reflection_memory: list[str]) -> StepResult[str]:
    plan = "Do the second hop explicitly and verify the final entity against context."
    return StepResult(value=plan, token_estimate=80, latency_ms=40)

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult, **_: object) -> StepResult[ReflectionEntry]:
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if example.qid == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )
    return StepResult(value=entry, token_estimate=110, latency_ms=55)
