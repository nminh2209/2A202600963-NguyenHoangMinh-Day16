from __future__ import annotations
from .schemas import AttemptTrace, JudgeResult

def infer_failure_mode(final_score: int, judge: JudgeResult, traces: list[AttemptTrace]) -> str:
    if final_score == 1:
        return "none"
    if len(traces) >= 2 and traces[-1].answer.strip().lower() == traces[-2].answer.strip().lower():
        return "looping"
    reason_lower = judge.reason.lower()
    if judge.missing_evidence and any("hop" in item.lower() for item in judge.missing_evidence):
        return "incomplete_multi_hop"
    if any(token in reason_lower for token in ("hop", "incomplete", "partial", "intermediate", "first hop")):
        return "incomplete_multi_hop"
    if judge.spurious_claims:
        return "entity_drift"
    if any(token in reason_lower for token in ("drift", "wrong entity", "second-hop", "second hop")):
        return "entity_drift"
    return "wrong_final_answer"
