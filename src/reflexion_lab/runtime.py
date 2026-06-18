from __future__ import annotations
import os
from dotenv import load_dotenv
from .schemas import JudgeResult, QAExample, ReflectionEntry, StepResult

load_dotenv()

_IMPL = None

def use_mock_runtime() -> bool:
    flag = os.getenv("USE_MOCK_RUNTIME", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return not os.getenv("OPENAI_API_KEY")

def _backend():
    global _IMPL
    if _IMPL is None:
        if use_mock_runtime():
            from . import mock_runtime as _IMPL  # type: ignore[assignment]
        else:
            from . import openai_runtime as _IMPL  # type: ignore[assignment]
    return _IMPL

def reset_runtime_backend() -> None:
    global _IMPL
    _IMPL = None

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> StepResult[str]:
    return _backend().actor_answer(example, attempt_id, agent_type, reflection_memory)

def evaluator(example: QAExample, answer: str) -> StepResult[JudgeResult]:
    return _backend().evaluator(example, answer)

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> StepResult[ReflectionEntry]:
    return _backend().reflector(example, attempt_id, judge)

from .mock_runtime import FAILURE_MODE_BY_QID

__all__ = ["actor_answer", "evaluator", "reflector", "FAILURE_MODE_BY_QID", "use_mock_runtime", "reset_runtime_backend"]
