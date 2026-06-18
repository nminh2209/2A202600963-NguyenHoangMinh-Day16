from __future__ import annotations
import os
from dotenv import load_dotenv

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

def plan_question(example, wrong_answers, reflection_memory):
    return _backend().plan_question(example, wrong_answers, reflection_memory)

def actor_answer(example, attempt_id, agent_type, reflection_memory, **kwargs):
    return _backend().actor_answer(example, attempt_id, agent_type, reflection_memory, **kwargs)

def evaluator(example, answer):
    return _backend().evaluator(example, answer)

def reflector(example, attempt_id, judge, **kwargs):
    return _backend().reflector(example, attempt_id, judge, **kwargs)

from .mock_runtime import FAILURE_MODE_BY_QID

__all__ = [
    "actor_answer",
    "evaluator",
    "plan_question",
    "reflector",
    "FAILURE_MODE_BY_QID",
    "use_mock_runtime",
    "reset_runtime_backend",
]
