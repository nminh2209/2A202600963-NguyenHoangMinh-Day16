from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .failure_modes import infer_failure_mode
from .runtime import actor_answer, evaluator, reflector, use_mock_runtime
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        judge = JudgeResult(score=0, reason="No attempts completed.")

        for attempt_id in range(1, self.max_attempts + 1):
            actor_result = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            answer = actor_result.value
            eval_result = evaluator(example, answer)
            judge = eval_result.value
            token_estimate = actor_result.token_estimate + eval_result.token_estimate
            latency_ms = actor_result.latency_ms + eval_result.latency_ms
            reflection_entry: ReflectionEntry | None = None

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    token_estimate=token_estimate,
                    latency_ms=latency_ms,
                )
                traces.append(trace)
                break

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflect_result = reflector(example, attempt_id, judge)
                reflection_entry = reflect_result.value
                reflections.append(reflection_entry)
                token_estimate += reflect_result.token_estimate
                latency_ms += reflect_result.latency_ms
                memory_line = (
                    f"Attempt {attempt_id}: {reflection_entry.lesson} "
                    f"Strategy: {reflection_entry.next_strategy}"
                )
                reflection_memory.append(memory_line)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)

        if final_score == 1:
            failure_mode = "none"
        elif use_mock_runtime():
            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        else:
            failure_mode = infer_failure_mode(final_score, judge, traces)

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,  # type: ignore[arg-type]
            reflections=reflections,
            traces=traces,
        )

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
