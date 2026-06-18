from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .failure_modes import infer_failure_mode
from .runtime import actor_answer, evaluator, plan_question, reflector, use_mock_runtime
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord
from .utils import answers_match, compress_reflection_memory, golden_inference_mode, normalize_answer

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    memory_limit: int = 3

    def _apply_hybrid_judge(self, example: QAExample, answer: str, judge: JudgeResult) -> JudgeResult:
        if golden_inference_mode() or not example.gold_answer.strip():
            return judge
        if answers_match(answer, example.gold_answer):
            return JudgeResult(
                score=1,
                reason="Matched gold answer after normalization.",
                missing_evidence=[],
                spurious_claims=[],
            )
        return judge

    def _is_looping(self, traces: list[AttemptTrace], answer: str) -> bool:
        if len(traces) < 2:
            return False
        current = normalize_answer(answer)
        return all(normalize_answer(trace.answer) == current for trace in traces[-2:])

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        wrong_answers: list[str] = []
        prior_strategies: list[str] = []
        plan = ""
        relevant_titles: list[str] = []
        plan_result = None
        final_answer = ""
        final_score = 0
        judge = JudgeResult(score=0, reason="No attempts completed.")

        for attempt_id in range(1, self.max_attempts + 1):
            if self.agent_type == "reflexion" and attempt_id >= 2:
                plan_result = plan_question(example, wrong_answers, reflection_memory)
                plan = plan_result.value
                if "Relevant titles:" in plan:
                    titles_part = plan.split("Relevant titles:", maxsplit=1)[1].split(". Hops:", maxsplit=1)[0]
                    relevant_titles = [title.strip() for title in titles_part.split(",") if title.strip()]

            actor_result = actor_answer(
                example,
                attempt_id,
                self.agent_type,
                reflection_memory,
                plan=plan if self.agent_type == "reflexion" else "",
                wrong_answers=wrong_answers,
                relevant_titles=relevant_titles,
            )
            answer = actor_result.value
            eval_result = evaluator(example, answer)
            judge = self._apply_hybrid_judge(example, answer, eval_result.value)
            token_estimate = actor_result.token_estimate + eval_result.token_estimate
            latency_ms = actor_result.latency_ms + eval_result.latency_ms
            if plan_result is not None and attempt_id >= 2:
                token_estimate += plan_result.token_estimate
                latency_ms += plan_result.latency_ms
            reflection_entry: ReflectionEntry | None = None

            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(
                    AttemptTrace(
                        attempt_id=attempt_id,
                        answer=answer,
                        score=judge.score,
                        reason=judge.reason,
                        token_estimate=token_estimate,
                        latency_ms=latency_ms,
                    )
                )
                break

            wrong_answers.append(answer)
            looping = self._is_looping(traces, answer)

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if not looping:
                    reflect_result = reflector(
                        example,
                        attempt_id,
                        judge,
                        wrong_answers=wrong_answers,
                        prior_strategies=prior_strategies,
                    )
                    reflection_entry = reflect_result.value
                    reflections.append(reflection_entry)
                    prior_strategies.append(reflection_entry.next_strategy)
                    token_estimate += reflect_result.token_estimate
                    latency_ms += reflect_result.latency_ms
                    memory_line = (
                        f"Attempt {attempt_id}: {reflection_entry.lesson} "
                        f"Strategy: {reflection_entry.next_strategy}"
                    )
                    reflection_memory = compress_reflection_memory(
                        [*reflection_memory, memory_line],
                        max_items=self.memory_limit,
                    )
                else:
                    reflection_memory = compress_reflection_memory(
                        [
                            *reflection_memory,
                            "Loop detected: give a DIFFERENT answer using a different bridge passage.",
                        ],
                        max_items=self.memory_limit,
                    )

            traces.append(
                AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    reflection=reflection_entry,
                    token_estimate=token_estimate,
                    latency_ms=latency_ms,
                )
            )

        total_tokens = sum(item.token_estimate for item in traces)
        total_latency = sum(item.latency_ms for item in traces)

        if final_score == 1:
            failure_mode = "none"
        elif use_mock_runtime():
            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        elif len(traces) >= 3 and self._is_looping(traces, traces[-1].answer):
            failure_mode = "reflection_overfit" if self.agent_type == "reflexion" else "looping"
        else:
            failure_mode = infer_failure_mode(final_score, judge, traces)

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=(
                answers_match(final_answer, example.gold_answer)
                if example.gold_answer.strip()
                else bool(final_score)
            ),
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
    def __init__(self, max_attempts: int = 3, memory_limit: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, memory_limit=memory_limit)
