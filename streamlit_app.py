from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.runtime import reset_runtime_backend, use_mock_runtime
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.utils import load_dataset, golden_inference_mode

DEFAULT_REPORT_DIR = ROOT / "outputs" / "full_run"

st.set_page_config(page_title="Reflexion Agent Lab 16", page_icon="🪞", layout="wide")


@st.cache_data
def load_report(report_dir: str) -> dict | None:
    path = Path(report_dir) / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_run_records(report_dir: str, agent_type: str) -> list[dict]:
    path = Path(report_dir) / f"{agent_type}_runs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text)


def render_overview(report: dict, report_dir: Path) -> None:
    st.title("Lab 16 — Reflexion Agent Demo")
    st.caption("ReAct vs Reflexion on HotpotQA-style multi-hop question answering")

    meta = report["meta"]
    summary = report["summary"]
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset", meta.get("dataset", "n/a"))
    with c2:
        metric_card("Mode", meta.get("mode", "n/a"))
    with c3:
        metric_card("Records", str(meta.get("num_records", 0)))
    with c4:
        metric_card("EM gain", f"{delta.get('em_abs', 0):+.1%}")

    st.markdown("### Architecture")
    st.code(
        "Question + Context → Actor → Evaluator → (if wrong) Reflector → memory → retry",
        language="text",
    )

    st.markdown("### Summary")
    table = pd.DataFrame(
        {
            "Metric": ["Exact Match", "Avg attempts", "Avg tokens", "Avg latency (ms)"],
            "ReAct": [
                react.get("em", 0),
                react.get("avg_attempts", 0),
                react.get("avg_token_estimate", 0),
                react.get("avg_latency_ms", 0),
            ],
            "Reflexion": [
                reflexion.get("em", 0),
                reflexion.get("avg_attempts", 0),
                reflexion.get("avg_token_estimate", 0),
                reflexion.get("avg_latency_ms", 0),
            ],
            "Delta": [
                delta.get("em_abs", 0),
                delta.get("attempts_abs", 0),
                delta.get("tokens_abs", 0),
                delta.get("latency_abs", 0),
            ],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    chart_df = pd.DataFrame(
        {
            "agent": ["ReAct", "Reflexion"],
            "em": [react.get("em", 0), reflexion.get("em", 0)],
            "tokens": [react.get("avg_token_estimate", 0), reflexion.get("avg_token_estimate", 0)],
            "latency_ms": [react.get("avg_latency_ms", 0), reflexion.get("avg_latency_ms", 0)],
        }
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(chart_df, x="agent", y="em", title="Exact Match", range_y=[0, 1]), use_container_width=True)
    with right:
        st.plotly_chart(px.bar(chart_df, x="agent", y="tokens", title="Average Tokens"), use_container_width=True)

    st.markdown("### Failure modes")
    st.json(report.get("failure_modes", {}))

    st.markdown("### Extensions")
    for item in report.get("extensions", []):
        st.markdown(f"- `{item}`")

    st.markdown("### Discussion")
    st.write(report.get("discussion", ""))

    md_path = report_dir / "report.md"
    if md_path.exists():
        with st.expander("Full markdown report"):
            st.markdown(md_path.read_text(encoding="utf-8"))


def render_examples(report_dir: Path) -> None:
    st.subheader("Question explorer")
    react_rows = load_run_records(str(report_dir), "react")
    reflex_rows = load_run_records(str(report_dir), "reflexion")
    if not react_rows:
        st.warning("No run records found.")
        return

    qids = sorted({row["qid"] for row in react_rows})
    qid = st.selectbox("Question ID", qids)
    react_row = next(row for row in react_rows if row["qid"] == qid)
    reflex_row = next((row for row in reflex_rows if row["qid"] == qid), None)

    st.markdown(f"**Question:** {react_row['question']}")
    st.markdown(f"**Gold answer:** `{react_row['gold_answer']}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ReAct")
        st.markdown(f"Predicted: `{react_row['predicted_answer']}`")
        st.markdown(f"Correct: **{react_row['is_correct']}** | Attempts: {react_row['attempts']}")
        st.markdown(f"Failure mode: `{react_row['failure_mode']}`")
        for trace in react_row.get("traces", []):
            st.json(trace)
    with col2:
        st.markdown("#### Reflexion")
        if reflex_row:
            st.markdown(f"Predicted: `{reflex_row['predicted_answer']}`")
            st.markdown(f"Correct: **{reflex_row['is_correct']}** | Attempts: {reflex_row['attempts']}")
            st.markdown(f"Failure mode: `{reflex_row['failure_mode']}`")
            if reflex_row.get("reflections"):
                st.markdown("**Reflections**")
                for reflection in reflex_row["reflections"]:
                    st.info(
                        f"Attempt {reflection['attempt_id']}: {reflection['lesson']}\n\n"
                        f"Strategy: {reflection['next_strategy']}"
                    )
            for trace in reflex_row.get("traces", []):
                st.json(trace)


def render_live_demo() -> None:
    st.subheader("Live single-question demo")
    dataset_files = sorted((ROOT / "data").glob("*.json"))
    dataset_path = st.selectbox(
        "Dataset file",
        [str(path) for path in dataset_files],
        index=0 if dataset_files else None,
    )
    use_mock = st.toggle("Use mock runtime (no API cost)", value=use_mock_runtime())
    golden_mode = st.toggle("Golden inference mode (no gold in evaluator)", value=golden_inference_mode())
    if use_mock:
        os.environ["USE_MOCK_RUNTIME"] = "1"
    else:
        os.environ["USE_MOCK_RUNTIME"] = "0"
    os.environ["GOLDEN_INFERENCE"] = "1" if golden_mode else "0"
    reset_runtime_backend()

    if not dataset_path:
        st.stop()

    examples = load_dataset(dataset_path)
    labels = [f"{ex.qid} — {ex.question[:80]}" for ex in examples]
    idx = st.selectbox("Example", range(len(labels)), format_func=lambda i: labels[i])
    example = examples[idx]

    st.markdown(f"**Question:** {example.question}")
    st.markdown(f"**Gold answer:** `{example.gold_answer}`")
    with st.expander("Context passages"):
        for chunk in example.context:
            st.markdown(f"**{chunk.title}**")
            st.write(chunk.text)

    if st.button("Run ReAct + Reflexion", type="primary"):
        with st.spinner("Running agents..."):
            react_record = ReActAgent().run(example)
            reflex_record = ReflexionAgent(max_attempts=3).run(example)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ReAct result")
            st.json(react_record.model_dump())
        with c2:
            st.markdown("### Reflexion result")
            st.json(reflex_record.model_dump())


def main() -> None:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Page", ["Benchmark report", "Question explorer", "Live demo"])
    report_dir = Path(st.sidebar.text_input("Report directory", str(DEFAULT_REPORT_DIR)))

    if page == "Benchmark report":
        report = load_report(str(report_dir))
        if report is None:
            st.error(f"No report found at `{report_dir / 'report.json'}`. Run the benchmark first.")
            st.code(
                "python run_benchmark.py --dataset data/hotpot_subset_55.json "
                "--out-dir outputs/full_run --live",
                language="bash",
            )
            return
        render_overview(report, report_dir)
    elif page == "Question explorer":
        render_examples(report_dir)
    else:
        render_live_demo()


if __name__ == "__main__":
    main()
