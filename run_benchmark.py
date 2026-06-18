from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print

load_dotenv()
if "--mock" in sys.argv:
    os.environ["USE_MOCK_RUNTIME"] = "1"
elif "--live" in sys.argv:
    os.environ["USE_MOCK_RUNTIME"] = "0"
if "--golden" not in sys.argv:
    os.environ.setdefault("GOLDEN_INFERENCE", "0")

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.runtime import reset_runtime_backend, use_mock_runtime
from src.reflexion_lab.utils import load_dataset, save_jsonl
from rich.progress import track

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mock: Optional[bool] = typer.Option(None, "--mock/--live", help="Force mock runtime or live OpenAI calls."),
) -> None:
    if mock is not None:
        os.environ["USE_MOCK_RUNTIME"] = "1" if mock else "0"
    reset_runtime_backend()

    mode_label = "mock" if use_mock_runtime() else "live"
    print(f"[cyan]Runtime mode:[/cyan] {mode_label}")

    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    react_records = [react.run(example) for example in track(examples, description="ReAct")]
    reflexion_records = [reflexion.run(example) for example in track(examples, description="Reflexion")]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode_label)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
