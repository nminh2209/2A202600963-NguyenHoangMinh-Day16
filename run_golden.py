from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print
from rich.progress import track

load_dotenv()
if "--mock" in sys.argv:
    os.environ["USE_MOCK_RUNTIME"] = "1"
elif "--live" in sys.argv:
    os.environ["USE_MOCK_RUNTIME"] = "0"

os.environ.setdefault("GOLDEN_INFERENCE", "1")

from src.reflexion_lab.agents import ReflexionAgent
from src.reflexion_lab.runtime import reset_runtime_backend, use_mock_runtime
from src.reflexion_lab.utils import load_dataset

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = typer.Argument(..., help="Path to golden test JSON (question + context, gold optional)."),
    out_dir: str = "outputs/golden_run",
    reflexion_attempts: int = 3,
) -> None:
    os.environ["GOLDEN_INFERENCE"] = "1"
    os.environ["USE_MOCK_RUNTIME"] = "0"
    reset_runtime_backend()

    print(f"[cyan]Golden inference mode[/cyan] (no gold answer in evaluator)")
    print(f"[cyan]Runtime:[/cyan] {'mock' if use_mock_runtime() else 'live'}")

    examples = load_dataset(dataset)
    agent = ReflexionAgent(max_attempts=reflexion_attempts)
    predictions = []
    records = []
    for example in track(examples, description="Reflexion"):
        record = agent.run(example)
        records.append(record)
        predictions.append(
            {
                "qid": record.qid,
                "question": record.question,
                "predicted_answer": record.predicted_answer,
                "attempts": record.attempts,
                "failure_mode": record.failure_mode,
            }
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_path = out_path / "predictions.json"
    runs_path = out_path / "reflexion_runs.jsonl"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    with runs_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")

    print(f"[green]Saved[/green] {pred_path}")
    print(f"[green]Saved[/green] {runs_path}")
    print(f"Predictions: {len(predictions)}")

if __name__ == "__main__":
    app()
