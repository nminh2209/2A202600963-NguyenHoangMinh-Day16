import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.reporting import build_report, save_report

def load_records(path: Path) -> list[RunRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(RunRecord.model_validate_json(line))
    return records

def main() -> None:
    out = ROOT / "outputs" / "full_run_100"
    records = []
    for name in ("react_runs.jsonl", "reflexion_runs.jsonl"):
        records.extend(load_records(out / name))
    report = build_report(
        records,
        dataset_name="hotpot_subset_100.json",
        mode="live",
        golden_dataset=ROOT / "hotpot_golden.json",
        golden_predictions=ROOT / "outputs" / "golden_run" / "predictions.json",
    )
    json_path, md_path = save_report(report, out)
    print(f"Updated {json_path}")
    print(f"Updated {md_path}")

if __name__ == "__main__":
    main()
