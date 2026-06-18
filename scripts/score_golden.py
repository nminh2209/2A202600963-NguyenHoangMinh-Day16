import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reflexion_lab.utils import load_dataset, answers_match

examples = {e.qid: e for e in load_dataset(ROOT / "hotpot_golden.json")}
preds = json.loads((ROOT / "outputs/golden_run/predictions.json").read_text(encoding="utf-8"))
correct = 0
for p in preds:
    ex = examples[p["qid"]]
    ok = answers_match(p["predicted_answer"], ex.gold_answer)
    correct += int(ok)
    status = "OK" if ok else "MISS"
    if not ok:
        print(f"{status} {p['qid']}: pred={p['predicted_answer']!r} gold={ex.gold_answer!r}")
print(f"EM: {correct}/{len(preds)} = {correct/len(preds):.1%}")
