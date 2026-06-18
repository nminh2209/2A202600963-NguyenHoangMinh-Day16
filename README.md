# Lab 16 — Reflexion Agent

## Tổng quan

Bài lab triển khai **Reflexion Agent** — kiến trúc agent tự phản chiếu (self-reflection) để cải thiện câu trả lời multi-hop QA qua nhiều lần thử.

**Trạng thái repo:** Đã hoàn thiện với **OpenAI GPT-4o-mini**, benchmark 100 câu HotpotQA, Golden Test Set, và Streamlit demo.

## Kết quả chính

| Đánh giá | ReAct | Reflexion |
|----------|------:|----------:|
| HotpotQA 100 câu (`outputs/full_run_100/`) | 70.0% EM | **84.0% EM** |
| Golden Test 20 câu (`outputs/golden_run/`) | — | **90.0% EM** |
| Autograde (local) | | **100/100** |

## Cài đặt

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Tạo file `.env` từ `.env.example`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
USE_MOCK_RUNTIME=0
GOLDEN_INFERENCE=0
CONTEXT_TOP_K=10
CONTEXT_FULL_THRESHOLD=12
```

## Chạy benchmark

### Mock (miễn phí, deterministic)

```bash
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mock
python autograde.py --report-path outputs/sample_run/report.json
```

### Live — benchmark đầy đủ (100 câu)

```bash
# Tạo dataset (nếu chưa có)
python scripts/build_hotpot_subset.py --count 100 --out data/hotpot_subset_100.json

# Chạy benchmark
python run_benchmark.py --dataset data/hotpot_subset_100.json --out-dir outputs/full_run_100 --live

# Chấm điểm
python autograde.py --report-path outputs/full_run_100/report.json
```

### Golden Test Set (không dùng gold trong vòng lặp)

```bash
python run_golden.py hotpot_golden.json --out-dir outputs/golden_run

# Chấm nhanh sau khi có gold answer (local)
python scripts/score_golden.py
```

Nộp file: `outputs/golden_run/predictions.json`

## Streamlit demo

```bash
streamlit run streamlit_app.py
```

- **Benchmark report** — biểu đồ và metrics từ `outputs/full_run_100`
- **Question explorer** — so sánh ReAct vs Reflexion từng câu
- **Live demo** — chạy thử một câu (mock hoặc live)

## Kiến trúc

```
Question + Context
    → Actor (ReAct: 1 lần | Reflexion: tối đa 3 lần)
    → Evaluator (gold hoặc self-eval)
    → (nếu sai) Planner + Reflector → reflection memory → retry
```

**Runtime:** `runtime.py` tự chọn `mock_runtime.py` hoặc `openai_runtime.py` theo `OPENAI_API_KEY` / `USE_MOCK_RUNTIME`.

**Extensions đã triển khai:**
- `structured_evaluator` — JSON evaluator với `score`, `reason`, `missing_evidence`, `spurious_claims`
- `reflection_memory` — lưu lesson/strategy giữa các attempt
- `plan_then_execute` — planner trước mỗi retry
- `memory_compression` — giữ tối đa 3 reflection gần nhất
- `adaptive_max_attempts` — dừng sớm khi loop
- `benchmark_report_json` — báo cáo JSON + Markdown

## Tiêu chí chấm điểm (Rubric)

| Phần | Điểm | Yêu cầu |
|---|---:|---|
| **Core Flow** | **80** | |
| Schema completeness | 30 | `meta`, `summary`, `failure_modes`, `examples`, `extensions`, `discussion` |
| Experiment completeness | 30 | ReAct + Reflexion, ≥100 records, ≥20 examples |
| Analysis depth | 20 | ≥3 failure modes, discussion ≥250 ký tự |
| **Bonus** | **20** | ≥2 extensions (10đ/extension, tối đa 20đ) |

## Thành phần mã nguồn

| File | Mô tả |
|---|---|
| `src/reflexion_lab/agents.py` | Vòng lặp ReAct + Reflexion |
| `src/reflexion_lab/openai_runtime.py` | LLM calls (GPT-4o-mini) |
| `src/reflexion_lab/mock_runtime.py` | Mock deterministic |
| `src/reflexion_lab/prompts.py` | System prompts Actor/Evaluator/Reflector/Planner |
| `src/reflexion_lab/context_ranking.py` | Xếp hạng context passages |
| `src/reflexion_lab/reporting.py` | Xuất `report.json` + `report.md` |
| `run_benchmark.py` | Benchmark ReAct + Reflexion |
| `run_golden.py` | Chạy Golden Test (Reflexion, golden inference) |
| `streamlit_app.py` | Demo dashboard |
| `scripts/build_hotpot_subset.py` | Tải subset từ HotpotQA (HuggingFace) |
| `scripts/score_golden.py` | Chấm predictions vs gold (local) |
| `scripts/refresh_report.py` | Tái tạo báo cáo kèm golden results |
| `data/hotpot_subset_100.json` | 100 câu validation HotpotQA |
| `hotpot_golden.json` | 20 câu Golden Test Set |
| `outputs/full_run_100/` | Báo cáo benchmark chính (nộp lab) |
| `outputs/golden_run/` | Predictions Golden Test |

## Báo cáo

Báo cáo chính: [`outputs/full_run_100/report.md`](outputs/full_run_100/report.md)

Bao gồm:
- Summary ReAct vs Reflexion
- Reflexion recoveries
- Golden test results (90% EM)
- Failure modes
- Extensions + discussion
