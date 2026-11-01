# Judge responses

One file per model, 640 rows each: 320 items × 2 protocols. Every row the runner recorded for these three models is here, verbatim, including failed API calls.

| File | Model | Usable (V / E) | API errors (V / E) | Malformed (V / E) |
|---|---|---|---|---|
| `results_g31lite.jsonl` | `gemini-3.1-flash-lite` | 187 / 187 | 133 / 133 | 0 / 0 |
| `results_g35lite.jsonl` | `gemini-3.5-flash-lite` | 250 / 250 | 70 / 69 | 0 / 1 |
| `results_gemma31b.jsonl` | `gemma-4-31b-it` | 262 / 252 | 58 / 68 | 0 / 0 |

The API errors are 406 quota responses (HTTP 429), 60 server errors (500), 33 overloaded responses (503) and 32 timeouts. They are kept so that coverage can be audited, and they are excluded from scoring.

## Fields

| Field | Meaning |
|---|---|
| `item_id` | Item in `data/spendaudit.jsonl` |
| `model` | Model identifier as sent to the API |
| `condition` | `verdict_only` or `evidence` |
| `raw` | The model's output text, unmodified |
| `parsed` | The JSON the runner parsed from `raw`, if any |
| `error` | Present only on failed calls: the HTTP status and message, or the timeout |
| `usage` | Token usage reported by the API |
| `latency_s` | Wall-clock seconds for the call |
| `finish_reason` | Gemma rerun only |

## Provenance

The runs took place on October 12–13, 2026. The first 14 rows for Gemini 3.1 Flash-Lite and the first 17 for Gemini 3.5 Flash-Lite were written to a shared results file before the per-model files existed; they are placed at the top of each model's file here, with every line copied unchanged. Smoke tests of other models and Gemma's earlier run with a 1,800-token budget, which was replaced after responses were truncated, are not included.
