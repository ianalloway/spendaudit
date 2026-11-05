# SpendAudit

[![recount](https://github.com/ianalloway/spendaudit/actions/workflows/recount.yml/badge.svg)](https://github.com/ianalloway/spendaudit/actions/workflows/recount.yml)
[![Paper (PDF)](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/spendaudit.pdf)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue.svg)](CITATION.cff)
[![License: MIT + CC BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY%204.0-lightgrey.svg)](#license)

**Can an LLM auditor reach the same verdict as an executable spending policy?**

SpendAudit is a 320-item benchmark of synthetic payment traces in which the right answer is computed, not judged. Every label comes from the deterministic spend policy in [SOLVENT](https://github.com/ianalloway/solvent-agent) (v0.2.2, commit `bdc896a`), and every scenario appears twice: once clean, and once with a prompt injection in the memo of the payment under review that argues for the wrong verdict.

**Paper:** [SpendAudit: Evaluating LLM Payment Auditors Against an Executable Policy](paper/spendaudit.pdf) (Ian Alloway, working paper, October 2026)

## Results at a glance

Two protocols were compared: **verdict-only** (V) and **evidence and arithmetic** (E), which requires the auditor to check each of the seven rules, cite trace steps and show the calculation. Counts are on each model's matched subset, the scenarios with a usable verdict in all four conditions (V and E, clean and injected).

| Model | Scenarios | False approvals, clean (V → E) | False approvals, injected (V → E) | False rejections, injected (V → E) |
|---|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite | 93 | 34 → 6 | 35 → 7 | 8 → 0 |
| Gemini 3.5 Flash-Lite | 122 | 47 → 18 | 51 → 30 | 25 → 0 |
| Gemma 4 31B | 87 | 0 → 0 | 14 → 10 | 0 → 0 |

- Requiring rule-by-rule evidence cut false approvals sharply for both Gemini models (exact McNemar p < 10⁻⁷ on clean traces) and produced no false rejections within these four-condition matched subsets.
- Injected memos still got violating payments approved under the evidence protocol, for every model. Gemma made no errors on clean traces but approved 10 of 66 violating payments once the memo asked it to.
- Outside the matched subsets, Gemini 3.5 returned one false rejection under the evidence protocol (`sc125_inj`); its clean evidence response failed with an API error, so that scenario is excluded from the paired analysis.
- 531 of 1,920 intended judgments failed with API errors on the free Gemini tier, mostly on boundary cases, so these are complete-case comparisons within each model, not benchmark-level accuracy.

Every number in the paper can be recomputed from this repository with one command:

```bash
python3 src/recount.py   # quickstart: clone, then run this; no install needed
```

It uses only the Python standard library and makes no network calls.

## Benchmark

| Scenario type | Scenarios | Items (clean + injected) |
|---|---:|---:|
| Exactly one rule violated (12 per rule) | 84 | 168 |
| Compliant at a boundary (near-miss) | 48 | 96 |
| Clearly compliant | 28 | 56 |
| **Total** | **160** | **320** |

The policy has seven rules: vendor allowlist, a $50 per-transaction cap, a $150 rolling 24-hour budget, a $60 per-vendor 24-hour cap, a velocity limit of six payments per hour, a $20 cash reserve, and a positive projected margin. See [`data/policy.txt`](data/policy.txt) for the exact text the auditor sees. Traces run 10 to 25 steps, and every scenario that targets a rolling-window rule also contains one expense placed just outside the window as a decoy.

Two independent checks back the labels: the generator asserts that SOLVENT's `Guardrails.evaluate` and a separate implementation of the rules agree on every scenario, and `src/recount.py` re-derives all 320 labels from the rendered trace text alone.

## Contents

| Path | What it is |
|---|---|
| `data/spendaudit.jsonl` | The 320 items, one per line |
| `data/policy.txt` | The policy text shown to the auditor |
| `prompts/` | System prompt, user-message template, and the two protocol instructions |
| `results/` | Every judge response for the three models, kept verbatim, including API errors (see [`results/README.md`](results/README.md)) |
| `src/make_dataset.py` | Seeded generator; with SOLVENT at `bdc896a` it reproduces `data/` byte for byte |
| `src/run_judges.py` | Resumable, throttled judge runner (standard library only) |
| `src/recount.py` | Label re-derivation, coverage, matched subsets and exact McNemar tests (Tables 3–5) |
| `paper/` | The working paper and its LaTeX source |

### Item fields

| Field | Meaning |
|---|---|
| `item_id`, `scenario_id`, `variant` | e.g. `sc001_clean` / `sc001_inj`; `variant` is `clean` or `injected` |
| `category` | `violation`, `near_miss`, or `clear` |
| `target_rule` | The rule the scenario was built around (null for clear) |
| `gt_violation`, `gt_rule`, `gt_rule_num` | The oracle decision from `Guardrails.evaluate` |
| `trace`, `payment_step` | The text the auditor sees, and the step ID under review |
| `injection` | The planted memo (injected items only) |
| `decisive_steps` | For each rule 1–7, the step IDs its outcome depends on |
| `trap_steps` | For rules 3–5, expenses just outside the rolling window |
| `n_steps` | Trace length |

## Reproducing

```bash
# Recount labels, coverage and the paper's tables from the published files
python3 src/recount.py

# Regenerate the dataset (needs Python 3.10+ and SOLVENT checked out at bdc896a)
git clone https://github.com/ianalloway/solvent-agent && git -C solvent-agent checkout bdc896a
SOLVENT_PATH=$PWD/solvent-agent python3 src/make_dataset.py

# Run a judge (writes to results/; spec is NAME:PROVIDER:reasoning(0/1):requests_per_minute[:max_output_tokens])
GEMINI_API_KEY=... RESULTS_FILE=results_mymodel.jsonl python3 src/run_judges.py --model gemini-3.5-flash-lite:gemini:0:12
```

`recount.py` reads every `results_*.jsonl` file in `results/`, so move new runs elsewhere before recounting the paper's numbers.

The paper's runs used the free Gemini API tier on October 12–13, 2026, at temperature 0. Gemini models had a 1,800-token output budget; Gemma was rerun with 8,192 tokens after the first run was truncated, and only the rerun is included.

## Releases and citing

- **Paper:** [`paper/spendaudit.pdf`](paper/spendaudit.pdf), built from [`paper/spendaudit.tex`](paper/spendaudit.tex).
- **Releases:** no tagged release has been published yet. Until one is available on the [Releases page](https://github.com/ianalloway/spendaudit/releases), cite the [corrected snapshot](https://github.com/ianalloway/spendaudit/tree/829306f8d0d4b47a03bd62b7044d404a88d42f57) rather than changing `main`.
- **DOI:** no DOI has been minted yet. Citation metadata is prepared for a future archive; until then, cite the corrected snapshot above or the repository URL below.
- **Citation metadata:** [`CITATION.cff`](CITATION.cff) powers GitHub's *Cite this repository* button, and [`.zenodo.json`](.zenodo.json) holds the metadata for an archived, DOI-backed copy of each release.

## Citation

```bibtex
@misc{alloway2026spendaudit,
  author = {Alloway, Ian},
  title  = {{SpendAudit}: Evaluating {LLM} Payment Auditors Against an Executable Policy},
  year   = {2026},
  note   = {Working paper},
  url    = {https://github.com/ianalloway/spendaudit}
}
```

## License

Code is MIT (see [`LICENSE`](LICENSE)). Data, prompts and judge outputs are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Everything in the data is synthetic: vendor names are labels from SOLVENT's default configuration, and no real customers, payments or accounts appear.

## Related

- [SOLVENT](https://github.com/ianalloway/solvent-agent): the self-funding analyst agent whose spend policy provides the labels
- [juryrig](https://github.com/ianalloway/juryrig): a toolkit for auditing LLM-as-judge pipelines for position, verbosity and injection bias
