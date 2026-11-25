# TRACE — Reliance- and Calibration-Aware Guardrails for AI Tutors

Code and dataset execution package for the paper *"Beyond Static Guardrails: Reliance- and Calibration-Aware Planning of AI-Tutor Assistance, Evaluated in a Learner Simulator Calibrated to a Field Experiment"*.

This repository provides the simulator, algorithms, evaluation stages, and replication dataset to execute and reproduce all findings:

| Stage | What it does | Generated Output |
|---|---|---|
| `replication` | Re-estimates Bastani et al. (PNAS 2025) Table 1 + overconfidence from public data | `results/real_*.csv` |
| `calibration` | Simulated-moments calibration of the learner simulator (12 moments) + validation | `results/params_logistic.json`, `results/calibration_fit.csv`, `results/validation_*.csv` |
| `bkt` | Calibrates the second (BKT) learner family | `results/params_bkt.json` |
| `main` | 10 policies × 10 test seeds × 825 students × 4 sessions; contrasts, equity, adaptivity | `results/main_*.csv`, `results/equity.csv`, ... |
| `ablation`, `pareto`, `robust`, `ai_accuracy`, `horizon`, `timing` | Further evaluation experiments | `results/*.csv` |
| `figures`, `tables` | Publication figures (PDF/PNG) and LaTeX tables | `figures/`, `tables/` |

## Setup & Execution

### 1. Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running Experiments

To run the complete experimental pipeline end-to-end:

```bash
python run_all.py                                      # ~25-40 min on a laptop
```

To run specific stages:

```bash
python run_all.py --stages replication calibration main figures tables
```

* The calibration dataset is included under `data/GenAICanHarmLearning/`.
* Generated artifacts (calibrated models, evaluation CSVs, figures, and tables) are created automatically under `results/`, `figures/`, and `tables/` during execution.

## Code Structure

```
tracetutor/
  data.py         load trial data, item bank, and empirical targets
  replicate.py    OLS + CR1 clustered SEs; heterogeneity and validation targets
  simulator.py    learner simulator (logistic and BKT dynamics): mastery, reliance, frustration, confidence
  calibrate.py    simulated-moments calibration (differential evolution + Nelder–Mead)
  policies.py     baselines + Belief (particle filter) + TRACE (rollout planner)
  experiments.py  all experiment stages
  figures.py      figure generation (PDF + PNG)
  tables.py       LaTeX tables + macros generation
data/
  GenAICanHarmLearning/  public RCT dataset, survey responses, and regression data
run_all.py        CLI entry point reproducing all stages
```

## Using TRACE with an LLM Tutor

`TRACE.act(request, t)` returns one of six help levels (`none, reflect, hint, steps, worked, answer`); map each to prompt blocks and feed back the correctness of the student's attempt with `TRACE.observe(...)`.
