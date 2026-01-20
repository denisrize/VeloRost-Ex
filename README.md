# Roster Ranker

This repository implements the roster-ranking research pipeline for professional road cycling. It focuses on learning-to-rank models built from TrueSkill leader and helper skills, roster aggregations, and race context features.

The preliminary results and method details are published in:  
["A Bayesian Dual-Skill Framework for Roster-Based Cycling Race Outcome Prediction" (ISACE 2025)](https://link.springer.com/chapter/10.1007/978-3-032-06167-6_15).  
The full journal paper is currently under peer review—stay tuned for final results.

## Research summary and results

### Statistically Enhanced Learning (SEL) results
Replace the placeholders below with your final numbers.

| Experiment | Metric | Value |
| --- | --- | --- |
| SEL (overall) | NDCG@10 | TBD |
| SEL (overall) | Recall@10 | TBD |
| SEL (by race class) | NDCG@10 | TBD |
| SEL (by race class) | Recall@10 | TBD |

### Helper marginal contribution
Use this table to summarize the marginal impact of helpers by configuration or race context.

| Setting | Baseline | With helpers | Delta |
| --- | --- | --- | --- |
| Example setting | TBD | TBD | TBD |

### Feature importance highlights
Summarize key features from permutation/LOFO/SHAP (top features or categories).

| Method | Top features / categories | Notes |
| --- | --- | --- |
| Permutation | TBD | TBD |
| LOFO | TBD | TBD |
| SHAP | TBD | TBD |

## Project layout

```
roster_ranker/
├── core/                   # Metrics + model training helpers
├── data/                   # Data loading + filtering pipeline
├── experiments/            # Base, ensemble, direct ranking, feature importance
├── feature_extraction/     # Rider + TrueSkill feature extraction
├── utils/                  # Config and shared utilities
├── run_experiments.py      # Main CLI entrypoint
└── README.md
```

Note: the roster simulation engine has been separated into its own package (`simulation_pkg/`).

## Pipelines overview

All pipelines are launched via:

```bash
python roster_ranker/run_experiments.py --race_class <all|WT> --pipeline <pipeline_name> [options]
```

### 1. Data filtering (pre-feature extraction)
Uses `MIN_TEAM_SIZE` and `MIN_TEAMS_PER_RACE` from `roster_ranker/utils/config.py` to clean the race results dataset.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline filter_race_results
```

### 2. Rider features extraction
Builds rider-level features (historical results, points, time-lag features).

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline extract_rider_features --time_gap 30
```

### 3. TrueSkill feature extraction
Generates leader and teammate TrueSkill features per scheme.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline extract_trueskill_features --scheme time_lag
```

### 4. Base models
Trains independent ranking models per scheme.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline base_only --k_value 10
```

### 5. Ensemble models
Combines base model predictions using simple averaging or meta-learning.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods simple_average
```

### 6. Direct ranking analysis
Evaluates direct ranking and marginal teammate contribution.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline direct_ranking --direct_ranking_mode evaluation
python roster_ranker/run_experiments.py --race_class all --pipeline direct_ranking --direct_ranking_mode marginal --k_penalty 6 --lambda_value 0.35
```

### 7. Feature importance
Runs permutation/LOFO/SHAP analysis across schemes.

```bash
python roster_ranker/run_experiments.py --race_class all --pipeline feature_importance --importance_methods permutation shap
```

## Experiment functionality by module

### `experiments/base_models.py`
- Trains ranking models per scheme (time_lag, equal_weight, rank_norm, leader).
- Handles hyperparameter tuning and evaluation.

### `experiments/ensemble.py`
- Trains ensemble models using base-model outputs.
- Supports simple averaging and meta-learning.

### `experiments/direct_ranking.py`
- Direct ranking evaluation based on TrueSkill-based scores.
- Marginal teammate contribution analysis (lambda, k-penalty, max teammates).

### `experiments/feature_importance.py`
- Permutation, LOFO, and SHAP feature importance.

## Configuration

Centralized in `roster_ranker/utils/config.py`. Key entries:

- `AVAILABLE_SCHEMES` and `CLUSTERS`
- `MIN_TEAM_SIZE`, `MIN_TEAMS_PER_RACE`
- Dataset directories and outputs (including filtered race results path)
- Default grids for direct ranking (`DIRECT_RANKING_K_GRID`, `DIRECT_RANKING_LAMBDA_GRID`)

## Typical workflow

1. Filter raw race results  
2. Extract rider features  
3. Extract TrueSkill features  
4. Train base models  
5. Run ensembles / direct ranking / feature importance  

## Citing the preliminary paper

If you reference this work, cite:  
["A Bayesian Dual-Skill Framework for Roster-Based Cycling Race Outcome Prediction" (ISACE 2025)](https://link.springer.com/chapter/10.1007/978-3-032-06167-6_15)