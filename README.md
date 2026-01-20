# VeloRost

This repository implements my thesis research on ranking riders using the VeloRost framework I developed and implemented. By combining their individual leader skills with their teammates' helper skills, VeloRost creates a comprehensive pipeline for ranking cyclists in professional road cycling races. It focuses on learning-to-rank models built from TrueSkill leader and helper skills, roster aggregations, and race context features.

The preliminary results and method details are published in:  
["A Bayesian Dual-Skill Framework for Roster-Based Cycling Race Outcome Prediction" (ISACE 2025)](https://link.springer.com/chapter/10.1007/978-3-032-06167-6_15).  
The full journal paper is currently under peer review—stay tuned for final results.

## Table of Contents

- [Dataset](#dataset)
- [Research Summary and Results](#research-summary-and-results)
  - [Statistically Enhanced Learning (SEL) Results](#statistically-enhanced-learning-sel-results)
  - [Helper Marginal Contribution](#helper-marginal-contribution)
  - [Feature Importance Analysis](#feature-importance-analysis)
- [Project Layout](#project-layout)
- [Pipelines Overview](#pipelines-overview)
  - [1. Data Filtering](#1-data-filtering-pre-feature-extraction)
  - [2. Rider Features Extraction](#2-rider-features-extraction)
  - [3. TrueSkill Feature Extraction](#3-trueskill-feature-extraction)
  - [4. Base Models](#4-base-models)
  - [5. Ensemble Models](#5-ensemble-models)
  - [6. Direct Ranking Analysis](#6-direct-ranking-analysis)
  - [7. Feature Importance](#7-feature-importance)
- [Experiment Functionality by Module](#experiment-functionality-by-module)
- [Configuration](#configuration)
- [Typical Workflow](#typical-workflow)
- [Citing the Preliminary Paper](#citing-the-preliminary-paper)

## Dataset

The complete raw race results dataset (2017-2023) used in these experiments is publicly available at:  
**[VeloRost-Ex-Data Repository](https://github.com/denisrize/VeloRost-Ex-Data)**  
DOI: [10.5281/zenodo.17225235](https://doi.org/10.5281/zenodo.17225235)

The dataset includes over 680,000 rider-race records across 5,600 unique race days, covering all UCI competition tiers (WorldTour, ProSeries, Class 1, Class 2) and seven race profiles.

## Research summary and results

### Statistically Enhanced Learning (SEL) results

Results across different prediction horizons (decision points before race):

#### NDCG@10 Performance by Time Gap

| Method | 1 day | 30 days | 90 days | Pre-season |
| --- | --- | --- | --- | --- |
| Leader Only (Baseline) | 0.431 | 0.394 | 0.382 | 0.365 |
| Leader + Enhanced Features | 0.450 | 0.404 | 0.394 | 0.375 |
| Uniform-All | 0.458 | 0.421 | 0.407 | 0.395 |
| Positional | 0.459 | 0.419 | 0.406 | 0.392 |
| Time-Lag | 0.460 | 0.419 | 0.402 | 0.391 |
| **Ensemble** | **0.464** | **0.423** | **0.408** | **0.396** |

**Key Findings:**
- Ensemble achieves **7.7% improvement** over Leader Only baseline at 1-day horizon
- Helper-based methods show **better temporal stability** (13.8-14.9% degradation) vs. baseline (16.6% degradation) from 1-day to pre-season
- All helper methods significantly outperform both baselines across all horizons (p < 0.05, Holm-Bonferroni corrected)

#### Recall@10 Performance by Time Gap

| Method | 1 day | 30 days | 90 days | Pre-season |
| --- | --- | --- | --- | --- |
| Leader Only (Baseline) | 0.396 | 0.361 | 0.353 | 0.337 |
| Leader + Enhanced Features | 0.408 | 0.367 | 0.360 | 0.344 |
| Uniform-All | 0.413 | 0.381 | 0.369 | 0.360 |
| Positional | 0.414 | 0.380 | 0.365 | 0.357 |
| Time-Lag | 0.415 | 0.380 | 0.365 | 0.354 |
| **Ensemble** | **0.419** | **0.383** | **0.368** | **0.360** |

### Helper marginal contribution

Analysis of performance gains from sequentially adding helpers (ordered by skill level):

| Number of Helpers | NDCG@10 Improvement (%) | Recall@10 Improvement (%) |
| --- | --- | --- |
| 0 (Leader Only) | 0.00 | 0.00 |
| 1 helper | +2.0 | -0.55 |
| 2 helpers | +3.2 | +0.67 |
| 3 helpers | +3.6 | +1.20 |
| 4 helpers | ~3.8 | +1.46 |
| 5 helpers | +3.93 (peak) | +1.60 |
| 6+ helpers | ~3.9 (plateau) | +1.66 (peak at 6) |

**Key Insights:**
- **First 2-3 helpers** deliver most of the improvement
- **Diminishing returns** observed beyond 3 helpers
- NDCG@10 peaks at 5 helpers (+3.93%), Recall@10 peaks at 6 helpers (+1.66%)
- Uniform-All and Time-Lag methods remain stable even with 7-8 helpers, demonstrating robustness for professional racing

### Feature Importance Analysis

SHAP-based analysis reveals how predictive signals shift across decision horizons:

#### Feature Group Importance by Prediction Horizon

| Feature Group | 1 Day Before | Pre-Season | Change Pattern |
| --- | --- | --- | --- |
| **Recent Form** | ~35-40% | 0% | Dominant short-term, vanishes at pre-season |
| **Leader Capability** | ~15% | ~26% | **+75% increase** - most efficient at pre-season |
| **Roster Depth & Support** | ~12% | ~20% | **+66% increase** - grows for long-term planning |
| **Performance Distribution** | ~20% | ~11% | ~46% decline - cross-season variability weakens signal |
| **Career Trajectory** | ~22% | ~27% | Stable - slow-moving indicators (age, experience) |
| **Race Context** | ~7% | ~10% | Stable background condition |

#### Top Features by Horizon

**1-Day Before Race (Short-term):**
1. Recent points gained (42-day window)
2. Season-start points indicators
3. Best result & days since top-3 by profile/tier
4. Leader µ (Profile) - TrueSkill mean
5. Peak performance markers with recency

**Pre-Season (Long-term planning):**
1. **Leader µ (Profile)** - profile-specific TrueSkill mean
2. **Roster Avg. µ (GC)** - average helper GC skill
3. **Roster Avg. µ (Profile)** - average helper profile skill
4. Previous season points accumulation
5. Age & Career Length (negative correlation at extremes)

**Key Finding:** Leader µ (GC) shows **inverse correlation** with stage wins - GC riders focus on limiting time losses rather than stage victories, making GC helper skills particularly valuable for versatile roster support in pre-season planning.


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