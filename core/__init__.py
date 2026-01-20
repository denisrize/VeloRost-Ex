"""
Core modeling and evaluation utilities.
"""

from .metrics import (
    dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    evaluate_race_predictions,
    calculate_summary_statistics,
)
from .models import (
    assign_roster_label,
    prepare_data_for_training,
    get_hyperparameter_grid,
    tune_hyperparameters,
    train_model,
    evaluate_model,
    save_model_results,
    get_feature_importance,
)

__all__ = [
    'dcg_at_k',
    'ndcg_at_k',
    'recall_at_k',
    'evaluate_race_predictions',
    'calculate_summary_statistics',
    'assign_roster_label',
    'prepare_data_for_training',
    'get_hyperparameter_grid',
    'tune_hyperparameters',
    'train_model',
    'evaluate_model',
    'save_model_results',
    'get_feature_importance',
]