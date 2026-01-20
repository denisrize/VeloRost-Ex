"""
Core Models Module

This module provides machine learning functionality for roster ranking experiments.
Includes hyperparameter tuning, model training, and prediction logic that can be
reused across all experiment types (base models, ensemble, fusion).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
import time
from .metrics import evaluate_race_predictions
from tqdm import tqdm
from ..utils.config import get_rank_col, get_record_id

def assign_roster_label(group, k_value=5, level='roster'):
    """
    Assign learning-to-rank labels for teams or riders within a race
    
    Args:
        group (DataFrame): Race group with rank column
        k_value (int): Number of top positions to assign positive labels
        level (str): Level of ranking ('roster' or 'rider') to determine rank column
        
    Returns:
        DataFrame: Group with 'label' column added
    """
    # Get the appropriate rank column for this level
    rank_col = get_rank_col(level)
    
    def get_label(rank):
        if rank <= k_value:
            return k_value - rank + 1
        return 0
    
    group = group.copy()
    group['label'] = group[rank_col].apply(get_label)
    return group


def prepare_data_for_training(df, feature_columns, k_value=5, level='roster'):
    """
    Prepare data for XGBoost training with consistent processing
    
    Args:
        df (DataFrame): Input data
        feature_columns (list): List of feature column names
        k_value (int): Number of top positions for ranking
        level (str): Data level ('roster' or 'rider') to determine rank column
        
    Returns:
        tuple: (X, y, groups) for XGBoost training
    """
    # Create race ID and sort consistently
    df_exp = df.copy()
    df_exp['race_id'] = df_exp['race'] + "_" + df_exp['date'].astype(str)
    df_exp = df_exp.sort_values('race_id').reset_index(drop=True)
    
    # Assign labels using level-specific configuration
    df_exp = df_exp.groupby('race_id', group_keys=False).apply(
        assign_roster_label, k_value=k_value, level=level
    )
    
    # Extract features and labels
    X = df_exp[feature_columns].values
    y = df_exp['label'].values
    groups = df_exp.groupby('race_id').size().tolist()
    
    return X, y, groups


def get_hyperparameter_grid():
    """
    Get the hyperparameter grid for XGBoost tuning
    Optimized for LambdaMART ranking with 30-80 features and 550k records
    Uses early stopping instead of fixed num_boost_round
    
    Returns:
        dict: Parameter grid with 972 combinations (3×4×3×3×3×3)
    """
    return {
        # Most critical for ranking performance
        'learning_rate': [0.05, 0.1, 0.2 ],     # 
        'max_depth': [6, 10, 15, 20 ],        # 
        
        # Critical for overfitting prevention with your feature count  
        'reg_alpha': [0, 8, 16 ],    # 
        'reg_lambda': [0, 0.5, 1.0 ],       # 
        # Important for large dataset (550k records)
        'subsample': [0.7, 0.8, 0.9 ],  # 
        'colsample_bytree': [0.6, 0.7, 0.8 ],          
    }


def tune_hyperparameters(train_df, val_df, feature_columns, param_grid=None, random_seed=42, k_value=10, level='rider'):
    """
    Tune hyperparameters using grid search with NDCG validation and early stopping
    
    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data  
        feature_columns (list): Feature column names
        param_grid (dict): Optional custom parameter grid
        random_seed (int): Random seed for reproducibility
        k_value (int): K value for NDCG evaluation
        level (str): Data level ('roster' or 'rider')
        
    Returns:
        tuple: (best_params, best_ndcg)
    """
    if param_grid is None:
        param_grid = get_hyperparameter_grid()
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"Tuning hyperparameters with {total_combinations} parameter combinations using early stopping...")
    print(f"Parameter grid: {param_grid}")
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Prepare training and validation data
    X_train, y_train, train_groups = prepare_data_for_training(train_df, feature_columns, k_value=k_value, level=level)
    X_val, y_val, val_groups = prepare_data_for_training(val_df, feature_columns, k_value=k_value, level=level)
    
    # Build DMatrices with feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dtrain.set_group(train_groups)
    
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    dval.set_group(val_groups)
    
    best_ndcg = 0
    best_params = None
    best_num_rounds = 0
    
    # Early stopping configuration
    max_boost_rounds = 1000  # Set high limit
    early_stopping_rounds = 100  # Stop if no improvement for 50 rounds
    
    # Prepare validation DataFrame for evaluation
    val_df_exp = val_df.copy()
    val_df_exp['race_id'] = val_df_exp['race'] + "_" + val_df_exp['date'].astype(str)
    val_df_exp = val_df_exp.sort_values('race_id').reset_index(drop=True)
    
    # Use level-specific configuration for validation labels
    val_df_exp = val_df_exp.groupby('race_id', group_keys=False).apply(
        assign_roster_label, k_value=k_value, level=level
    )
    
    print(f"Starting hyperparameter search with early stopping (max_rounds={max_boost_rounds}, early_stop={early_stopping_rounds})...")
    
    for i, hyperparams in tqdm(enumerate(all_param_combinations), total=len(all_param_combinations), desc="Tuning hyperparameters"):
        try:
            # Set up XGBoost parameters
            params = hyperparams.copy()
            params['objective'] = 'rank:ndcg'
            params['eval_metric'] = f'ndcg@{k_value}'
            params['seed'] = random_seed  # Ensure reproducibility
            
            # Train model with early stopping
            evals = [(dtrain, 'train'), (dval, 'validation')]
            model = xgb.train(
                params, 
                dtrain, 
                num_boost_round=max_boost_rounds,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False  # Suppress XGBoost output during tuning
            )
            
            # Get the optimal number of rounds from early stopping
            optimal_rounds = model.best_iteration + 1  # XGBoost uses 0-based indexing
            
            # Predict using the early-stopped model
            val_df_exp['pred_score'] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            
            # Calculate NDCG per race for final evaluation
            ndcg_scores = []
            for race_id, group in val_df_exp.groupby('race_id'):
                race_metrics = evaluate_race_predictions(group, k_values=[k_value], level=level)
                ndcg_score = race_metrics[f'NDCG@{k_value}']
                if not np.isnan(ndcg_score):
                    ndcg_scores.append(ndcg_score)
            
            if len(ndcg_scores) == 0:
                print(f"Warning: No valid NDCG scores for combination {i+1}")
                continue
                
            mean_ndcg = np.mean(ndcg_scores)
            
            # Update best parameters if this is better
            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_params = hyperparams.copy()
                best_num_rounds = optimal_rounds
                print(f"New best NDCG@{k_value}: {best_ndcg:.4f} (rounds: {best_num_rounds}, combination {i+1}/{len(all_param_combinations)})")
                
        except Exception as e:
            print(f"Error with combination {i+1}: {e}")
            continue
    
    if best_params is None:
        raise ValueError("Hyperparameter tuning failed - no valid parameter combination found")
    
    # Add the optimal number of rounds to best params for future reference
    best_params['optimal_num_boost_round'] = best_num_rounds
    
    print(f"✓ Hyperparameter tuning completed!")
    print(f"✓ Best NDCG@{k_value}: {best_ndcg:.4f}")
    print(f"✓ Best parameters: {best_params}")
    print(f"✓ Optimal boosting rounds: {best_num_rounds}")
    
    return best_params, best_ndcg


def train_model(train_df, feature_columns, best_params, model_name="model", random_seed=42, k_value=10, level='roster'):
    """
    Train XGBoost model with given hyperparameters
    
    Args:
        train_df (DataFrame): Training data
        feature_columns (list): Feature column names
        best_params (dict): Hyperparameters (including optimal_num_boost_round from tuning)
        model_name (str): Model name for logging
        random_seed (int): Random seed for reproducibility
        k_value (int): K value for ranking evaluation
        level (str): Data level ('roster' or 'rider')
        
    Returns:
        xgb.Booster: Trained XGBoost model
    """
    print(f"Training {model_name}...")
    
    # Prepare training data
    X_train, y_train, train_groups = prepare_data_for_training(train_df, feature_columns, k_value=k_value, level=level)
    
    # Build training DMatrix with explicit feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dtrain.set_group(train_groups)
    
    # Set up parameters
    params = best_params.copy()
    params['objective'] = 'rank:ndcg'
    params['eval_metric'] = f'ndcg@{k_value}'
    params['seed'] = random_seed  # Ensure reproducibility
    
    # Use optimal number of boost rounds found during hyperparameter tuning
    # If not available, fall back to default
    num_boost_round = params.pop('optimal_num_boost_round', 300)
    
    # Remove any other non-XGBoost parameters that might be in best_params
    if 'num_boost_round' in params:
        params.pop('num_boost_round')
    
    print(f"Training with {num_boost_round} boosting rounds (from hyperparameter tuning)")
    
    # Train model
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    
    print(f"✓ {model_name} training completed")
    
    return model


def evaluate_model(model, test_df, feature_columns, model_name="model", k_values=[3, 5, 10], k_value=5, level='roster'):
    """
    Evaluate model on test data and return per-race results
    
    Args:
        model (xgb.Booster): Trained XGBoost model
        test_df (DataFrame): Test data
        feature_columns (list): Feature column names
        model_name (str): Model name for logging
        k_values (list): List of k values for evaluation
        k_value (int): K value used for training (for label assignment)
        level (str): Data level ('roster' or 'rider')
        
    Returns:
        DataFrame: Per-race evaluation results
    """
    print(f"Evaluating {model_name}...")
    
    # Prepare test data
    test_df_exp = test_df.copy()
    test_df_exp['race_id'] = test_df_exp['race'] + "_" + test_df_exp['date'].astype(str)
    test_df_exp = test_df_exp.sort_values('race_id').reset_index(drop=True)
    
    # Use level-specific configuration for test labels
    test_df_exp = test_df_exp.groupby('race_id', group_keys=False).apply(
        assign_roster_label, k_value=k_value, level=level
    )
    
    # Build test DMatrix with feature names
    X_test = test_df_exp[feature_columns].values
    dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
    
    # Make predictions
    test_df_exp['pred_score'] = model.predict(dtest)
    
    # Evaluate per race using level-specific record ID
    record_id = get_record_id(level)
    results = []
    for race_id, group in test_df_exp.groupby('race_id'):
        # Get race information
        race_info = group.iloc[0]
        
        # Calculate metrics for this race using level-specific configuration
        race_metrics = evaluate_race_predictions(group, k_values=k_values, level=level)
        
        # Compile results
        result = {
            'race': race_info['race'],
            'date': race_info['date'],
            'cluster': race_info['cluster'],
            'race_class': race_info['race_class'],
            'riders_number': len(group),
            'teams_number': len(group['team'].unique())
        }
        result.update(race_metrics)
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    print(f"✓ {model_name} evaluation completed: {len(results_df)} races")
    
    return results_df


def save_model_results(results_df, model_name, race_class, output_dir, summary_stats=None, feature_importance=None):
    """
    Save model results to CSV with comprehensive summary
    
    Args:
        results_df (DataFrame): Per-race results
        model_name (str): Name of the model
        race_class (str): Race class ('all' or 'WT')
        output_dir (str): Output directory path
        summary_stats (dict): Optional pre-calculated summary statistics
        feature_importance (DataFrame): Optional feature importance DataFrame
    """
    import os
    from .metrics import calculate_summary_statistics
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = f"{output_dir}/{model_name}_results.csv"
    results_df.to_csv(results_file, index=False)
    
    # Calculate summary if not provided
    if summary_stats is None:
        summary_stats = calculate_summary_statistics(results_df)
    
    # Save summary
    summary_file = f"{output_dir}/{model_name}_summary.csv"
    summary_data = []
    
    for metric, stats in summary_stats.items():
        if isinstance(stats, dict):
            row = {'Metric': metric}
            row.update(stats)
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    # Save feature importance if provided
    if feature_importance is not None and len(feature_importance) > 0:
        importance_file = f"{output_dir}/{model_name}_feature_importance.csv"
        feature_importance.to_csv(importance_file, index=False)
        print(f"✓ Feature importance saved: {importance_file}")
    
    print(f"✓ Results saved:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")


def get_feature_importance(model, importance_type='gain'):
    """
    Extract feature importance from trained XGBoost model
    
    Args:
        model (xgb.Booster): Trained XGBoost model
        importance_type (str): Type of importance ('gain', 'weight', 'cover')
        
    Returns:
        DataFrame: Feature importance sorted by importance
    """
    # Get importance scores from model (now uses real feature names)
    importance_scores = model.get_score(importance_type=importance_type)
    
    # Convert to DataFrame directly since we now have real feature names
    importance_data = []
    
    for feature_name, importance in importance_scores.items():
        importance_data.append({
            'feature': feature_name,
            'importance': importance
        })
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df 