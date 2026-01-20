"""
Core Metrics Module

This module provides all evaluation metrics used across roster ranking experiments.
All metrics follow consistent interfaces and are optimized for team ranking tasks.
"""

import numpy as np
import pandas as pd
from ..utils.config import get_rank_col, get_record_id

def dcg_at_k(relevances):
    """
    Compute DCG@k for team ranking
    
    Args:
        relevances (array-like): Array of relevance scores in predicted order
        
    Returns:
        float: DCG score
    """
    relevances = np.asfarray(relevances)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(predicted_order, actual_order, k):
    """
    Compute NDCG@k for team ranking with linear gain decrease.
    
    Gain structure: 1st place gets 5, 2nd gets 4, 3rd gets 3, 4th gets 2, 5th gets 1, 6th+ gets 0
    
    Args:
        predicted_order (list): Teams in predicted finishing order
        actual_order (list): Teams in actual finishing order
        k (int): Number of positions to evaluate
        
    Returns:
        float: NDCG@k score (0 to 1)
    """
    # Build relevance dictionary from actual finishing order
    relevance = {}
    for i, team in enumerate(actual_order):
        pos = i + 1  # finishing position (1-indexed)
        # Linear gain decrease up to kth place
        if pos <= k:
            rel = k - pos + 1
        else:
            rel = 0  # kth place and beyond get 0
        relevance[team] = rel
    
    # Get predicted relevances
    pred_rels = [relevance.get(team, 0) for team in predicted_order[:k]]
    # Ideal order: sort actual relevances in descending order
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    
    dcg = dcg_at_k(pred_rels)
    idcg = dcg_at_k(ideal_rels)
    
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(predicted, actual, k):
    """
    Calculate Recall@k for team ranking
    
    Args:
        predicted (list): Teams in predicted order
        actual (list): Teams in actual order
        k (int): Number of positions to evaluate
        
    Returns:
        float: Recall@k score (0 to 1) or NaN if no relevant items
    """
    if k <= 0:
        return 0.0
        
    relevant = set(actual[:k])
    retrieved = set(predicted[:k])
    
    if len(relevant) == 0:
        return np.nan
        
    return len(relevant.intersection(retrieved)) / len(relevant)


def evaluate_race_predictions(race_predictions_df, k_values=[3, 5, 10], level='rider'):
    """
    Evaluate predictions for a single race across multiple k values
    
    Args:
        race_predictions_df (DataFrame): Race data with record ID, rank, and 'pred_score' columns
        k_values (list): List of k values to evaluate
        level (str): Level of ranking ('roster' or 'rider') to determine column names
        
    Returns:
        dict: Dictionary with metrics for each k value
    """
    # Get level-specific configuration
    rank_col = get_rank_col(level)
    record_id = get_record_id(level)
    
    # Sort by prediction scores (descending) and actual rank (ascending)
    predicted_group = race_predictions_df.sort_values('pred_score', ascending=False)
    actual_group = race_predictions_df.sort_values(rank_col, ascending=True)
    
    predicted_order = list(predicted_group[record_id])
    actual_order = list(actual_group[record_id])
    
    results = {}
    
    for k in k_values:
        results[f'NDCG@{k}'] = ndcg_at_k(predicted_order, actual_order, k)
        results[f'Recall@{k}'] = recall_at_k(predicted_order, actual_order, k)
    
    max_k = max(k_values)
    results['predicted_order'] = predicted_order[:max_k]
    results['actual_order'] = actual_order[:max_k]
    
    return results


def calculate_summary_statistics(results_df, metrics=['NDCG@3', 'NDCG@5','NDCG@10', 'Recall@3', 'Recall@5','Recall@10']):
    """
    Calculate comprehensive summary statistics for experiment results
    
    Args:
        results_df (DataFrame): Results with metric columns and riders_number/teams_number for weighting
        metrics (list): List of metric column names
        
    Returns:
        dict: Nested dictionary with statistics for each metric, including normalized variants
    """
    summary = {}
    
    # Check if weighting columns are available
    has_riders_weights = 'riders_number' in results_df.columns
    has_teams_weights = 'teams_number' in results_df.columns
    
    for metric in metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                # Standard statistics
                summary[metric] = {
                    'mean': round(float(values.mean()), 4),
                    'std': round(float(values.std()), 4),
                    'min': round(float(values.min()), 4),
                    'max': round(float(values.max()), 4),
                    'count': int(len(values)),
                }
                
                # Calculate weighted averages if weight columns are available
                non_na_mask = results_df[metric].notna()
                clean_df = results_df[non_na_mask]
                
                if has_riders_weights and len(clean_df) > 0:
                    metric_values = clean_df[metric]
                    weights = clean_df['riders_number']
                    
                    # Calculate weighted average by riders number
                    if weights.sum() > 0:
                        weighted_mean = (metric_values * weights).sum() / weights.sum()
                        summary[f'{metric}_R_Norm'] = {
                            'mean': round(float(weighted_mean), 4),
                            'std': round(float(values.std()), 4),  # Keep original std for reference
                            'min': round(float(values.min()), 4),
                            'max': round(float(values.max()), 4),
                            'count': int(len(values)),
                        }
                    else:
                        summary[f'{metric}_R_Norm'] = {
                            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                        }
                
                if has_teams_weights and len(clean_df) > 0:
                    metric_values = clean_df[metric]
                    weights = clean_df['teams_number']
                    
                    # Calculate weighted average by teams number
                    if weights.sum() > 0:
                        weighted_mean = (metric_values * weights).sum() / weights.sum()
                        summary[f'{metric}_T_Norm'] = {
                            'mean': round(float(weighted_mean), 4),
                            'std': round(float(values.std()), 4),  # Keep original std for reference
                            'min': round(float(values.min()), 4),
                            'max': round(float(values.max()), 4),
                            'count': int(len(values)),
                        }
                    else:
                        summary[f'{metric}_T_Norm'] = {
                            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                        }
                        
            else:
                # Initialize with zeros if no valid values
                summary[metric] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0,
                }
                
                # Also initialize normalized versions
                if has_riders_weights:
                    summary[f'{metric}_R_Norm'] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                    }
                if has_teams_weights:
                    summary[f'{metric}_T_Norm'] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                    }
        else:
            # Initialize with zeros if metric not found
            summary[metric] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'min': 0.0, 'max': 0.0, 'count': 0,
            }
            
            # Also initialize normalized versions
            if has_riders_weights:
                summary[f'{metric}_R_Norm'] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                }
            if has_teams_weights:
                summary[f'{metric}_T_Norm'] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0,
                }
    
    return summary