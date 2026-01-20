"""
Data Loaders Module

This module provides centralized data loading functionality for roster ranking experiments.
It handles loading, caching, and preprocessing of datasets for different schemes and race classes.
"""

import os
import pandas as pd
from ..utils.config import *
from ..data.features import *
import numpy as np
from tqdm import tqdm
import sys
import gc          # Python’s garbage-collector interface
from ..feature_extraction.extract_trueSkill_features import compute_team_perf_simplified

def map_class(race_class):
    if 'WT' in race_class:
        return 'WT'
    elif 'Pro' in race_class:
        return 'Pro'
    elif '.1' in race_class:
        return '1'
    else:
        return '2'
    
def load_and_merge_features(race_class, scheme, time_gap=None):
    """
    Common pipeline for loading and merging race data, rider features, and TrueSkill features
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm', 'baseline', 'leader')
        time_gap (int): Time gap for time-lagged features (default: None)
    Returns:
        tuple: (combined_features DataFrame, race_feature_cols list)
    """
    # Load race results data
    race_results_path = get_data_dir('riders_race_results')
    race_results = pd.read_csv(race_results_path)
    
    # Load rider features using configured path
    rider_features_path = get_rider_features_path(race_class, time_gap=time_gap)
    riders_features = pd.read_csv(rider_features_path)

    # Prepare data
    race_results['date'] = pd.to_datetime(race_results['date'])
    riders_features['date'] = pd.to_datetime(riders_features['date'])
    race_results['year'] = race_results['date'].dt.year
    
    # Filter by race class if WT
    if race_class == 'WT':
        race_results = race_results[(race_results['classification'] == '1.UWT') | (race_results['classification'] == '2.UWT')]

    # Merge race results and rider features
    merge_on = ['race', 'date', 'rider', 'cluster','classification','rank_number']

    all_rider_features = pd.merge(
        race_results[merge_on+['rider_race_points']], 
        riders_features, 
        on=merge_on, 
        how='inner'
    )
    # Conditionally load and merge TrueSkill features based on scheme
    leader_power_path = get_leader_power_path(race_class, time_gap=time_gap)
    true_skill_features_leader = pd.read_csv(leader_power_path)
    true_skill_features_leader['date'] = pd.to_datetime(true_skill_features_leader['date'])
    true_skill_features_leader['classification'] = true_skill_features_leader['classification'].astype(str)

    if scheme != 'leader':
        team_power_path = get_team_power_path(race_class, scheme, time_gap=time_gap)
        true_skill_features_teams = pd.read_csv(team_power_path)
        true_skill_features_teams['date'] = pd.to_datetime(true_skill_features_teams['date'])
        true_skill_features_teams['classification'] = true_skill_features_teams['classification'].astype(str)
    
        print(f'Merging features of leader {true_skill_features_leader.shape} and teammates {true_skill_features_teams.shape}')
        true_skill_features = pd.merge(
            true_skill_features_teams[merge_on + ROSTER_TRUE_SKILL_K6_TEAMMATE_BOTH + RAW_TRUE_SKILL_TEAMMATE],     
            true_skill_features_leader[merge_on+ALL_RAW_LEADER_FEATURES+['team_rank']], 
            on=merge_on, how='inner'
        )
        print(f'Merged features shape: {true_skill_features.shape}')
    else:
        true_skill_features = true_skill_features_leader[merge_on+ALL_RAW_LEADER_FEATURES+['team_rank']].copy()

    # Prepare data
    true_skill_features['date'] = pd.to_datetime(true_skill_features['date'])

    # Merge all features
    print("Merging features...", true_skill_features.shape, all_rider_features.shape)
    combined_features = pd.merge(
        all_rider_features, 
        true_skill_features, 
        on=merge_on, 
        how='inner'
    )

    print("Combined features shape:", combined_features.shape)

    # Add race features
    combined_features['race_class'] = combined_features['classification'].apply(map_class)
    combined_features, race_feature_cols = extract_race_features(combined_features)
    combined_features['year'] = combined_features['date'].dt.year

    print(f"Final combined features shape: {combined_features.shape}")
    return combined_features 

def load_or_create_dataset(race_class, scheme, level='rider', time_gap=None, exp_name='class_features'):
    """
    Load or create dataset for a specific race class and scheme at the specified level
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm', 'baseline', 'leader')
        level (str): Level of ranking ('roster' or 'rider')
        
    Returns:
        DataFrame: Dataset with features and rankings at the specified level
    """
    if level == 'roster':
        return load_or_create_roster_dataset(race_class, scheme, time_gap=time_gap, exp_name=exp_name)
    elif level == 'rider':
        return load_or_create_rider_dataset(race_class, scheme, time_gap=time_gap, exp_name=exp_name)
    else:
        raise ValueError(f"Unsupported level: {level}. Use 'roster' or 'rider'.")

def load_or_create_roster_dataset(race_class, scheme, time_gap=None, exp_name='class_features'):
    """
    Load or create roster dataset for a specific race class and scheme
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm', 'baseline', 'leader')
        time_gap (int): Time gap for time-lagged features (default: None)
    Returns:
        DataFrame: Roster-level dataset with features and team rankings
    """
    # Get configured dataset path
    dataset_path = get_roster_dataset_path(race_class, scheme)
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        print(f"Loading existing roster dataset from {dataset_path}")
        roster_df = pd.read_csv(dataset_path)
        roster_df['date'] = pd.to_datetime(roster_df['date'])
        return roster_df
    
    print(f"Roster dataset not found. Creating new dataset...")

    # Use common pipeline for loading and merging features
    combined_features, race_feature_cols = load_and_merge_features(race_class, scheme, time_gap=time_gap)
    
    # Create roster dataset with TrueSkill features
    print("Creating roster-level dataset...")
    roster_df = create_roster_dataset(combined_features, race_feature_cols, leader_only=scheme == 'leader', include_skill_features=scheme != 'baseline')
   
    # Save the dataset for future use
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    roster_df.to_csv(dataset_path, index=False)
    print(f"Roster dataset saved to {dataset_path}")
    
    return roster_df

def create_roster_dataset(combined_features, race_feature_cols, leader_only=False, include_skill_features=True):
    """
    Convert rider-level data to roster-level data with team rankings based on total points
    """
    roster_data = []
    print(f'Creating roster dataset with skill features: {include_skill_features} and leader only: {leader_only}')
    # Group by race to calculate team rankings
    for (race, date), race_group in tqdm(combined_features.groupby(['race', 'date']), 
                                        desc='Processing races', file=sys.stdout, 
                                        dynamic_ncols=True):
        # Calculate team total points and best rank for ranking
        team_stats = race_group.groupby('team').agg({
            'rider_race_points': 'sum',        # Sum of all rider points (higher is better)
            'rank_number': 'min',   # Best rank achieved by any rider in team (lower is better) 
            'rider': 'count'        # Team size
        }).reset_index()
        
        # Rank teams by total points (higher is better), tie-break by best rank (lower is better)
        team_stats = team_stats.sort_values(['rider_race_points', 'rank_number'], ascending=[False, True])
        team_stats['team_rank'] = range(1, len(team_stats) + 1)
        race_class = race_group['classification'].iloc[0]
        race_cluster = race_group['cluster'].iloc[0]
        race_year = race_group['year'].iloc[0]
        # Create features for each team
        for team, team_riders in race_group.groupby('team'):
            # Get team stats
            team_info = team_stats[team_stats['team'] == team].iloc[0]
            team_rank = team_info['team_rank']
            total_points = team_info['rider_race_points']
            best_rank = team_info['rank_number'].min()
            
            # Create roster features
            roster_features = create_comprehensive_roster_features(team_roster_df=team_riders, leader_only=leader_only, include_skill_features=include_skill_features)
            
            # Add race context
            roster_features.update({
                'race': race,
                'date': date,
                'team': team,
                'team_rank': team_rank,
                'total_points': total_points,
                'best_rank': best_rank,
                'cluster': race_cluster,
                'race_class': race_class,
                'year': race_year
            })
            
            # Add all race features
            race_info = team_riders.iloc[0]
            for col in race_feature_cols:
                if col in race_info:
                    roster_features[col] = race_info[col]
            
            roster_data.append(roster_features)
    
    return pd.DataFrame(roster_data)

def load_or_create_rider_dataset(race_class, scheme, exp_name='class_features',time_gap=None):
    """
    Load or create rider dataset for a specific race class and scheme
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm', 'baseline', 'leader')
        time_gap (int): Time gap for time-lagged features (default: None)
    Returns:
        DataFrame: Rider-level dataset with features and individual rider rankings
    """
    
    print(f"Rider dataset not found. Creating new dataset...")

    # Use common pipeline for loading and merging features
    combined_features = load_and_merge_features(race_class, scheme, time_gap=time_gap)

    # features = RIDER_BASELINE_FEATURES + RACE_BASELINE_FEATURES + RAW_TRUE_SKILL_LEADER_FEATURES
    features = get_features_by_experiment(exp_name)
    rider_df = combined_features[features+ID_COLS].copy()
    
    return rider_df