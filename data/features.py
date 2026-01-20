import pandas as pd
import numpy as np

def create_comprehensive_roster_features(team_roster_df, k_values=[3, 7], leader_only=False, teammate_only=False, include_skill_features=True, exclude_current_rider=None):
    """
    Create comprehensive roster-level features combining skill and rider features
    """
    all_features = {}
    
    # Add skill-based features only if requested (skip for baseline scheme)
    if include_skill_features:
        all_features.update(create_roster_skill_features(team_roster_df, k_values, leader_only, teammate_only, exclude_current_rider))
        # all_features.update(create_paired_roster_skill_features(team_roster_df=team_roster_df, k_values=k_values, leader_only=leader_only))
    # Add rider-based features
    all_features.update(create_roster_rider_features(team_roster_df))
    
    return all_features

def create_roster_rider_features(team_roster_df):
    """
    Aggregate individual rider features into roster-level features
    """
    features = {}
    
    # Filter to existing columns
    existing_numeric = ['form', 'best_result_cluster', 'time_since_best_result_cluster', 
    'PointsPastSeasons', 'pointsMinus1', 'pointsMinus2', 'pointsMinus3', 'CareerSlope',
    'best_result_class','time_since_best_result_class','best_result_cluster_class', 'time_since_best_result_cluster_class',
    'PointsStartYear', 'PointsStartYearRaceType', 'PointsPastSeasonsRaceType', 
    'PointsStartYearRaceClass', 'PointsPastSeasonsRaceClass', 'Result 2 Races Ago Cluster', 
    'Result 3 Races Ago Cluster', 'Result 4 Races Ago Cluster', 'Result 5 Races Ago Cluster']
    
    for feature in existing_numeric:
        if feature in team_roster_df.columns and team_roster_df[feature].notna().sum() > 0:  # Only if we have non-null values
            feature_values = team_roster_df[feature].dropna()
            
            # Basic statistics
            features[f'roster_{feature}_max'] = round(feature_values.max(), 2)
            features[f'roster_{feature}_min'] = round(feature_values.min(), 2)
            features[f'roster_{feature}_median'] = round(feature_values.median(), 2)
            features[f'roster_{feature}_75th'] = round(feature_values.quantile(0.75), 2)
            features[f'roster_{feature}_25th'] = round(feature_values.quantile(0.25), 2)
            
    # Team composition features
    features['roster_size'] = len(team_roster_df)
    features['roster_avg_experience'] = team_roster_df['CareerLength'].mean() if 'CareerLength' in team_roster_df.columns else 0
    features['roster_veteran_count'] = (team_roster_df['age'] > 30).sum() if 'age' in team_roster_df.columns else 0
    features['roster_young_count'] = (team_roster_df['age'] < 25).sum() if 'age' in team_roster_df.columns else 0
    
    return features

def create_roster_skill_features(team_roster_df, k_values=[3, 7], leader_only=False, teammate_only=False, exclude_current_rider=None):
    """
    Create roster-level features from both race_cluster and General_Classification TrueSkill ratings
    
    Features created:
    - roster_{skill_type}_{source}_k{k}_{stat}: Conservative skill aggregations for each skill source
      where source is 'race_cluster' or 'general_classification'
      and stat is 'max', 'min', '75th', '25th', 'median'
    - roster_time_since_last_update_{source}_{rank}: Time since last update for top 5 riders per source
      ranked by conservative skills (ranks 1-5)
    - teammates_num: Number of teammates after exclusion
    
    Parameters:
    -----------
    team_roster_df : pd.DataFrame
        DataFrame containing rider information with TrueSkill ratings
    k_values : list
        List of k values for conservative skill estimates
    leader_only : bool
        If True, only create leader features
    teammate_only : bool
        If True, only create teammate features
    exclude_current_rider : str, optional
        Rider identifier to exclude from teammate calculations
        
    Returns:
    --------
    dict : Dictionary of roster-level features
    """
    features = {}
    
    # Create working dataframe, excluding current rider if specified
    working_df = team_roster_df.copy()
    if exclude_current_rider is not None and 'rider' in working_df.columns:
        working_df = working_df[working_df['rider'] != exclude_current_rider]
    
    # Determine skill types based on flags
    if leader_only:
        skill_types = ['leader']
    elif teammate_only:
        skill_types = ['teammate']
    else:
        skill_types = ['leader', 'teammate']
    
    # Define skill sources to process
    skill_sources = [
        ('race_cluster', 'race_cluster_{skill_type}_mu', 'race_cluster_{skill_type}_sigma'),
        ('general_classification', 'General_Classification_{skill_type}_mu', 'General_Classification_{skill_type}_sigma')
    ]
    
    # Process each skill source and skill type combination
    for source_name, mu_pattern, sigma_pattern in skill_sources:
        for skill_type in skill_types:
            mu_col = mu_pattern.format(skill_type=skill_type)
            sigma_col = sigma_pattern.format(skill_type=skill_type)
            
            # Conservative skill estimates for different k values
            for k in k_values:
                conservative_skills = working_df[mu_col] - k * working_df[sigma_col]
                
                if len(conservative_skills) == 0:
                    # No teammates after exclusion
                    features[f'roster_{skill_type}_{source_name}_k{k}_max'] = np.nan
                    features[f'roster_{skill_type}_{source_name}_k{k}_min'] = np.nan
                    features[f'roster_{skill_type}_{source_name}_k{k}_75th'] = np.nan
                    features[f'roster_{skill_type}_{source_name}_k{k}_25th'] = np.nan
                    features[f'roster_{skill_type}_{source_name}_k{k}_median'] = np.nan
                else:
                    # Statistical aggregations
                    features[f'roster_{skill_type}_{source_name}_k{k}_max'] = round(conservative_skills.max(), 2)
                    features[f'roster_{skill_type}_{source_name}_k{k}_min'] = round(conservative_skills.min(), 2)
                    features[f'roster_{skill_type}_{source_name}_k{k}_75th'] = round(conservative_skills.quantile(0.75), 2)
                    features[f'roster_{skill_type}_{source_name}_k{k}_25th'] = round(conservative_skills.quantile(0.25), 2)
                    features[f'roster_{skill_type}_{source_name}_k{k}_median'] = round(conservative_skills.median(), 2)

    # Time since last skill update for top 5 riders ranked by conservative skills (for each skill source)
    if len(working_df) > 0:
        ranking_skill_type = skill_types[0] if skill_types else 'teammate'
        ranking_k = k_values[0] if k_values else 3
        
        # Process time since features for each skill source
        for source_name, mu_pattern, sigma_pattern in skill_sources:
            mu_col = mu_pattern.format(skill_type=ranking_skill_type)
            sigma_col = sigma_pattern.format(skill_type=ranking_skill_type)
            
            # Determine time column based on source and skill type
            if source_name == 'race_cluster':
                time_col = f'race_cluster_last_update_{ranking_skill_type}'
            else:  # general_classification
                time_col = f'General_Classification_last_update_{ranking_skill_type}'
                
            # Calculate conservative skills for ranking
            conservative_skills = working_df[mu_col] - ranking_k * working_df[sigma_col]
            
            # Create dataframe with skills and time since for sorting
            ranking_df = pd.DataFrame({
                'conservative_skill': conservative_skills,
                'time_since': working_df[time_col],
                'original_index': working_df.index
            })
            
            # Sort by conservative skill (descending) and take top 5
            ranking_df_sorted = ranking_df.sort_values('conservative_skill', ascending=False, na_position='last')
            
            # Create time_since features for top 5 riders for this source
            for rank in range(1, 6):  # ranks 1-5
                if rank <= len(ranking_df_sorted):
                    features[f'roster_time_since_last_update_{source_name}_{rank}'] = ranking_df_sorted.iloc[rank-1]['time_since']
                else:
                    features[f'roster_time_since_last_update_{source_name}_{rank}'] = np.nan

    else:
        # If no working dataframe, create NaN features for all sources
        for source_name, _, _ in skill_sources:
            for rank in range(1, 6):
                features[f'roster_time_since_last_update_{source_name}_{rank}'] = np.nan
    
    # Add number of teammates (after exclusion)
    features['teammates_num'] = len(working_df)
    
    return features

def create_roster_cluster_class_skill_features(team_roster_df, k_values=[3, 7], leader_only=False, teammate_only=False, exclude_current_rider=None):
    """
    Create roster-level features specifically for the current race cluster and GC, broken down by race class.
    
    Features created for both current race cluster and GC:
    - roster_{skill_type}_{cluster}_{race_class}_k{k}_{stat}: Conservative skill aggregations for each race class
      where skill_type is 'leader' or 'teammate'
      and stat is 'max', 'min', '75th', '25th', 'median'
    - roster_time_since_last_update_{cluster}_{race_class}_{rank}: Time since last update for top 3 riders per race class
      ranked by conservative skills (ranks 1-3)
    - teammates_num_{cluster}_{race_class}: Number of teammates with experience in this cluster-class combination
    
    Parameters:
    -----------
    team_roster_df : pd.DataFrame
        DataFrame containing rider information with TrueSkill ratings
    k_values : list
        List of k values for conservative skill estimates
    leader_only : bool
        If True, only create leader features
    teammate_only : bool
        If True, only create teammate features
    exclude_current_rider : str, optional
        Rider identifier to exclude from teammate calculations
        
    Returns:
    --------
    dict : Dictionary of roster-level features
    """
    features = {}
    
    # Create working dataframe, excluding current rider if specified
    working_df = team_roster_df.copy()
    if exclude_current_rider is not None and 'rider' in working_df.columns:
        working_df = working_df[working_df['rider'] != exclude_current_rider]
    
    if len(working_df) == 0:
        return features
    
    # Determine skill types based on flags
    if leader_only:
        skill_types = ['leader']
    elif teammate_only:
        skill_types = ['teammate']
    else:
        skill_types = ['leader', 'teammate']
    
    # Define clusters to process (current race cluster and GC)
    clusters_to_process = ['race_cluster', 'General Classification']
    
    # Get unique race classes from the column names
    race_classes =  ['WT', 'Pro', '1', '2']

    # Process each cluster, race class, and skill type combination
    for cluster in clusters_to_process:
        cluster_safe = cluster.replace(" ", "_").replace(",", "_")
        for race_class in race_classes:
            
            for skill_type in skill_types:
                # Get mu and sigma columns for this combination
                mu_col = f'{cluster_safe}_{race_class}_{skill_type}_mu'
                sigma_col = f'{cluster_safe}_{race_class}_{skill_type}_sigma'
                time_col = f'{cluster_safe}_{race_class}_last_update_{skill_type}'
                
                if mu_col not in working_df.columns or sigma_col not in working_df.columns:
                    continue
                
                # Get riders with experience in this cluster-class combination
                experienced_riders = working_df[working_df[mu_col].notna()]
                                
                if len(experienced_riders) == 0:
                    # No riders with experience in this combination
                    for k in k_values:
                        features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_max'] = np.nan
                        features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_min'] = np.nan
                        features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_75th'] = np.nan
                        features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_25th'] = np.nan
                        features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_median'] = np.nan
                    
                    # Set time since features to NaN
                    for rank in range(1, 4):  # top 3 riders
                        features[f'roster_time_since_last_update_{cluster_safe}_{race_class}_{rank}'] = np.nan
                    continue
                
                # Calculate conservative skills for each k value
                for k in k_values:
                    conservative_skills = experienced_riders[mu_col] - k * experienced_riders[sigma_col]
                    
                    # Calculate statistical aggregations
                    features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_max'] = round(conservative_skills.max(), 2)
                    features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_min'] = round(conservative_skills.min(), 2)
                    features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_75th'] = round(conservative_skills.quantile(0.75), 2)
                    features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_25th'] = round(conservative_skills.quantile(0.25), 2)
                    features[f'roster_{skill_type}_{cluster_safe}_{race_class}_k{k}_median'] = round(conservative_skills.median(), 2)
                
                # Calculate time since features for top 3 riders
                if time_col in working_df.columns:
                    # Use first k value for ranking
                    k = k_values[0]
                    conservative_skills = experienced_riders[mu_col] - k * experienced_riders[sigma_col]
                    
                    # Create ranking dataframe
                    ranking_df = pd.DataFrame({
                        'conservative_skill': conservative_skills,
                        'time_since': experienced_riders[time_col]
                    })
                    
                    # Sort by conservative skill and get top 3
                    ranking_df_sorted = ranking_df.sort_values('conservative_skill', ascending=False, na_position='last')
                    
                    # Create time since features
                    for rank in range(1, 6):  # top 5 riders
                        if rank <= len(ranking_df_sorted):
                            features[f'roster_time_since_last_update_{cluster_safe}_{race_class}_{rank}'] = ranking_df_sorted.iloc[rank-1]['time_since']
                        else:
                            features[f'roster_time_since_last_update_{cluster_safe}_{race_class}_{rank}'] = np.nan
    
    return features

def extract_race_features(race_result):
    """Extract race context features"""
    race_result = race_result.copy()

    def race_class_to_ordinal(race_class):
        if race_class == 'WT':
            return 4
        elif race_class == 'Pro':
            return 3
        elif race_class == '1':
            return 2
        else: 
            return 1
    
    race_result['race_class_ord'] = race_result['race_class'].apply(race_class_to_ordinal)
    race_result['StageRace'] = race_result['race'].apply(lambda x: 1 if 'Stage' in x else 0)

    # One hot encode the season phase and cluster
    parcours = pd.get_dummies(race_result['cluster'], prefix='cluster')

    # Concatenate the one-hot encoded columns
    race_result = pd.concat([race_result, parcours], axis=1)

    parcours_cols = list(parcours.columns)
    
    # Transform to integers
    race_result[parcours_cols] = race_result[parcours_cols].astype(int)

    # New columns names combined 
    new_cols = parcours_cols + ['race_class_ord', 'distance', 'verticalMeters', 'profileScore', 'StageRace']
    return race_result, new_cols

def extract_race_ensemble_features(race_data):
        """
        Extract race-level context features
        
        Args:
            race_data (pd.DataFrame): Race data
            
        Returns:
            np.array: Race context features
        """
        # Cluster encoding
        cluster_features = {
            'Flat': [1, 0, 0, 0, 0],
            'Hills, flat finish': [0, 1, 0, 0, 0],
            'Hills, uphill finish': [0, 0, 1, 0, 0],
            'Mountains, flat finish': [0, 0, 0, 1, 0],
            'Mountains, uphill finish': [0, 0, 0, 0, 1],
            'Time Trial': [0, 0, 0, 0, 0]
        }
        
        cluster = race_data['cluster'].iloc[0]
        cluster_encoding = cluster_features.get(cluster, [0, 0, 0, 0, 0])
        
        # Race class and other features
        race_class = [race_data['race_class_ord'].iloc[0]] if 'race_class_ord' in race_data.columns else [1]
        stage_race_flag = [race_data['StageRace'].iloc[0]] if 'StageRace' in race_data.columns else [0]
        n_teams = len(race_data)
        team_count = [n_teams]
        
        # Combine features
        race_features = cluster_encoding + race_class + team_count + stage_race_flag
        return np.array(race_features)