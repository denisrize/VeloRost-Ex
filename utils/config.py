"""
Configuration constants, paths, and feature definitions.
Organized by category to make edits and discovery easier.
"""

# ============================================================================
# Core constants and defaults
# ============================================================================

ROOT_DIR = '/sise/robertmo-group/Denis/Models/'
RACE_CLASSES = ['WT', 'Pro', '1', '2']
AVAILABLE_SCHEMES = ['time_lag', 'equal_weight', 'rank_norm']  # 'time_lag_perc' baseline,'leader' []
CLUSTERS = [
    'Flat', 'Hills, flat finish', 'Hills, uphill finish',
    'Mountains, flat finish', 'Mountains, uphill finish', 'Time Trial'
]
MIN_TEAM_SIZE = 3
MIN_TEAMS_PER_RACE = 10
FILTERED_RACE_RESULTS_PATH = (
    f'data_sets/race_results/riders_race_results_filtered_'
    f'{MIN_TEAM_SIZE}_team_min_{MIN_TEAMS_PER_RACE}_teams_min_updated.csv'
)

# Columns excluded across experiments
ID_COLS = [
    'race', 'date', 'team', 'rider', 'rank_number', 'classification',
    'cluster', 'race_class', 'year', 'rank'
]
EXCLUDE_COLS = ID_COLS + [
    'team_rank', 'total_points', 'best_rank', 'race_id', 'label',
    'fusion_scheme', 'fusion_cluster', 'rider_race_points'
]

# Direct ranking defaults
DIRECT_RANKING_K_GRID = list(range(1, 29))
DIRECT_RANKING_LAMBDA_GRID = [i * 0.05 for i in range(21)]

# ============================================================================
# Output and data directory configuration
# ============================================================================

# These can be overridden by environment variables or passed parameters.
DEFAULT_OUTPUT_CONFIG = {
    'base_models': 'results/roster_ranking',
    'ensemble': 'results/roster_ensemble_ranking',
    'rider_base_models': 'results/rider_ranking',
    'rider_ensemble': 'results/rider_ensemble_ranking/class_features',
    'feature_importance': 'results/feature_importance_analysis',
    'rider_feature_importance': 'results/rider_feature_importance_analysis',
    'job_outputs': 'job_outputs',
    'job_scripts': 'job_scripts',
    'direct_ranking': 'results/direct_ranking'
}

# These define where to load/save datasets for each experiment type.
DEFAULT_DATA_CONFIG = {
    'roster_datasets': 'data_sets/roster_datasets_new',
    'rider_datasets': 'data_sets/rider_datasets_new',
    'leader_power': 'data_sets/leader_power/cluster',
    'leader_power_class': 'data_sets/leader_power/race_class',
    'team_power': 'data_sets/team_power_flatten/alone',
    'team_power_class': 'data_sets/team_power_new/race_class',
    'rider_features': 'data_sets/rider_features',
    'raw_riders_race_results': 'data_set\race_results\riders_race_results.csv',
    'riders_race_results': FILTERED_RACE_RESULTS_PATH,
}

# ============================================================================
# Directory and path helpers
# ============================================================================

def get_output_dir(experiment_type, race_class='all', custom_base=None, level='roster'):
    """
    Get the output directory for a specific experiment type.
    
    Args:
        experiment_type (str): Type of experiment ('base_models', 'ensemble', 'fusion')
        race_class (str): Race class ('all' or 'WT')
        year (int): Year for results (default: 2023)
        custom_base (str): Custom base directory (overrides default)
        k_value (int): K value for NDCG@k evaluation (default: 5)
        level (str): Level of ranking ('roster' or 'rider')
    Returns:
        str: Full output directory path
    """
    import os
    
    # Determine the config key based on level
    config_key = f"{level}_{experiment_type}" if level == 'rider' else experiment_type
    
    # Check for environment variable override
    env_var = f"ROSTER_RANKING_{config_key.upper()}_DIR"
    if env_var in os.environ:
        base_dir = os.environ[env_var]
    else:
        base_dir = ROOT_DIR + f'teams_ranking/trueSkill_ranking/{DEFAULT_OUTPUT_CONFIG[config_key]}'
    
    if custom_base:
        base_dir += custom_base

    return f"{base_dir}/{race_class}"

def get_hyperparams_dir(race_class, year=2023, custom_base=None, k_value=5, level='roster'):
    """
    Get the hyperparameters directory.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        custom_base (str): Custom base directory (overrides default)
        k_value (int): K value for NDCG@k evaluation (default: 5)
        level (str): Level of ranking ('roster' or 'rider')
    Returns:
        str: Full hyperparameters directory path
    """
    import os
    
    # Determine the config key based on level
    config_key = f"{level}_base_models" if level == 'rider' else "base_models"
    
    # Check for environment variable override
    env_var = "ROSTER_RANKING_HYPERPARAMS_DIR"
    if env_var in os.environ:
        base_dir = os.environ[env_var]
    elif custom_base:
        base_dir = custom_base
    else:
        base_dir = ROOT_DIR + f'teams_ranking/trueSkill_ranking/{DEFAULT_OUTPUT_CONFIG[config_key]}'
    
    return f"{base_dir}/{race_class}/hyperparameters"

def get_data_dir(data_type, custom_base=None):
    """
    Get the data directory for a specific data type.
    
    Args:
        data_type (str): Type of data ('roster_datasets', 'leader_power', 'team_power', 'rider_features', etc.)
        custom_base (str): Custom base directory (overrides default)
        
    Returns:
        str: Full data directory path
    """
    import os
    
    # Check for environment variable override
    env_var = f"ROSTER_RANKING_{data_type.upper()}_DIR"
    if env_var in os.environ:
        return os.environ[env_var]
    elif custom_base:
        return custom_base
    else:
        base_dir = ROOT_DIR + f'teams_ranking/trueSkill_ranking/{DEFAULT_DATA_CONFIG[data_type]}'
        return base_dir

def get_roster_dataset_path(race_class, scheme):
    """
    Get the full path for a roster dataset file.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name
        
    Returns:
        str: Full file path for the roster dataset
    """
    dataset_filename = f'roster_dataset_{race_class}_{scheme}_scheme.csv'
    data_dir = get_data_dir('roster_datasets')
    return f"{data_dir}/{dataset_filename}"

def get_rider_dataset_path(race_class, scheme):
    """
    Get the full path for a rider dataset file.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name
        
    Returns:
        str: Full file path for the rider dataset
    """
    dataset_filename = f'rider_dataset_{race_class}_{scheme}_scheme.csv'
    data_dir = get_data_dir('rider_datasets')
    return f"{data_dir}/{dataset_filename}"

def get_leader_power_path(race_class, by_class=False, time_gap=None):
    """
    Get the full path for leader power/TrueSkill features.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        time_lag (int): Time lag for time-lagged features (default: None)
    Returns:
        str: Full file path for leader power features
    """
    if time_gap:
        if time_gap == 365:
            time_gap = 'season_start'
        filename = 'trueSkill_features_class_all_ratings_snapshot_leader_scheme_2024_2025_2026_years_season_start_tg.csv'
        # filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_leader_scheme_{time_gap}_tg.csv'
    else:
        filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_leader_scheme.csv'

    if by_class:
        data_dir = get_data_dir('leader_power_class')
    else:
        data_dir = get_data_dir('leader_power')
    return f"{data_dir}/{filename}"

def get_team_power_path(race_class, scheme, by_class=False, time_gap=None):
    """
    Get the full path for team power/TrueSkill features.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name
        
    Returns:
        str: Full file path for team power features
    """
    if time_gap:
        if time_gap == 365:
            time_gap = 'season_start'
        filename = 'trueSkill_features_class_all_ratings_snapshot_equal_weight_scheme_2024_2025_2026_years_season_start_tg.csv'
        # filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme_{time_gap}_tg.csv'
    else:
        filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme.csv'
    if by_class:
        data_dir = get_data_dir('team_power_class')
    else:
        data_dir = get_data_dir('team_power')
        
    return f"{data_dir}/{filename}"

def get_rider_features_path(race_class, time_gap=None):
    """
    Get the full path for rider features.
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        time_lag (int): Time lag for time-lagged features (default: None)
    Returns:
        str: Full file path for rider features
    """
    if time_gap:
        if time_gap == 365:
            time_gap = 'season_start'
        filename = 'all_riders_features_all_race_class_years_updated_365_tg.csv'
        # filename = f'all_riders_features_{race_class}_race_class_{time_gap}_tg.csv'
    else:
        filename = f'all_riders_features_{race_class}_race_class.csv'
    data_dir = get_data_dir('rider_features')
    return f"{data_dir}/{filename}"

def get_previous_trueSkill_rating(race_class, scheme, time_gap, leader=True):
    """
    Get the file path for a previous TrueSkill snapshot.
    """
    if time_gap:
        if time_gap == 365:
            time_gap = 'season_start'
        filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme_{time_gap}_tg.csv'
    else:
        filename = f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme.csv'
    if leader:
        data_dir = get_data_dir('leader_power')
    else:
        data_dir = get_data_dir('team_power')
        
    return f"{data_dir}/{filename}"

# ============================================================================
# Ranking helpers
# ============================================================================

def get_ranking_config(level='roster'):
    """
    Get level-specific ranking configuration.
    
    Args:
        level (str): Level of ranking ('roster' or 'rider')
        
    Returns:
        tuple: (rank_column, record_id_column)
    """
    if level == 'roster':
        return 'team_rank', 'team'
    elif level == 'rider':
        return 'rank_number', 'rider'
    else:
        raise ValueError(f"Invalid level: {level}. Must be 'roster' or 'rider'")

def get_rank_col(level='roster'):
    """
    Get the rank column name for a specific level.
    """
    rank_col, _ = get_ranking_config(level)
    return rank_col

def get_record_id(level='roster'):
    """
    Get the record ID column name for a specific level.
    """
    _, record_id = get_ranking_config(level)
    return record_id

# ============================================================================
# Gating network hyperparameter grids
# ============================================================================

def get_logistic_regression_hyperparameter_grid():
    """
    Get comprehensive hyperparameter grid for logistic regression gating network.
    """
    from sklearn.model_selection import ParameterGrid
    
    # Define compatible combinations
    param_combinations = [
        # L2 penalty with liblinear (only supports ovr)
        {
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        },
        # L2 penalty with lbfgs - test both multinomial (preferred) and ovr
        {
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'multi_class': ['multinomial', 'ovr'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        },
        # L2 penalty with saga - test both multinomial (preferred) and ovr
        {
            'penalty': ['l2'],
            'solver': ['saga'],
            'multi_class': ['multinomial', 'ovr'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        },
        # L1 penalty with liblinear (only supports ovr)
        {
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'multi_class': ['ovr'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        },
        # L1 penalty with saga - test both multinomial and ovr
        {
            'penalty': ['l1'],
            'solver': ['saga'],
            'multi_class': ['multinomial', 'ovr'],
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        },
        # Elasticnet penalty with saga - test both multinomial (preferred) and ovr
        {
            'penalty': ['elasticnet'],
            'solver': ['saga'],
            'multi_class': ['multinomial', 'ovr'],
            'C': [0.1, 1.0, 10.0],  # Fewer C values since elasticnet is more expensive
            'l1_ratio': [0.1, 0.5, 0.9],  # L1 ratio for elasticnet
            'max_iter': [2000, 3000],  # More iterations for saga solver
            'class_weight': [None, 'balanced']
        }
    ]
    
    # Generate all valid combinations
    param_grid = []
    for combo in param_combinations:
        param_grid.extend(list(ParameterGrid(combo)))
    
    return param_grid

def get_mlp_hyperparameter_grid():
    """
    Get hyperparameter grid for MLP gating network.
    """
    from sklearn.model_selection import ParameterGrid
    
    # MLP hyperparameter grid  
    param_grid = [
        {
            'batch_size': [32],  # Focus around 32
            'lr': [0.00005,0.0001, 0.0005],  # Narrow around 0.001
            'hidden_dims': [
                [64, 32], [128, 64],  # Two layer variations
                [64, 32, 16], [128, 64, 32] # Three layer option
            ],
            'dropout': [0.1, 0.15, 0.2],  # Fine-tune around working range
            'weight_decay': [0.00005, 0.0001],  # Fine-tune around 0.0001
            'learnable_temperature': [False],  # Stick with fixed
            'init_temperature': [1.0]
        }
    ]
    
    # Flatten grid for MLP
    param_grid = list(ParameterGrid(param_grid))
    return param_grid

# ============================================================================
# Feature name generators
# ============================================================================

def generate_roster_skill_features(skill_types, sources, k_values, stats):
    """Generate roster skill feature names systematically."""
    features = []
    for skill_type in skill_types:
        for source in sources:
            for k in k_values:
                for stat in stats:
                    features.append(f'roster_{skill_type}_{source}_k{k}_{stat}')
    return features

def generate_roster_k6_flattened_features(sources):
    """Generate K=6 flattened roster feature names systematically."""
    features = []
    for source in sources:
        # Helper features (5 teammates)
        for i in range(1, 6):
            features.append(f'roster_helper_{i}_mu_{source}')
            features.append(f'roster_helper_{i}_sigma_{source}')
        
        # Global aggregates
        features.append(f'roster_mean_mu_{source}')
        features.append(f'roster_mean_sigma_{source}')
        features.append(f'roster_mean_mu_sigma_ratio_{source}')
    
    # Add roster size (not per source)
    features.append('roster_size')
    
    return features

def generate_roster_skill_features_by_class(skill_types, sources, k_values, stats, race_classes):
    """Generate roster skill feature names systematically."""
    features = []
    for skill_type in skill_types:
        for source in sources:
            for race_class in race_classes:
                for k in k_values:
                    for stat in stats:
                        features.append(f'roster_{skill_type}_{source}_{race_class}_k{k}_{stat}')
    return features

def generate_time_since_features(sources, ranks=range(1, 6)):
    """Generate time since last update features for top 5 riders."""
    features = []
    for source in sources:
        for rank in ranks:
            features.append(f'roster_time_since_last_update_{source}_{rank}')
    return features

# ============================================================================
# TrueSkill feature names (raw and roster)
# ============================================================================

# Base feature components
SKILL_TYPES = ['leader', 'teammate']
SKILL_SOURCES = ['race_cluster', 'general_classification']
K_VALUES = [3, 7]
STATS = ['max', 'min', '75th', '25th', 'median']

# Raw TrueSkill features by race class
RAW_TRUE_SKILL_LEADER_CLASS_FEATURES = [
    'race_cluster_WT_leader_mu', 'race_cluster_WT_leader_sigma',
    'race_cluster_Pro_leader_mu', 'race_cluster_Pro_leader_sigma',
    'race_cluster_1_leader_mu', 'race_cluster_1_leader_sigma',
    'race_cluster_2_leader_mu', 'race_cluster_2_leader_sigma',
]
RAW_TRUE_SKILL_LEADER_TIME_FEATURES_CLASS = [
    'race_cluster_WT_last_update_leader', 'race_cluster_Pro_last_update_leader',
    'race_cluster_1_last_update_leader', 'race_cluster_2_last_update_leader'
]
ALL_RAW_LEADER_CLASS_FEATURES = RAW_TRUE_SKILL_LEADER_CLASS_FEATURES + RAW_TRUE_SKILL_LEADER_TIME_FEATURES_CLASS

RAW_TRUE_SKILL_LEADER_CLASS_GC_FEATURES = [
    'General_Classification_WT_leader_mu', 'General_Classification_WT_leader_sigma',
    'General_Classification_Pro_leader_mu', 'General_Classification_Pro_leader_sigma',
    'General_Classification_1_leader_mu', 'General_Classification_1_leader_sigma',
    'General_Classification_2_leader_mu', 'General_Classification_2_leader_sigma',
]
RAW_TRUE_SKILL_TEAMMATE_CLASS_FEATURES = [
    'race_cluster_WT_teammate_mu', 'race_cluster_WT_teammate_sigma',
    'race_cluster_Pro_teammate_mu', 'race_cluster_Pro_teammate_sigma',
    'race_cluster_1_teammate_mu', 'race_cluster_1_teammate_sigma',
    'race_cluster_2_teammate_mu', 'race_cluster_2_teammate_sigma',
]
RAW_TRUE_SKILL_TEAMMATE_CLASS_GC_FEATURES = [
    'General_Classification_WT_teammate_mu', 'General_Classification_WT_teammate_sigma',
    'General_Classification_Pro_teammate_mu', 'General_Classification_Pro_teammate_sigma',
    'General_Classification_1_teammate_mu', 'General_Classification_1_teammate_sigma',
    'General_Classification_2_teammate_mu', 'General_Classification_2_teammate_sigma',
]

# Leader features by cluster
RAW_TRUE_SKILL_LEADER_FEATURES = ['race_cluster_leader_mu', 'race_cluster_leader_sigma']
RAW_TIME_LEADER = ['race_cluster_last_update_leader']
RAW_TRUE_SKILL_LEADER_GC_FEATURES = ['gc_leader_mu', 'gc_leader_sigma']
RAW_TIME_LEADER_GC = ['gc_last_update_leader']
ALL_RAW_LEADER_FEATURES = RAW_TRUE_SKILL_LEADER_FEATURES + RAW_TRUE_SKILL_LEADER_GC_FEATURES + RAW_TIME_LEADER + RAW_TIME_LEADER_GC

# Teammate features by cluster
RAW_TRUE_SKILL_TEAMMATE_FEATURES = ['race_cluster_teammate_mu', 'race_cluster_teammate_sigma']
RAW_TIME_TEAMMATE = ['race_cluster_last_update_teammate']
RAW_TRUE_SKILL_TEAMMATE_GC_FEATURES = ['gc_teammate_mu', 'gc_teammate_sigma']
RAW_TIME_TEAMMATE_GC = ['gc_last_update_teammate']
RAW_TRUE_SKILL_TEAMMATE = RAW_TRUE_SKILL_TEAMMATE_FEATURES + RAW_TRUE_SKILL_TEAMMATE_GC_FEATURES + RAW_TIME_TEAMMATE + RAW_TIME_TEAMMATE_GC

# TrueSkill roster features by class
ROSTER_TRUE_SKILL_CLASS_LEADER_FEATURES_3 = generate_roster_skill_features_by_class(['leader'], ['race_cluster'], [3], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_LEADER_FEATURES_7 = generate_roster_skill_features_by_class(['leader'], ['race_cluster'], [7], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_LEADER_GC_FEATURES_3 = generate_roster_skill_features_by_class(['leader'], ['general_classification'], [3], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_LEADER_GC_FEATURES_7 = generate_roster_skill_features_by_class(['leader'], ['general_classification'], [7], STATS, RACE_CLASSES)

ROSTER_TRUE_SKILL_CLASS_TEAMMATE_FEATURES_3 = generate_roster_skill_features_by_class(['teammate'], ['race_cluster'], [3], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_TEAMMATE_FEATURES_7 = generate_roster_skill_features_by_class(['teammate'], ['race_cluster'], [7], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_TEAMMATE_GC_FEATURES_3 = generate_roster_skill_features_by_class(['teammate'], ['general_classification'], [3], STATS, RACE_CLASSES)
ROSTER_TRUE_SKILL_CLASS_TEAMMATE_GC_FEATURES_7 = generate_roster_skill_features_by_class(['teammate'], ['general_classification'], [7], STATS, RACE_CLASSES)

# TrueSkill roster features by cluster
ROSTER_TRUE_SKILL_LEADER_FEATURES_3 = generate_roster_skill_features(['leader'], ['race_cluster'], [3], STATS)
ROSTER_TRUE_SKILL_LEADER_FEATURES_7 = generate_roster_skill_features(['leader'], ['race_cluster'], [7], STATS)
ROSTER_TRUE_SKILL_LEADER_GC_FEATURES_3 = generate_roster_skill_features(['leader'], ['general_classification'], [3], STATS)
ROSTER_TRUE_SKILL_LEADER_GC_FEATURES_7 = generate_roster_skill_features(['leader'], ['general_classification'], [7], STATS)

ROSTER_TRUE_SKILL_TEAMMATE_FEATURES_3 = generate_roster_skill_features(['teammate'], ['race_cluster'], [3], STATS)
ROSTER_TRUE_SKILL_TEAMMATE_FEATURES_7 = generate_roster_skill_features(['teammate'], ['race_cluster'], [7], STATS)
ROSTER_TRUE_SKILL_TEAMMATE_GC_FEATURES_3 = generate_roster_skill_features(['teammate'], ['general_classification'], [3], STATS)
ROSTER_TRUE_SKILL_TEAMMATE_GC_FEATURES_7 = generate_roster_skill_features(['teammate'], ['general_classification'], [7], STATS)

# Time since features by class
ROSTER_TIME_SINCE_FEATURES = generate_time_since_features(['race_cluster'], range(1, 6))
ROSTER_TIME_SINCE_FEATURES_GC = generate_time_since_features(['general_classification'], range(1, 6))
ROSTER_FEATURES = ['teammates_num'] + ROSTER_TRUE_SKILL_TEAMMATE_FEATURES_7 + ROSTER_TRUE_SKILL_TEAMMATE_FEATURES_3 + ROSTER_TRUE_SKILL_TEAMMATE_GC_FEATURES_3

# K=6 flattened TrueSkill roster features
ROSTER_TRUE_SKILL_K6_TEAMMATE_RACE_CLUSTER = generate_roster_k6_flattened_features(['race_cluster'])
ROSTER_TRUE_SKILL_K6_TEAMMATE_GC = generate_roster_k6_flattened_features(['gc'])
ROSTER_TRUE_SKILL_K6_TEAMMATE_BOTH = generate_roster_k6_flattened_features(['race_cluster', 'gc'])

ROSTER_TRUE_SKILL_K6_CLASS_TEAMMATE_RACE_CLUSTER = generate_roster_k6_flattened_features(['race_cluster_class'])
ROSTER_TRUE_SKILL_K6_CLASS_TEAMMATE_GC = generate_roster_k6_flattened_features(['gc_class'])
ROSTER_TRUE_SKILL_K6_CLASS_TEAMMATE_BOTH = generate_roster_k6_flattened_features(['race_cluster_class', 'gc_class'])

# ============================================================================
# Feature importance analysis configuration
# ============================================================================

def get_feature_importance_methods_config():
    """
    Get configuration for feature importance analysis methods.
    """
    return {
        'permutation': {
            'name': 'Permutation Importance',
            'color': '#1f77b4',
            'order': 1,
            'enabled': True,
            'n_repeats': 30,  # Number of permutation repeats
            'top_features': 20,  # Number of top features to show
            'random_seed': 42
        },
        'lofo': {
            'name': 'LOFO/Drop-Column',
            'color': '#ff7f0e',
            'order': 2,
            'enabled': True,
            'top_features_only': 20,  # Only analyze top N features from permutation
            'display_features': 20    # Number of features to show in plots
        },
        'shap': {
            'name': 'SHAP (TreeSHAP)',
            'color': '#2ca02c',
            'order': 3,
            'enabled': True,
            'sample_size': 10000,  # Sample size for SHAP analysis
            'top_features': 20,   # Number of top features to show
            'save_warm_shap': True,  # Save data for warm SHAP plots
            'create_dependence_plots': True,  # Create SHAP dependence plots
            'max_dependence_features': 5  # Max features for dependence plots
        }
    }

def get_feature_importance_output_config():
    """
    Get output configuration for feature importance analysis.
    """
    return {
        'save_plots': True,
        'save_results': True,
        'plot_format': 'png',
        'plot_dpi': 300,
        'create_ensemble_summary': True,
        'create_method_comparisons': True
    }

# ============================================================================
# Rider and race feature definitions
# ============================================================================

RIDER_PREVIOUS_RESULTS_FEATURES = [
    'Previous Result Race Cluster',
    'Result 2 Races Ago Cluster',
    'Result 3 Races Ago Cluster',
    'Result 4 Races Ago Cluster',
    'Result 5 Races Ago Cluster',
    'Result 6 Races Ago Cluster',
    'Result 7 Races Ago Cluster',
    'Result 8 Races Ago Cluster',
    'Result 9 Races Ago Cluster',
    'Result 10 Races Ago Cluster'
]
RIDER_CORE_FEATURES = [
    'age', 'CareerLength', 'form', 'best_result_cluster', 'time_since_best_result_cluster',
    'pointsMinus1', 'pointsMinus2', 'pointsMinus3', 'CareerSlope',
    'PointsPastSeasons', 'PointsStartYear', 'PointsStartYearRaceType', 'PointsPastSeasonsRaceType'
]

RIDER_BASELINE_FEATURES = RIDER_PREVIOUS_RESULTS_FEATURES + RIDER_CORE_FEATURES
RIDER_BASELINE_FEATURES_EXTEND = RIDER_BASELINE_FEATURES + [
    'best_result_class',
    'time_since_best_result_class',
    'PointsStartYearRaceClass',
    'PointsPastSeasonsRaceClass'
]

RIDER_FEATURES_WT = RIDER_CORE_FEATURES + [
    # Points features
    'PointsStartYearCluster',
    'PointsPastSeasonsCluster',
    'PointsStartYearGC',
    'PointsPastSeasonsGC',
    
    # Historical performance bins - Cluster based
    'Cluster_rank_1_3_count',
    'Cluster_rank_4_10_count',
    'Cluster_rank_11_30_count',
    'Cluster_rank_31_50_count',
    'Cluster_dnf_count',
    'Cluster_rank_1_3_days_since',
    'Cluster_rank_4_10_days_since',
    'Cluster_rank_11_30_days_since',
    'Cluster_rank_31_50_days_since',
    'Cluster_dnf_days_since',
    'Cluster_total_races',
    
    # Historical performance bins - Race Class based
    'RaceClass_rank_1_3_count',
    'RaceClass_rank_4_10_count',
    'RaceClass_rank_11_30_count',
    'RaceClass_rank_31_50_count',
    'RaceClass_dnf_count',
    'RaceClass_rank_1_3_days_since',
    'RaceClass_rank_4_10_days_since',
    'RaceClass_rank_11_30_days_since',
    'RaceClass_rank_31_50_days_since',
    'RaceClass_dnf_days_since',
    'RaceClass_total_races',
    
    # Best result features
    'best_result_class',
    'time_since_best_result_class',
    'best_result_gc',
    'time_since_best_result_gc',
]
RIDER_FEATURES = RIDER_FEATURES_WT + [
    'best_result_cluster_class', 'time_since_best_result_cluster_class',
    'PointsStartYearRaceClass',
    'PointsPastSeasonsRaceClass',

    # Historical performance bins - Cluster & Race Class combined
    'ClusterClass_rank_1_3_count',
    'ClusterClass_rank_4_10_count',
    'ClusterClass_rank_11_30_count',
    'ClusterClass_rank_31_50_count',
    'ClusterClass_dnf_count',
    'ClusterClass_rank_1_3_days_since',
    'ClusterClass_rank_4_10_days_since',
    'ClusterClass_rank_11_30_days_since',
    'ClusterClass_rank_31_50_days_since',
    'ClusterClass_dnf_days_since',
    'ClusterClass_total_races'
]

RACE_BASELINE_FEATURES = [
    'StageRace', 'cluster_Flat', 'cluster_Hills, flat finish', 'cluster_Hills, uphill finish',
    'cluster_Mountains, flat finish', 'cluster_Mountains, uphill finish', 'cluster_Time Trial',
]

RACE_FEATURES_WT = RACE_BASELINE_FEATURES + ['distance', 'verticalMeters', 'profileScore']
RACE_FEATURES_ALL = RACE_BASELINE_FEATURES + ['distance', 'verticalMeters', 'profileScore', 'race_class_ord', 'stage_ratio']

DEFAULT_FEATURES_BY_EXPERIMENT = {
    'baseline': RIDER_BASELINE_FEATURES + RACE_BASELINE_FEATURES + RAW_TRUE_SKILL_LEADER_FEATURES,
    'baseline_enhanced': RIDER_FEATURES + RAW_TRUE_SKILL_LEADER_FEATURES + RACE_FEATURES_ALL,
    'class_features': RIDER_FEATURES + RAW_TRUE_SKILL_LEADER_FEATURES + RACE_FEATURES_ALL + ROSTER_TRUE_SKILL_K6_TEAMMATE_BOTH + RAW_TRUE_SKILL_LEADER_GC_FEATURES,
}

def get_features_by_experiment(experiment_type):
    return DEFAULT_FEATURES_BY_EXPERIMENT[experiment_type]
