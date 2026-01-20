#!/usr/bin/env python3

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output for cluster jobs
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from roster_ranker.experiments import (
    BaseModelExperiment,
    EnsembleExperiment,
    FeatureImportanceExperiment,
    DirectRankingExperiment,
)
from roster_ranker.utils import (
    get_output_dir,
    get_hyperparams_dir,
    get_data_dir,
    AVAILABLE_SCHEMES,
    MIN_TEAM_SIZE,
    MIN_TEAMS_PER_RACE,
    FILTERED_RACE_RESULTS_PATH,
    ROOT_DIR,
)
from roster_ranker.data import run_race_results_filtering_pipeline as run_data_filtering_pipeline
from roster_ranker.feature_extraction import (
    run_rider_features_extraction_pipeline,
    run_trueskill_features_extraction_pipeline,
)

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * len(title))


def run_base_models_pipeline(race_class, ensemble_type=None, force_retune=False, year=2023, k_value=10, time_gap=None, save_results=False, level='rider', schemes=None, optimization_strategy='cluster_race_class', exp_name='class_features'):
    """
    Run base models pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        force_retune (bool): Force hyperparameter retuning
        year (int): Year for results (default: 2023)
        level (str): Level of ranking ('roster' or 'rider')
        optimization_strategy (str): Strategy for cluster performance optimization
        
    Returns:
        BaseModelExperiment: Experiment instance with results
    """
    print_header("BASE MODELS PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Experiment name: {exp_name}")
    print(f"Level: {level}")
    print(f"Time gap: {time_gap}")
    print(f"Optimization strategy: {optimization_strategy}")
    print(f"Ensemble type: {ensemble_type}")
    print(f"K value: {k_value}")
    print(f"Schemes: {schemes}")
    print(f"Force retune: {force_retune}")
    
    # Get directories from config
    base_models_dir = get_output_dir('base_models', race_class, level=level)
    hyperparams_dir = get_hyperparams_dir(race_class, k_value=k_value, level=level)
    
    print(f"Output directory: {base_models_dir}")
    print(f"Hyperparams directory: {hyperparams_dir}")
    print(f"Data directories:")
    dataset_type = 'rider_datasets' if level == 'rider' else 'roster_datasets'
    print(f"  {dataset_type.title()}: {get_data_dir(dataset_type)}")
    print(f"  Leader power: {get_data_dir('leader_power')}")
    print(f"  Team power: {get_data_dir('team_power')}")
    print(f"  Rider features: {get_data_dir('rider_features')}")
    
    # Initialize experiment with configured directories
    base_experiment = BaseModelExperiment(
        race_class, 
        schemes=schemes,
        custom_output_dir=base_models_dir,
        custom_hyperparams_dir=hyperparams_dir,
        year=year,
        ensemble_type=ensemble_type,
        k_value=k_value,
        time_gap=time_gap,
        level=level,
        optimization_strategy=optimization_strategy,
        exp_name=exp_name
    )
    
    # Train all base models
    results = base_experiment.train_all_models(force_retune=force_retune, k_value=k_value, save_results=save_results)
    
    print_section("Base Models Results Summary")
    for scheme, (model, results_df, hyperparams) in results.items():
        from roster_ranker.core import calculate_summary_statistics
        summary = calculate_summary_statistics(results_df)
        ndcg = summary.get(f'NDCG@{k_value}', {}).get('mean', 0)
        recall = summary.get(f'Recall@{k_value}', {}).get('mean', 0)
        print(f"  {scheme:12s}: NDCG@{k_value}={ndcg:.4f}, Recall@{k_value}={recall:.4f}")
    
    return base_experiment


def run_ensemble_pipeline(race_class, ensemble_methods=None, force_retune=False, year=2023, k_value=10, time_gap=None, level='rider', optimization_strategy='cluster_race_class', gating_model_type='logistic', exp_name='class_features'):
    """
    Run ensemble pipeline (base models + ensemble)
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        ensemble_methods (list): List of ensemble methods to run
        force_retune (bool): Force hyperparameter retuning
        year (int): Year for results (default: 2023)
        level (str): Level of ranking ('roster' or 'rider')
        optimization_strategy (str): Strategy for cluster performance optimization
        gating_model_type (str): Type of gating model for MoE methods ('logistic' or 'mlp')
        
    Returns:
        tuple: (BaseModelExperiment, EnsembleExperiment)
    """
    print_header("ENSEMBLE PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Level: {level}")
    print(f"Level: {level}")
    print(f"Time gap: {time_gap}")
    print(f"Optimization strategy: {optimization_strategy}")
    print(f"Gating model type: {gating_model_type}")
    print(f"Force retune: {force_retune}")
    print(f"K value: {k_value}")
    # Step 1: Train base models
    print_section("Step 1: Training Base Models")
    base_experiment = run_base_models_pipeline(race_class, ensemble_methods, force_retune, year, k_value, time_gap=time_gap, level=level, optimization_strategy=optimization_strategy, exp_name=exp_name)
    
    # Step 2: Train ensemble
    print_section("Step 2: Training Ensemble")
    
    ensemble_experiment = EnsembleExperiment(
        race_class,
        base_models=base_experiment,
        ensemble_methods=ensemble_methods,
        year=year,
        k_value=k_value,
        time_gap=time_gap,
        level=level,
        optimization_strategy=optimization_strategy,
        gating_model_type=gating_model_type,
        exp_name=exp_name
    )
    
    # Run all ensemble methods
    ensemble_results = ensemble_experiment.run_all_ensemble_methods(force_retune_base=False)  # Base models already trained
    
    print_section("Ensemble Results Summary")
    for method, results_df in ensemble_results.items():
        from roster_ranker.core import calculate_summary_statistics
        summary = calculate_summary_statistics(results_df)
        ndcg = summary.get(f'NDCG@{k_value}', {}).get('mean', 0)
        recall = summary.get(f'Recall@{k_value}', {}).get('mean', 0)
        print(f"  {method:15s}: NDCG@{k_value}={ndcg:.4f}, Recall@{k_value}={recall:.4f}")
    
    return base_experiment, ensemble_experiment


def run_rider_features_pipeline(race_class, time_gap=None, target_years=None):
    """
    Run rider features extraction pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT') 
        time_gap (int): Time gap in days before race for feature extraction (default: 30)
        target_years (list): List of years to extract features for (e.g., [2024, 2025]).
                            If None, extracts for all years in the data.
                            Features will still be calculated using ALL historical data.
        
    Returns:
        str: Path to created rider features file
    """
    print_header("RIDER FEATURES EXTRACTION PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Time gap: {time_gap} days")
    if target_years is not None:
        print(f"Target years: {target_years}")
    try:
        # Import and call the pipeline function from the feature extraction module
        output_path = run_rider_features_extraction_pipeline(race_class, time_gap, target_years)
        return output_path
        
    except Exception as e:
        print(f"❌ Error in rider features extraction: {e}")
        raise


def run_trueskill_features_pipeline(race_class, schemes, time_gap=None, target_years=None):
    """
    Run TrueSkill features extraction pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        schemes (list): List of scheme names ('time_lag', 'equal_weight', 'rank_norm', 'leader', 'baseline')
        k_value (int): K value for team weighting schemes (default: 5)
        force_recreate (bool): Force recreation even if file exists
        
    Returns:
        str: Path to created TrueSkill features file
    """
    print_header("TRUESKILL FEATURES EXTRACTION PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Schemes: {schemes}")
    
    try:
        for scheme in schemes:
            # Import and call the pipeline function from the feature extraction module
            output_path = run_trueskill_features_extraction_pipeline(race_class, scheme, time_gap, target_years)
        return output_path
        
    except Exception as e:
        print(f"❌ Error in TrueSkill features extraction: {e}")
        raise


def run_race_results_filtering_pipeline(
    min_team_size=MIN_TEAM_SIZE,
    min_teams_per_race=MIN_TEAMS_PER_RACE,
):
    """
    Run race results filtering pipeline.
    """
    print_header("RACE RESULTS FILTERING PIPELINE")
    print(f"Min team size in race: {min_team_size}")
    print(f"Min teams per race: {min_teams_per_race}")

    return run_data_filtering_pipeline(
        input_path=get_data_dir('riders_race_results'),
        min_team_size=min_team_size,
        min_teams_per_race=min_teams_per_race,
    )


def run_feature_importance_pipeline(race_class, methods=None, schemes=None, year=2023, k_value=10, time_gap=None, level='rider', force_retune=False):
    """
    Run feature importance analysis pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        methods (list): List of feature importance methods to run ('permutation', 'lofo', 'shap')
        schemes (list): List of schemes to analyze (default: all available)
        year (int): Year for results (default: 2023)
        k_value (int): K value for NDCG@k evaluation (default: 10)
        time_gap (int): Time gap for features (default: None)
        level (str): Level of ranking ('roster' or 'rider')
        force_retune (bool): Force hyperparameter retuning for base models

    Returns:
        FeatureImportanceExperiment: Feature importance experiment instance with results
    """
    print_header("FEATURE IMPORTANCE ANALYSIS PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Level: {level}")
    print(f"Methods: {methods or 'all enabled'}")
    print(f"Schemes: {schemes or 'all available'}")
    print(f"Year: {year}")
    print(f"K value: {k_value}")
    if time_gap:
        print(f"Time gap: {time_gap} days")
    
    try:
        # Initialize feature importance experiment
        feature_importance_experiment = FeatureImportanceExperiment(
            race_class=race_class,
            schemes=schemes,
            methods=methods,
            year=year,
            k_value=k_value,
            time_gap=time_gap,
            level=level
        )
        
        # Run complete feature importance analysis
        results = feature_importance_experiment.run_feature_importance_analysis(
            force_retune=force_retune
        )
        
        print(f"\n✅ Feature importance analysis completed successfully!")
        print(f"  📊 Methods: {list(results.keys())}")
        print(f"  🎯 Schemes: {list(feature_importance_experiment.schemes)}")
        print(f"  💾 Results saved to: {feature_importance_experiment.output_dir}")
        
        return feature_importance_experiment
        
    except Exception as e:
        print(f"❌ Error in feature importance analysis: {e}")
        raise

        
def run_direct_ranking_pipeline(
    race_class,
    schemes=None,
    year=2023,
    k_value=10,
    direct_ranking_mode='evaluation',
    k_penalty=None,
    lambda_value=None,
    top_n=None,
    max_teammates=None
):
    """
    Run direct ranking pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        year (int): Year for results (default: 2023)
        k_value (int): K value for NDCG@k evaluation (default: 10)

    Returns:
        str: Path to created direct ranking file
    """
    print_header("DIRECT RANKING PIPELINE")
    print(f"Race class: {race_class}")
    print(f"Year: {year}")
    print(f"K value: {k_value}")
    print(f"Mode: {direct_ranking_mode}")

    try:
        baseline = False
        if schemes is None:
            schemes = AVAILABLE_SCHEMES
        if schemes == ['leader']:
            baseline = True

        if direct_ranking_mode == 'marginal':
            if k_penalty is None or lambda_value is None:
                raise ValueError("Marginal mode requires --k_penalty and --lambda_value.")

        experiment = DirectRankingExperiment(
            race_class=race_class,
            schemes=schemes,
            k_value=k_value,
            baseline=baseline,
            top_n=top_n,
            max_teammates=max_teammates
        )

        if direct_ranking_mode == 'evaluation':
            return experiment.run_direct_ranking_evaluation(
                race_class=race_class,
                schemes=schemes,
                year=year,
                k_value=k_value,
                baseline=baseline,
                top_n=top_n
            )

        if direct_ranking_mode == 'tune_lambda_by_context':
            return experiment.tune_lambda_by_context(
                race_class=race_class,
                year=year,
                scheme=schemes[0],
                k_value=k_value,
                top_n=top_n
            )

        if direct_ranking_mode == 'marginal':
            return experiment.run_marginal_teammate_contribution_analysis(
                race_class=race_class,
                year=year,
                scheme=schemes[0],
                k_penalty=k_penalty,
                lambda_value=lambda_value,
                k_value=k_value,
                max_teammates=max_teammates
            )

        raise ValueError(f"Invalid direct ranking mode: {direct_ranking_mode}")
        
    except Exception as e:
        print(f"❌ Error in direct ranking: {e}")
        raise


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(
        description="Unified Roster Ranking Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train base models only (roster level)
  python run_experiments.py --race_class all --pipeline base_only
  
  # Train base models only (rider level)
  python run_experiments.py --race_class all --pipeline base_only --level rider
  
  # Train base models + simple ensemble (roster level)
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods simple_average
  
  # Train base models + simple ensemble (rider level)
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods simple_average --level rider
  
  # Train base models + adaptive feature selection ensemble
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods adaptive_feature_selection
  
  # Train base models + ensemble with cluster-only optimization
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods static_moe --optimization_strategy cluster_only
  
  # Train base models + multiple ensemble methods
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods simple_average meta_learning static_moe
  
  # Train base models + all ensemble methods
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods simple_average meta_learning static_moe scheme_specialists adaptive_feature_selection hard_moe_gating soft_moe_gating
  
  # Train base models + MoE gating methods only
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods hard_moe_gating soft_moe_gating
  
  # Train base models + MoE gating methods with MLP gating network
  python run_experiments.py --race_class all --pipeline base_ensemble --ensemble_methods hard_moe_gating soft_moe_gating --gating_model_type mlp
  
  # Extract rider features
  python run_experiments.py --race_class all --pipeline extract_rider_features --time_gap 30
  
  # Extract TrueSkill features with time_lag scheme
  python run_experiments.py --race_class all --pipeline extract_trueskill_features --scheme time_lag
  
  # Extract TrueSkill features with leader scheme  
  python run_experiments.py --race_class WT --pipeline extract_trueskill_features --scheme leader
  
  # Run feature importance analysis with all methods (roster level)
  python run_experiments.py --race_class all --pipeline feature_importance
  
  # Run feature importance analysis with specific methods (rider level) 
  python run_experiments.py --race_class all --pipeline feature_importance --level rider --importance_methods permutation shap
  
  # Run feature importance analysis with permutation importance only
  python run_experiments.py --race_class all --pipeline feature_importance --importance_methods permutation
  
  # Run feature importance analysis with time gap
  python run_experiments.py --race_class all --pipeline feature_importance --time_gap 30
  
  # Use custom year and force retuning
  python run_experiments.py --race_class all --pipeline base_only --year 2024 --force_retune
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--race_class',
        required=True,
        choices=['all', 'WT'],
        help='Race class to experiment with'
    )
    
    # Pipeline control
    parser.add_argument(
        '--pipeline',
        choices=['base_only', 'base_ensemble', 'extract_rider_features', 'extract_trueskill_features', 'direct_ranking', 'feature_importance', 'filter_race_results'],
        default='base_only',
        help='Pipeline to run (default: base_only)'
    )
    
    parser.add_argument(
        '--level',
        choices=['roster', 'rider'],
        default='rider',
        help='Level of ranking (default: rider)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--ensemble_methods',
        nargs='+',
        choices=['simple_average', 'meta_learning', 'static_moe', 'scheme_specialists', 'adaptive_feature_selection', 'hard_moe_gating', 'soft_moe_gating'],
        default=['simple_average'],
        help='Ensemble methods for base_ensemble pipeline (default: simple_average)'
    )
    
    parser.add_argument(
        '--gating_model_type',
        choices=['logistic', 'mlp'],
        default='logistic',
        help='Type of gating model for MoE methods (default: logistic)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='Year for test set (default: 2023)'
    )
    
    parser.add_argument(
        '--force_retune',
        action='store_true',
        help='Force hyperparameter retuning even if saved hyperparams exist'
    )
    
    parser.add_argument(
        '--k_value',
        type=int,
        default=10,
        help='K value for NDCG@k evaluation (default: 10)'
    )
    
    parser.add_argument(
        '--optimization_strategy',
        choices=['cluster_race_class', 'cluster_only'],
        default=None,
        help='Strategy for optimization ensmble performance by Cluster / Cluster & Race Class (default: cluster_race_class)'
    )
    
    parser.add_argument(
        '--scheme',
        nargs='+',
        choices=['time_lag', 'equal_weight', 'rank_norm', 'leader'],
        default=['time_lag'],
        help='Scheme for TrueSkill feature extraction (default: time_lag)'
    )
    
    parser.add_argument(
        '--time_gap',
        type=int,
        default=None,
        help='Time gap in days before race start date (for feature extraction, prediction, etc.)'
    )

    parser.add_argument(
        '--min_team_size',
        type=int,
        default=MIN_TEAM_SIZE,
        help='Minimum riders per team per race (default: config MIN_TEAM_SIZE)'
    )

    parser.add_argument(
        '--min_teams_per_race',
        type=int,
        default=MIN_TEAMS_PER_RACE,
        help='Minimum teams per race to keep (default: config MIN_TEAMS_PER_RACE)'
    )


    parser.add_argument(
        '--target_years',
        nargs='+',
        type=int,
        default=None,
        help='Target years for rider feature extraction (e.g., 2024 2025)'
    )
    
    parser.add_argument(
        '--importance_methods',
        nargs='+',
        choices=['permutation', 'lofo', 'shap'],
        default=None,
        help='Feature importance methods to run (default: all enabled)'
    )

    parser.add_argument(
        '--exp_name',
        default='class_features',
        help='The Directory name of the experiment to save outputs(default: class_features)'
    )

    parser.add_argument(
        '--direct_ranking_mode',
        choices=['evaluation', 'tune_lambda_by_context', 'marginal'],
        default='evaluation',
        help='Direct ranking functionality to run (default: evaluation)'
    )
    parser.add_argument(
        '--lambda_value',
        type=float,
        default=None,
        help='Lambda value for marginal analysis (default: scheme-based)'
    )
    parser.add_argument(
        '--k_penalty',
        type=float,
        default=None,
        help='K penalty for marginal analysis (default: scheme-based)'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=None,
        help='Use top N teammates when averaging (default: all)'
    )
    parser.add_argument(
        '--max_teammates',
        type=int,
        default=None,
        help='Max teammates to evaluate in marginal analysis (default: 8)'
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print_header("ROSTER RANKING EXPERIMENT CONFIGURATION")
    print(f"Race class: {args.race_class}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Level: {args.level}")
    print(f"Year: {args.year}")
    if args.pipeline in ['base_only', 'base_ensemble']:
        print(f"Optimization strategy: {args.optimization_strategy}")
    if args.pipeline == 'base_ensemble':
        print(f"Ensemble methods: {args.ensemble_methods}")
    if args.pipeline == 'extract_trueskill_features':
        print(f"Scheme: {args.scheme}")
    if args.pipeline == 'extract_rider_features':
        print(f"Time gap: {args.time_gap} days")
    if args.pipeline == 'feature_importance':
        print(f"Feature importance methods: {args.importance_methods or 'all enabled'}")
        if args.time_gap:
            print(f"Time gap: {args.time_gap} days")
    if args.pipeline == 'direct_ranking':
        print(f"Direct ranking mode: {args.direct_ranking_mode}")
        if args.top_n is not None:
            print(f"Top N teammates: {args.top_n}")
        if args.k_penalty is not None:
            print(f"K penalty: {args.k_penalty}")
        if args.lambda_value is not None:
            print(f"Lambda value: {args.lambda_value}")
        if args.max_teammates is not None:
            print(f"Max teammates: {args.max_teammates}")
    if args.pipeline == 'filter_race_results':
        print(f"Min team size in race: {args.min_team_size}")
        print(f"Min teams per race: {args.min_teams_per_race}")
    if args.pipeline == 'base_only' or args.pipeline == 'base_ensemble':
        print(f"Experiment name: {args.exp_name}")
    
    start_time = time.time()
    
    try:
        # Run appropriate pipeline
        if args.pipeline == 'base_only':
            base_experiment = run_base_models_pipeline(race_class=args.race_class, force_retune=args.force_retune, year=args.year, k_value=args.k_value, time_gap=args.time_gap, save_results=True, level=args.level, schemes=args.scheme, optimization_strategy=args.optimization_strategy, exp_name=args.exp_name)
            
        elif args.pipeline == 'base_ensemble':
            base_experiment, ensemble_experiment = run_ensemble_pipeline(
                race_class=args.race_class, ensemble_methods=args.ensemble_methods, force_retune=args.force_retune, year=args.year, k_value=args.k_value, level=args.level, time_gap=args.time_gap, optimization_strategy=args.optimization_strategy, gating_model_type=args.gating_model_type, exp_name=args.exp_name
            )
        
        elif args.pipeline == 'extract_rider_features':
            output_path = run_rider_features_pipeline(
                race_class=args.race_class, time_gap=args.time_gap, target_years=args.target_years
            )
            
        elif args.pipeline == 'extract_trueskill_features':
            output_path = run_trueskill_features_pipeline(
                race_class=args.race_class, schemes=args.scheme, time_gap=args.time_gap, target_years=args.target_years
            )
        
        elif args.pipeline == 'filter_race_results':
            output_path = run_race_results_filtering_pipeline(
                min_team_size=args.min_team_size,
                min_teams_per_race=args.min_teams_per_race,
            )
        
        elif args.pipeline == 'direct_ranking':
            output_path = run_direct_ranking_pipeline(
                race_class=args.race_class,
                schemes=args.scheme,
                year=args.year,
                k_value=args.k_value,
                direct_ranking_mode=args.direct_ranking_mode,
                k_penalty=args.k_penalty,
                lambda_value=args.lambda_value,
                top_n=args.top_n,
                max_teammates=args.max_teammates
            )
        
        elif args.pipeline == 'feature_importance':
            feature_importance_experiment = run_feature_importance_pipeline(
                race_class=args.race_class, methods=args.importance_methods, schemes=args.scheme, year=args.year, k_value=args.k_value, time_gap=args.time_gap, level=args.level, force_retune=args.force_retune
            )
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print_header("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Pipeline: {args.pipeline}")
        print(f"Race class: {args.race_class}")
        if args.pipeline not in ['extract_rider_features', 'extract_trueskill_features']:
            print(f"Level: {args.level}")
            print(f"Year: {args.year}")
            print(f"K value: {args.k_value}")

        if args.pipeline == 'base_ensemble':
            print("\nBase models and ensemble pipeline completed successfully!")
        elif args.pipeline == 'fusion':
            print("\nFusion pipeline completed successfully!")
        elif args.pipeline == 'extract_rider_features':
            print(f"\nRider features extraction completed successfully!")
            print(f"Output saved to: {output_path}")
        elif args.pipeline == 'extract_trueskill_features':
            print(f"\nTrueSkill features extraction completed successfully!")
            print(f"Scheme: {args.scheme}")
            print(f"Output saved to: {output_path}")
        elif args.pipeline == 'feature_importance':
            print(f"\nFeature importance analysis completed successfully!")
            print(f"Methods: {args.importance_methods or 'all enabled'}")
        elif args.pipeline == 'filter_race_results':
            print(f"\nRace results filtering completed successfully!")
            print(f"Output saved to: {output_path}")
            print(f"Level: {args.level}")
            print(f"Results saved to: {feature_importance_experiment.output_dir}")
        else:
            print("\nBase models completed successfully!")
            
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 