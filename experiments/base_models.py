"""
Base Models Experiments Module

This module handles training and evaluation of individual base models for each scheme.
Base models form the foundation for ensemble and fusion methods.
"""

import os
import json
import time
import pandas as pd
from ..utils.config import *
from ..data.loaders import load_or_create_dataset
from ..core.models import tune_hyperparameters, train_model, evaluate_model, save_model_results, get_feature_importance
from ..core.metrics import calculate_summary_statistics


class BaseModelExperiment:
    """
    Manages base model experiments for roster or rider ranking
    
    Handles training individual models for each scheme (time_lag, equal_weight, rank_norm)
    with automatic hyperparameter tuning and comprehensive evaluation.
    """
    
    def __init__(self, race_class, schemes=None, custom_output_dir=None, custom_hyperparams_dir=None, year=2023, ensemble_type=None, k_value=5, time_gap=None, level='roster', optimization_strategy='cluster_race_class', exp_name='class_features'):
        """
        Initialize base model experiment
        
        Args:
            race_class (str): Race class ('all' or 'WT')
            schemes (list): List of schemes to train (default: ['time_lag', 'equal_weight', 'rank_norm'])
            custom_output_dir (str): Custom output directory (overrides default)
            custom_hyperparams_dir (str): Custom hyperparameters directory (overrides default)
            year (int): Year for results (default: 2023)
            level (str): Level of ranking ('roster' or 'rider')
            optimization_strategy (str): Strategy for cluster performance optimization:
                - 'cluster_race_class': Use cluster+race_class combinations (default)
                - 'cluster_only': Use cluster-only combinations
        """
        self.race_class = race_class
        self.schemes = schemes or AVAILABLE_SCHEMES
        self.year = year
        self.level = level
        self.time_gap = time_gap
        self.optimization_strategy = optimization_strategy
        self.base_models = {}
        self.base_results = {}
        self.hyperparameters = {}
        self.base_feature_importance = {}  # scheme -> feature importance DataFrame
        self.datasets = {}  # scheme -> {'train', 'val', 'test', 'train_val', 'full'}
        self.feature_columns_dict = {}  # scheme -> feature_columns
        self.ensemble_type = ensemble_type
        self.k_value = k_value
        self.exp_name = exp_name
        # Cluster-specific validation performance for ensemble expert selection
        self.cluster_performance = {}  # cluster -> scheme -> validation_score
        
        # Race type × scheme performance matrix for scheme specialists ensemble
        self.race_type_scheme_matrix = {}  # race_type -> scheme -> {ndcg@3, ndcg@5, recall@3, recall@5}
        
        # Race type-specific hyperparameters for specialists
        self.race_type_hyperparameters = {}  # race_type -> scheme -> hyperparameters
        
        # Set up output directories using configurable paths
        if custom_output_dir:
            self.results_dir = custom_output_dir
        else:
            self.results_dir = get_output_dir('base_models', race_class, level=level)
        if custom_hyperparams_dir:
            self.hyperparams_dir = custom_hyperparams_dir
        else:
            self.hyperparams_dir = get_hyperparams_dir(race_class, k_value=k_value, level=level)
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.hyperparams_dir, exist_ok=True)
    
    def load_saved_hyperparameters(self, exp_name, scheme):
        """
        Load saved hyperparameters for a scheme if they exist
        
        Args:
            exp_name (str): Experiment name
            
        Returns:
            dict or None: Saved hyperparameters or None if not found
        """
        if self.time_gap:
            hyperparams_file = f"{self.hyperparams_dir}/{exp_name}/{scheme}_time_gap_{self.time_gap}_best_hyperparams.json"
        else:
            hyperparams_file = f"{self.hyperparams_dir}/{exp_name}/{scheme}_best_hyperparams.json"
        if os.path.exists(hyperparams_file):
            try:
                with open(hyperparams_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load hyperparameters for {exp_name}: {e}")
        
        return None
    
    def save_hyperparameters(self, exp_name, scheme, best_params):
        """
        Save hyperparameters for future use
        
        Args:
            scheme (str): Scheme name
            best_params (dict): Best hyperparameters
        """
        if self.time_gap:
            hyperparams_file = f"{self.hyperparams_dir}/{exp_name}/{scheme}_time_gap_{self.time_gap}_best_hyperparams.json"
        else:
            hyperparams_file = f"{self.hyperparams_dir}/{exp_name}/{scheme}_best_hyperparams.json"
        os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)
        try:
            with open(hyperparams_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"✓ Hyperparameters saved for {exp_name}")
        except Exception as e:
            print(f"Warning: Could not save hyperparameters for {exp_name}: {e}")
    
    def train_single_model(self, scheme, force_retune=False, k_value=10, save_results=False):
        """
        Train a single base model for a given scheme
        
        Args:
            scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm')
            force_retune (bool): Force hyperparameter retuning even if saved params exist
            
        Returns:
            tuple: (model, test_results, hyperparameters)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING BASE MODEL: {scheme.upper()}")
        print(f"{'='*80}")
        print(f"Race class: {self.race_class}")
        print(f"Scheme: {scheme}")
        print(f"K value: {k_value}")
        start_time = time.time()
        
        # === STEP 1: Load dataset ===
        print(f"\n1. LOADING DATASET...")
        print("-" * 50)
        print(f"Level: {self.level}")
        dataset_df = load_or_create_dataset(self.race_class, scheme, level=self.level, time_gap=self.time_gap, exp_name=self.exp_name)
        print(f"✓ Dataset loaded: {len(dataset_df)} records")
        print(f"✓ Date range: {dataset_df['date'].min()} to {dataset_df['date'].max()}")
        
        # === STEP 2: Split data ===
        print(f"\n2. SPLITTING DATA...")
        print("-" * 50)
        
        train_df = dataset_df[dataset_df['year'] < self.year - 1].copy()
        val_df = dataset_df[dataset_df['year'] == self.year - 1].copy()
        test_df = dataset_df[dataset_df['year'] == self.year].copy()
        
        print(f"✓ Train: {len(train_df)} records")
        print(f"✓ Validation: {len(val_df)} records")
        print(f"✓ Test: {len(test_df)} records")
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("Insufficient data in one or more splits")
        
        # === STEP 3: Prepare features ===
        print(f"\n3. PREPARING FEATURES...")
        print("-" * 50)
        
        feature_columns = [col for col in dataset_df.columns if col not in EXCLUDE_COLS]
        print(f"✓ Feature columns: {len(feature_columns)} features")
        
        # === STEP 4: Hyperparameter tuning ===
        print(f"\n4. HYPERPARAMETER TUNING...")
        print("-" * 50)
        
        # Check for saved hyperparameters
        saved_params = self.load_saved_hyperparameters(self.exp_name, scheme) if not force_retune else None
        if saved_params is not None:
            print(f"✓ Using saved hyperparameters: {saved_params}")
            best_params = saved_params
            best_ndcg = 0  # We don't save the validation score
        else:
            print("Tuning hyperparameters...")
            best_params, best_ndcg = tune_hyperparameters(train_df, val_df, feature_columns, k_value=k_value, level=self.level)
            print(f"✓ Best hyperparameters: {best_params}")
            print(f"✓ Best validation NDCG@{k_value}: {best_ndcg:.4f}")
            
            # Save hyperparameters for future use
            self.save_hyperparameters(self.exp_name, scheme, best_params)
        
        if self.ensemble_type and ('static_moe' in self.ensemble_type or 'adaptive_feature_selection' in self.ensemble_type):
            print(f"✓ Evaluating static MOE cluster performance")
            # Use dedicated method for static MOE cluster performance evaluation
            unique_comb = [(cluster, race_class) for cluster, race_class in dataset_df[['race_class', 'cluster']].drop_duplicates().values]
            self.evaluate_static_moe_cluster_performance(scheme, train_df, val_df, feature_columns, best_params, unique_comb)
        if self.ensemble_type and 'scheme_specialists' in self.ensemble_type:
            print(f"✓ Building race type × scheme performance matrix")
            # Use dedicated method for building race type × scheme performance matrix
            self.build_scheme_specialists_matrix(scheme, train_df, val_df, feature_columns)
        
        # === STEP 6: Train final model ===
        print(f"\n6. TRAINING FINAL MODEL...")
        print("-" * 50)
        
        # Combine train + validation for final model
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        
        model = train_model(full_train_df, feature_columns, best_params, f"{scheme}_model", k_value=k_value, level=self.level)
        print(f"✓ Final model trained")
        
        # === STEP 7: Evaluate on test set ===
        print(f"\n7. EVALUATION ON TEST SET...")
        print("-" * 50)
        
        test_results = evaluate_model(model, test_df, feature_columns, f"{scheme}_model", k_value=k_value, level=self.level)
        print(f"✓ Test evaluation completed: {len(test_results)} races")
        
        # === STEP 8: Extract feature importance ===
        print(f"\n8. EXTRACTING FEATURE IMPORTANCE...")
        print("-" * 50)
        
        try:
            feature_importance_df = get_feature_importance(model, feature_columns, importance_type='gain')
            print(f"✓ Feature importance extracted ({len(feature_importance_df)} features)")
            
            # Print top 5 features for logging
            print(f"Top 5 most important features:")
            for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows()):
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
                
        except Exception as e:
            print(f"⚠ Feature importance extraction failed: {e}")
            feature_importance_df = None
        
        # === STEP 9: Calculate summary statistics ===
        print(f"\n9. CALCULATING SUMMARY STATISTICS...")
        print("-" * 50)
        
        summary_stats = calculate_summary_statistics(test_results)
        
        for metric, stats in summary_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # === STEP 10: Save results ===
        print(f"\n10. SAVING RESULTS...")
        print("-" * 50)
        
        if save_results:
            if self.time_gap:
                scheme_output_dir = f"{self.results_dir}/{self.exp_name}/{self.time_gap}_days/{scheme}"
            else:
                scheme_output_dir = f"{self.results_dir}/{self.exp_name}/{scheme}"
            save_model_results(test_results, scheme, self.race_class, scheme_output_dir, summary_stats, feature_importance_df)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n✓ {scheme.upper()} model completed successfully!")
        print(f"  Mean NDCG@{self.k_value}: {summary_stats.get(f'NDCG@{self.k_value}', {}).get('mean', 0):.4f}")
        print(f"  Execution time: {execution_time:.1f} seconds")
        
        # Store results
        self.base_models[scheme] = model
        self.base_results[scheme] = test_results
        self.hyperparameters[scheme] = best_params
        if feature_importance_df is not None:
            self.base_feature_importance[scheme] = feature_importance_df
        
        # Store datasets and feature columns for feature importance analysis
        self.datasets[scheme] = {
            'full': dataset_df,
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'train_val': full_train_df
        }
        self.feature_columns_dict[scheme] = feature_columns
        
        return model, test_results, best_params
    
    def train_all_models(self, force_retune=False, k_value=10, save_results=False):
        """
        Train all base models for all schemes
        
        Args:
            force_retune (bool): Force hyperparameter retuning for all schemes
            
        Returns:
            dict: Dictionary mapping scheme names to (model, results, hyperparams) tuples
        """
        print(f"\n{'='*80}")
        print(f"TRAINING ALL BASE MODELS")
        print(f"{'='*80}")
        print(f"Race class: {self.race_class}")
        print(f"Schemes: {self.schemes}")
        print(f"Experiment name: {self.exp_name}")
        
        all_results = {}
        
        for scheme in self.schemes:
            try:
                model, results, hyperparams = self.train_single_model(scheme=scheme, force_retune=force_retune, k_value=k_value, save_results=save_results)
                all_results[scheme] = (model, results, hyperparams)
            except Exception as e:
                print(f"❌ Error training {scheme} model: {e}")
                raise e
        
        # Save comprehensive comparison
        if save_results:
            # self.save_base_models_comparison()
            # Automatically save cluster performance after evaluation
            self.save_cluster_performance()
            print(f"✓ Cluster performance saved automatically")
        
        print(f"\n{'='*80}")
        print(f"ALL BASE MODELS COMPLETED")
        print(f"{'='*80}")
        print(f"Successful models: {list(all_results.keys())}")
        
        return all_results
    
    def load_existing_scheme_results(self):
        """
        Load results from existing scheme directories that have summary files
        
        Returns:
            dict: Dictionary mapping scheme names to their summary data
        """
        existing_results = {}
        
        if not os.path.exists(self.results_dir):
            return existing_results
        
        print(f"\n🔍 Scanning for existing scheme results in: {self.results_dir}")
        
        # Look for scheme directories
        for item in os.listdir(self.results_dir):
            scheme_dir = os.path.join(self.results_dir, item)
            
            # Skip if not a directory
            if not os.path.isdir(scheme_dir):
                continue
            
            # Skip if it's a known non-scheme directory
            if item in ['hyperparameters', 'plots', 'analysis']:
                continue
            
            # Look for summary file in this directory
            summary_file = os.path.join(scheme_dir, f"{item}_summary.csv")
            
            if os.path.exists(summary_file):
                try:
                    print(f"  📁 Found {item} scheme results")
                    
                    # Load summary CSV
                    summary_df = pd.read_csv(summary_file)
                    
                    # Convert to the format we need for comparison
                    scheme_data = {
                        'Scheme': item,
                        'Race_Class': self.race_class,
                        'Source': 'Existing'  # Mark as existing result
                    }
                    
                    # Extract metrics from summary file
                    for _, row in summary_df.iterrows():
                        metric = row['Metric']
                        scheme_data[f'{metric}_mean'] = row['mean']
                        scheme_data[f'{metric}_std'] = row['std'] 
                        scheme_data[f'{metric}_count'] = row['count']
                        
                        # Also add min/max if available
                        if 'min' in row:
                            scheme_data[f'{metric}_min'] = row['min']
                        if 'max' in row:
                            scheme_data[f'{metric}_max'] = row['max']
                    
                    existing_results[item] = scheme_data
                    print(f"    ✓ Loaded {len(summary_df)} metrics for {item}")
                    
                except Exception as e:
                    print(f"    ⚠ Could not load {item} summary: {e}")
                    continue
            else:
                print(f"  📁 Found {item} directory but no summary file")
        
        print(f"✓ Found {len(existing_results)} existing scheme results")
        return existing_results
    
    def save_base_models_comparison(self):
        """
        Save a comprehensive comparison of all base models performance,
        including both current session results and existing scheme results
        """
        comparison_data = []
        current_schemes = set()
        
        # First, add results from current session
        if self.base_results:
            print(f"\n📊 Adding current session results...")
            for scheme, results_df in self.base_results.items():
                summary_stats = calculate_summary_statistics(results_df)
                
                row = {
                    'Scheme': scheme,
                    'Race_Class': self.race_class,
                    'Source': 'Current'  # Mark as current session result
                }
                
                # Add all metrics
                for metric, stats in summary_stats.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        row[f'{metric}_mean'] = stats['mean']
                        row[f'{metric}_std'] = stats['std']
                        row[f'{metric}_count'] = stats['count']
                        if 'min' in stats:
                            row[f'{metric}_min'] = stats['min']
                        if 'max' in stats:
                            row[f'{metric}_max'] = stats['max']
                
                comparison_data.append(row)
                current_schemes.add(scheme)
                print(f"  ✓ Added {scheme} from current session")
        
        # Then, add results from existing scheme directories
        existing_results = self.load_existing_scheme_results()
        
        for scheme, scheme_data in existing_results.items():
            # Skip if we already have this scheme from current session
            if scheme in current_schemes:
                print(f"  ⚠ Skipping {scheme} (already in current session)")
                continue
            
            comparison_data.append(scheme_data)
            print(f"  ✓ Added {scheme} from existing results")
        
        if not comparison_data:
            print("⚠ No base model results to compare")
            return
        
        # Create comprehensive comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by NDCG@5 if available
        if 'NDCG@5_mean' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('NDCG@5_mean', ascending=False)
        
        # Save comparison
        comparison_file = f"{self.results_dir}/base_models_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\n✓ Comprehensive base models comparison saved to: {comparison_file}")
        print(f"  📊 Total schemes: {len(comparison_df)}")
        print(f"  🔄 Current session: {len(current_schemes)}")
        print(f"  📁 Existing results: {len(existing_results)}")
        
        # Print summary
        print(f"\nComprehensive Base Models Performance Summary:")
        print("-" * 70)
        for _, row in comparison_df.iterrows():
            ndcg5 = row.get('NDCG@5_mean', 0)
            recall5 = row.get('Recall@5_mean', 0)
            source = row.get('Source', 'Unknown')
            source_icon = "🔄" if source == 'Current' else "📁"
            print(f"  {source_icon} {row['Scheme']:<15}: NDCG@5={ndcg5:.4f}, Recall@5={recall5:.4f} ({source})")

    def load_cluster_performance(self):
        """
        Load cluster performance data from saved JSON file
        """
        output_dir = get_output_dir('ensemble', race_class=self.race_class, level=self.level, k_value=self.k_value)
        if self.time_lag:
            cluster_perf_file = f"{output_dir}/cluster_performance_{self.optimization_strategy}_time_lag_{self.time_lag}.json"
        else:
            cluster_perf_file = f"{output_dir}/cluster_performance_{self.optimization_strategy}.json"
        
        if not os.path.exists(cluster_perf_file):
            print("⚠ No saved cluster performance file found")
            return False
        
        try:
            with open(cluster_perf_file, 'r') as f:
                json_data = json.load(f)
            
            # Convert string keys back to tuples
            self.cluster_performance = {}
            for key_str, scheme_scores in json_data.items():
                if '|' in key_str:
                    # Convert "cluster|race_class" back to tuple (cluster, race_class)
                    cluster, race_class = key_str.split('|', 1)
                    cluster_race_key = (cluster, race_class)
                else:
                    # Handle legacy single cluster keys
                    cluster_race_key = key_str
                self.cluster_performance[cluster_race_key] = scheme_scores
            
            print(f"✓ Cluster performance loaded from: {cluster_perf_file}")
            print(f"✓ Loaded {len(self.cluster_performance)} combinations ({self.optimization_strategy})")
            return True
        except Exception as e:
            print(f"Warning: Could not load cluster performance: {e}")
            raise e

    def save_cluster_performance(self):
        """
        Save cluster performance data for ensemble expert selection
        """
        if not self.cluster_performance:
            print("⚠ No cluster performance data to save")
            return
        output_dir = get_output_dir('ensemble', race_class=self.race_class, level=self.level, k_value=self.k_value)
        if self.time_lag:
            cluster_perf_file = f"{output_dir}/cluster_performance_{self.optimization_strategy}_time_lag_{self.time_lag}.json"
        else:
            cluster_perf_file = f"{output_dir}/cluster_performance_{self.optimization_strategy}.json"
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Convert tuple keys to strings for JSON serialization
            json_serializable_data = {}
            for cluster_race_key, scheme_scores in self.cluster_performance.items():
                if isinstance(cluster_race_key, tuple):
                    # Convert tuple (cluster, race_class) to string "cluster|race_class"
                    key_str = f"{cluster_race_key[0]}|{cluster_race_key[1]}"
                else:
                    # Handle legacy single cluster keys
                    key_str = str(cluster_race_key)
                json_serializable_data[key_str] = scheme_scores
            
            with open(cluster_perf_file, 'w') as f:
                json.dump(json_serializable_data, f, indent=2)
            
            print(f"✓ Cluster performance saved to: {cluster_perf_file}")
            
            # Print summary
            print(f"\nCluster Performance Summary ({self.optimization_strategy}):")
            print("-" * 60)
            for cluster_race_key, scheme_scores in self.cluster_performance.items():
                if isinstance(cluster_race_key, tuple):
                    display_key = f"{cluster_race_key[0]} + {cluster_race_key[1]}"
                else:
                    display_key = str(cluster_race_key)
                print(f"\n  {display_key}:")
                sorted_schemes = sorted(scheme_scores.items(), key=lambda x: x[1], reverse=True)
                for i, (scheme, score) in enumerate(sorted_schemes):
                    marker = "🏆" if i == 0 else "  "
                    print(f"    {marker} {scheme:<15}: {score:.4f}")
                    
        except Exception as e:
            print(f"Warning: Could not save cluster performance: {e}")
    
    def save_race_type_scheme_matrix(self):
        """
        Save race type × scheme performance matrix for scheme specialists ensemble
        """
        if not self.race_type_scheme_matrix:
            print("⚠ No race type × scheme matrix to save")
            return
        
        matrix_file = f"{self.results_dir}/race_type_scheme_matrix.json"
        
        try:
            with open(matrix_file, 'w') as f:
                json.dump(self.race_type_scheme_matrix, f, indent=2)
            
            print(f"✓ Race type × scheme matrix saved to: {matrix_file}")
            
            # Print summary table
            print(f"\nRace Type × Scheme Performance Matrix (NDCG@5):")
            print("-" * 80)
            
            # Get all schemes
            all_schemes = set()
            for race_type_data in self.race_type_scheme_matrix.values():
                all_schemes.update(race_type_data.keys())
            all_schemes = sorted(list(all_schemes))
            
            # Print header
            header = f"{'Race Type':<25}"
            for scheme in all_schemes:
                header += f"{scheme:<15}"
            header += "Best Scheme"
            print(header)
            print("-" * len(header))
            
            # Print each race type
            for race_type, scheme_data in self.race_type_scheme_matrix.items():
                row = f"{race_type:<25}"
                best_score = -1
                best_scheme = None
                
                for scheme in all_schemes:
                    if scheme in scheme_data:
                        score = scheme_data[scheme]['ndcg@5']
                        row += f"{score:<15.4f}"
                        if score > best_score:
                            best_score = score
                            best_scheme = scheme
                    else:
                        row += f"{'N/A':<15}"
                
                row += f"{best_scheme or 'N/A'}"
                print(row)
                
        except Exception as e:
            print(f"Warning: Could not save race type × scheme matrix: {e}")
    
    def save_race_type_hyperparameters(self):
        """
        Save all race type-specific hyperparameters to consolidated file
        """
        consolidated_file = f"{self.hyperparams_dir}/race_type_hyperparameters.json"
        
        try:
            with open(consolidated_file, 'w') as f:
                json.dump(self.race_type_hyperparameters, f, indent=2)
            print(f"✓ Race type hyperparameters saved to: {consolidated_file}")
        except Exception as e:
            print(f"⚠ Could not save consolidated race type hyperparameters: {e}")

    def evaluate_static_moe_cluster_performance(self, scheme, train_df, val_df, feature_columns, best_params, unique_comb):
        """
        Evaluate cluster performance for static mixture of experts ensemble selection
        
        Args:
            scheme (str): Current scheme name
            train_df (DataFrame): Training data
            val_df (DataFrame): Validation data  
            feature_columns (list): Feature column names
            best_params (dict): Best hyperparameters for the model
        """
        print(f"\n5. EVALUATING CLUSTER PERFORMANCE FOR STATIC MOE ({self.optimization_strategy})...")
        print("-" * 50)

        # Check if cluster performance file exists
        if self.load_cluster_performance():            
            return            
        
        # Train model on training data only (not train+val) for cluster evaluation
        train_only_model = train_model(train_df, feature_columns, best_params, f"{scheme}_cluster_eval", k_value=self.k_value, level=self.level)
        print(f"✓ Train-only model created for cluster evaluation")
        
        # Evaluate on validation data split by cluster
        val_results = evaluate_model(train_only_model, val_df, feature_columns, f"{scheme}_cluster_eval", k_value=self.k_value, level=self.level)
        print(f"✓ Validation evaluation completed: {len(val_results)} races")
        
        # Use optimization strategy to determine how to group performance
        if self.optimization_strategy == 'cluster_race_class':
            # Group by cluster and race_class and calculate performance
            for (race_class, cluster) in unique_comb:
                cluster_race_data = val_results[(val_results['cluster'] == cluster) & (val_results['race_class'] == race_class)]
                if len(cluster_race_data) == 0:
                    # Fall down to cluster & one/multi day race_class
                    cluster_race_data = val_results[(val_results['cluster'] == cluster) & (val_results['race_class'].str.startswith(race_class[0]))]
                    if len(cluster_race_data) == 0:
                        raise ValueError(f"No data found for {cluster} + {race_class}")
                    print(f'No data found for {cluster} + {race_class}, using {cluster} + {race_class[0]} event type')
                
                cluster_stats = calculate_summary_statistics(cluster_race_data)
                ndcg_mean = cluster_stats.get(f'NDCG@{self.k_value}', {}).get('mean', 0.0)
                
                # Store cluster-race_class performance using tuple key
                cluster_race_key = (cluster, race_class)
                if cluster_race_key not in self.cluster_performance:
                    self.cluster_performance[cluster_race_key] = {}
                self.cluster_performance[cluster_race_key][scheme] = ndcg_mean
                
                print(f"  {cluster} + {race_class}: NDCG@{self.k_value} = {ndcg_mean:.4f} ({len(cluster_race_data)} races)")
        
        elif self.optimization_strategy == 'cluster_only':
            # Group by cluster only and calculate performance
            unique_clusters = val_results['cluster'].unique()
            for cluster in unique_clusters:
                cluster_data = val_results[val_results['cluster'] == cluster]
                if len(cluster_data) == 0:
                    print(f"⚠ No data found for cluster {cluster}")
                    continue
                
                cluster_stats = calculate_summary_statistics(cluster_data)
                ndcg_mean = cluster_stats.get(f'NDCG@{self.k_value}', {}).get('mean', 0.0)
                
                # Store cluster-only performance using cluster key
                cluster_key = cluster
                if cluster_key not in self.cluster_performance:
                    self.cluster_performance[cluster_key] = {}
                self.cluster_performance[cluster_key][scheme] = ndcg_mean
                
                print(f"  {cluster}: NDCG@{self.k_value} = {ndcg_mean:.4f} ({len(cluster_data)} races)")
        
        else:
            raise ValueError(f"Invalid optimization_strategy: {self.optimization_strategy}")
    
    def load_race_type_hyperparameters(self, race_type, scheme):
        """
        Load hyperparameters for a specific race type and scheme combination
        
        Args:
            race_type (str): Race type/cluster name
            scheme (str): Scheme name
            
        Returns:
            dict: Hyperparameters if found, None otherwise
        """
        # Try to load from consolidated race type hyperparameters file
        consolidated_hyperparams_file = f"{self.hyperparams_dir}/race_type_hyperparameters.json"
        
        if os.path.exists(consolidated_hyperparams_file):
            print(f"    🔍 Loading from consolidated race type hyperparameters...")
            try:
                with open(consolidated_hyperparams_file, 'r') as f:
                    consolidated_data = json.load(f)
                
                # Extract hyperparameters for this specific race_type + scheme combination
                if (race_type in consolidated_data and 
                    scheme in consolidated_data[race_type]):
                    race_type_best_params = consolidated_data[race_type][scheme]
                    print(f"    ✓ Found hyperparameters in consolidated file for {race_type} + {scheme}")
                    print(f"    ✓ Using saved hyperparameters: {race_type_best_params}")
                    return race_type_best_params
                else:
                    print(f"    ⚠ No hyperparameters found in consolidated file for {race_type} + {scheme}")
            except Exception as e:
                print(f"    ⚠ Could not load consolidated hyperparameters: {e}")
                raise
        
        return None
    
    def save_race_type_hyperparameters_individual(self, scheme, race_type, best_params):
        """
        Save hyperparameters for a specific race type and scheme to individual file
        
        Args:
            scheme (str): Scheme name
            race_type (str): Race type/cluster name
            best_params (dict): Best hyperparameters to save
        """
        race_type_hyperparams_file = f"{self.hyperparams_dir}/{scheme}_{race_type}_hyperparams.json"
        try:
            with open(race_type_hyperparams_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"    ✓ Race type hyperparameters saved to individual file")
        except Exception as e:
            print(f"    ⚠ Could not save individual race type hyperparameters: {e}")
    
    def train_scheme_specialist(self, scheme, race_type, race_type_train, race_type_val, feature_columns):
        """
        Train a specialist model for a specific scheme and race type combination
        
        Args:
            scheme (str): Scheme name
            race_type (str): Race type/cluster name
            race_type_train (DataFrame): Training data for this race type
            race_type_val (DataFrame): Validation data for this race type
            feature_columns (list): Feature column names
            
        Returns:
            dict: Specialist performance metrics or None if failed
        """
        print(f"\n  Training {scheme} specialist for race type: {race_type}")
        
        if len(race_type_train) == 0:
            print(f"    ⚠ No training data for {race_type}")
            return None
        if len(race_type_val) == 0:
            print(f"    ⚠ No validation data for {race_type}")
            return None
            
        print(f"    Train: {len(race_type_train)} records, Val: {len(race_type_val)} records")
        
        try:
            # Check for saved race type-specific hyperparameters first
            race_type_best_params = self.load_race_type_hyperparameters(race_type, scheme)
            
            if race_type_best_params is None:
                # Tune hyperparameters specifically for this race type
                print(f"    🔧 Tuning hyperparameters for {race_type}...")
                race_type_best_params, race_type_best_ndcg = tune_hyperparameters(
                    race_type_train, race_type_val, feature_columns, k_value=self.k_value, level=self.level
                )
                print(f"    ✓ Race type best params: {race_type_best_params}")
                print(f"    ✓ Race type best NDCG@5: {race_type_best_ndcg:.4f}")
                
                # Save race type-specific hyperparameters for future use (individual file)
                self.save_race_type_hyperparameters_individual(scheme, race_type, race_type_best_params)
            
            # Store race type-specific hyperparameters in memory
            if race_type not in self.race_type_hyperparameters:
                self.race_type_hyperparameters[race_type] = {}
            self.race_type_hyperparameters[race_type][scheme] = race_type_best_params
            
            # Train specialist model for this race type only with race-specific hyperparameters
            specialist_model = train_model(
                race_type_train, feature_columns, race_type_best_params, 
                f"{scheme}_{race_type}_specialist",
                k_value=self.k_value, level=self.level
            )
            
            # Evaluate specialist on validation data of this race type
            specialist_results = evaluate_model(
                specialist_model, race_type_val, feature_columns, 
                f"{scheme}_{race_type}_specialist", 
                k_value=self.k_value, level=self.level
            )
            
            if len(specialist_results) > 0:
                # Calculate comprehensive performance metrics
                specialist_stats = calculate_summary_statistics(specialist_results)
                
                specialist_performance = {
                    'ndcg@3': specialist_stats.get('NDCG@3', {}).get('mean', 0.0),
                    'ndcg@5': specialist_stats.get('NDCG@5', {}).get('mean', 0.0),
                    'recall@3': specialist_stats.get('Recall@3', {}).get('mean', 0.0),
                    'recall@5': specialist_stats.get('Recall@5', {}).get('mean', 0.0),
                    'n_races': len(specialist_results)
                }
                
                ndcg5 = specialist_performance['ndcg@5']
                print(f"    ✓ Specialist performance: NDCG@5 = {ndcg5:.4f}")
                
                return specialist_performance
            else:
                print(f"    ⚠ No evaluation results for specialist")
                return None
                
        except Exception as e:
            print(f"    ❌ Error training specialist: {e}")
            raise
    
    def build_scheme_specialists_matrix(self, scheme, train_df, val_df, feature_columns):
        """
        Build race type × scheme performance matrix for scheme specialists ensemble
        
        Args:
            scheme (str): Current scheme name
            train_df (DataFrame): Training data
            val_df (DataFrame): Validation data
            feature_columns (list): Feature column names
        """
        print(f"\n5B. BUILDING RACE TYPE × SCHEME MATRIX...")
        print("-" * 50)
        # First try to load the matrix from file
        if os.path.exists(f"{self.results_dir}/race_type_scheme_matrix.json"):
            with open(f"{self.results_dir}/race_type_scheme_matrix.json", 'r') as f:
                self.race_type_scheme_matrix = json.load(f)
            print(f"✓ Race type × scheme matrix loaded from file")
            return
        
        # Get all unique race types from validation data
        unique_race_types = val_df['cluster'].unique()
        
        for race_type in unique_race_types:
            # Get training and validation data for this specific race type
            race_type_train = train_df[train_df['cluster'] == race_type].copy()
            race_type_val = val_df[val_df['cluster'] == race_type].copy()
            
            # Train specialist and get performance
            specialist_performance = self.train_scheme_specialist(
                scheme, race_type, race_type_train, race_type_val, feature_columns
            )
            
            # Store in race type × scheme matrix
            if specialist_performance is not None:
                if race_type not in self.race_type_scheme_matrix:
                    self.race_type_scheme_matrix[race_type] = {}
                self.race_type_scheme_matrix[race_type][scheme] = specialist_performance
        
        # Automatically save race type × scheme matrix and hyperparameters after building
        self.save_race_type_scheme_matrix()
        print(f"✓ Race type × scheme matrix saved automatically")
        
        self.save_race_type_hyperparameters()
        print(f"✓ Race type hyperparameters saved automatically")

    def get_best_scheme(self, metric='NDCG@5'):
        """
        Get the best performing scheme based on a metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            str: Best scheme name
        """
        if not self.base_results:
            raise ValueError("No base models have been trained yet")
        
        best_score = -1
        best_scheme = None
        
        for scheme, results_df in self.base_results.items():
            summary_stats = calculate_summary_statistics(results_df)
            score = summary_stats.get(metric, {}).get('mean', 0)
            
            if score > best_score:
                best_score = score
                best_scheme = scheme
        
        return best_scheme 