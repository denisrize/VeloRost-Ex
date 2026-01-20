"""
Feature Importance Analysis Experiments Module

This module handles comprehensive feature importance analysis using multiple methods:
- Permutation Importance: Aligned to NDCG@k with query groups
- LOFO/Drop-Column Importance: For a subset of top features  
- SHAP (TreeSHAP): For directionality & interactions with warm SHAP support

The analysis is performed on individual scheme models and ensemble averages.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

# Feature importance libraries
from sklearn.inspection import permutation_importance
from sklearn.metrics import ndcg_score
import shap

from ..utils.config import *
from ..core.models import train_model, evaluate_model, prepare_data_for_training
from ..core.metrics import calculate_summary_statistics, evaluate_race_predictions
from .base_models import BaseModelExperiment


class FeatureImportanceExperiment:
    """
    Manages feature importance experiments for roster or rider ranking
    
    Handles training base models and analyzing feature importance using multiple methods
    with comprehensive visualization and ensemble analysis capabilities.
    """
    
    def __init__(self, race_class, schemes=None, methods=None, year=2023, k_value=10, 
                 time_gap=None, level='rider', custom_output_dir=None):
        """
        Initialize feature importance experiment
        
        Args:
            race_class (str): Race class ('all' or 'WT')
            schemes (list): List of schemes to analyze (default: all available)
            methods (list): List of methods to use ('permutation', 'lofo', 'shap')
            year (int): Year for results (default: 2023)
            k_value (int): K value for NDCG@k evaluation (default: 10)
            time_gap (int): Time gap for features (default: None)
            level (str): Level of ranking ('roster' or 'rider')
            custom_output_dir (str): Custom output directory (overrides default)
        """
        self.race_class = race_class
        self.schemes = schemes or AVAILABLE_SCHEMES
        self.methods = methods or ['permutation', 'shap'] # , 'lofo'
        self.year = year
        self.k_value = k_value
        self.time_gap = time_gap
        self.level = level
        
        # Load configurations
        self.methods_config = get_feature_importance_methods_config()
        self.output_config = get_feature_importance_output_config()
        
        # Filter methods based on enabled status and user selection
        self.active_methods = {}
        for method in self.methods:
            if method in self.methods_config and self.methods_config[method]['enabled']:
                self.active_methods[method] = self.methods_config[method]
        
        # Set up output directory
        if custom_output_dir:
            self.output_dir = custom_output_dir
        else:
            config_key = f"{level}_feature_importance" if level == 'rider' else 'feature_importance'
            base_dir = get_output_dir('feature_importance', race_class, level=level)
            self.output_dir = f"{base_dir}/{race_class}"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for results
        self.base_experiment = None
        self.feature_importance_results = {}  # method -> scheme -> results
        self.ensemble_importance = {}
        self.shap_data_store = {}  # For warm SHAP plots
        
        # Store datasets and feature columns for easier access
        self.datasets = {}
        self.feature_columns_dict = {}
        
        print(f"Feature Importance Experiment initialized:")
        print(f"  Race class: {race_class}")
        print(f"  Level: {level}")
        print(f"  Schemes: {self.schemes}")
        print(f"  Methods: {list(self.active_methods.keys())}")
        print(f"  Output directory: {self.output_dir}")
    
    def train_base_models(self, force_retune=False):
        """
        Train base models for all schemes using existing BaseModelExperiment
        
        Args:
            force_retune (bool): Force hyperparameter retuning
            
        Returns:
            BaseModelExperiment: Trained base model experiment instance
        """
        print(f"\n{'='*80}")
        print(f"TRAINING BASE MODELS FOR FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        # Initialize base model experiment
        self.base_experiment = BaseModelExperiment(
            race_class=self.race_class,
            schemes=self.schemes,
            year=self.year,
            k_value=self.k_value,
            time_gap=self.time_gap,
            level=self.level
        )
        
        # Train all base models
        base_results = self.base_experiment.train_all_models(
            force_retune=force_retune, 
            k_value=self.k_value, 
        )
        
        # Store datasets and feature columns for easier access
        self.datasets = self.base_experiment.datasets
        self.feature_columns_dict = self.base_experiment.feature_columns_dict
        
        print(f"\n✅ Base models training completed")
        print(f"✅ Trained schemes: {list(base_results.keys())}")
        
        return self.base_experiment
    
    def calculate_permutation_importance(self, scheme):
        """
        Calculate permutation importance for a scheme using the notebook approach
        
        Args:
            scheme (str): Scheme name
            
        Returns:
            pd.DataFrame: Permutation importance results
        """
        print(f"\nCalculating permutation importance for {scheme}...")
        
        config = self.active_methods['permutation']
        model = self.base_experiment.base_models[scheme]
        test_df = self.datasets[scheme]['test']
        feature_columns = self.feature_columns_dict[scheme]
        
        # Prepare data
        X_test, y_test, test_groups = prepare_data_for_training(
            test_df, feature_columns, k_value=self.k_value, level=self.level
        )
        
        print(f"  Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"  Query groups: {len(test_groups)} races")
        
        # Calculate baseline score
        dtest_baseline = xgb.DMatrix(X_test, feature_names=feature_columns)
        y_pred_baseline = model.predict(dtest_baseline)
        baseline_score = self._custom_ndcg_scorer(y_test, y_pred_baseline, test_groups)
        
        print(f"  Baseline NDCG@{self.k_value}: {baseline_score:.4f}")
        
        # Calculate permutation importance for each feature
        importances = []
        n_repeats = config['n_repeats']
        random_seed = config['random_seed']
        
        for i, feature in enumerate(tqdm(feature_columns, desc="Permutation Importance")):
            feature_importances = []
            
            for repeat in range(n_repeats):
                # Create a copy of the data
                X_permuted = X_test.copy()
                
                # Permute the feature column
                np.random.seed(random_seed + repeat * len(feature_columns) + i)
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                
                # Calculate score with permuted feature
                dtest_permuted = xgb.DMatrix(X_permuted, feature_names=feature_columns)
                y_pred_permuted = model.predict(dtest_permuted)
                permuted_score = self._custom_ndcg_scorer(y_test, y_pred_permuted, test_groups)
                
                # Importance is the decrease in performance
                importance = baseline_score - permuted_score
                feature_importances.append(importance)
            
            # Store mean and std of importance across repeats
            importances.append({
                'feature': feature,
                'importance_mean': np.mean(feature_importances),
                'importance_std': np.std(feature_importances)
            })
        
        # Convert to DataFrame and sort
        perm_df = pd.DataFrame(importances)
        perm_df = perm_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
        
        print(f"  ✓ Permutation importance calculated for {len(perm_df)} features")
        if perm_df['importance_mean'].sum() > 0:
            print(f"  Top feature: {perm_df.iloc[0]['feature']} (importance: {perm_df.iloc[0]['importance_mean']:.4f})")
        
        return perm_df
    
    def calculate_lofo_importance(self, scheme):
        """
        Calculate LOFO/Drop-column importance for a scheme
        
        Args:
            scheme (str): Scheme name
            
        Returns:
            pd.DataFrame: LOFO importance results
        """
        print(f"\nCalculating LOFO importance for {scheme}...")
        
        config = self.active_methods['lofo']
        model = self.base_experiment.base_models[scheme]
        train_df = self.datasets[scheme]['train_val']
        test_df = self.datasets[scheme]['test']
        feature_columns = self.feature_columns_dict[scheme]
        hyperparams = self.base_experiment.hyperparameters[scheme]
        
        # Get baseline NDCG with all features
        baseline_ndcg = self._calculate_baseline_ndcg(model, test_df, feature_columns)
        print(f"  Baseline NDCG@{self.k_value}: {baseline_ndcg:.4f}")
        
        # Get top features from permutation importance (if available)
        if ('permutation' in self.feature_importance_results and 
            scheme in self.feature_importance_results['permutation']):
            top_features_df = self.feature_importance_results['permutation'][scheme]
            top_features = top_features_df.head(config['top_features_only'])['feature'].tolist()
            print(f"  Analyzing top {len(top_features)} features from permutation importance")
        else:
            top_features = feature_columns[:config['top_features_only']]
            print(f"  Analyzing first {len(top_features)} features (no permutation results available)")
        
        lofo_results = []
        
        for i, feature_to_drop in enumerate(tqdm(top_features, desc="LOFO Analysis")):
            try:
                # Create feature list without this feature
                reduced_features = [f for f in feature_columns if f != feature_to_drop]
                
                # Train new model without this feature
                reduced_model = train_model(
                    train_df, reduced_features, hyperparams,
                    f"{scheme}_lofo_{i}", 
                    k_value=self.k_value, level=self.level
                )
                
                # Calculate NDCG without this feature
                reduced_ndcg = self._calculate_baseline_ndcg(reduced_model, test_df, reduced_features)
                
                # Calculate importance as difference
                importance = baseline_ndcg - reduced_ndcg
                
                lofo_results.append({
                    'feature': feature_to_drop,
                    'baseline_ndcg': baseline_ndcg,
                    'reduced_ndcg': reduced_ndcg,
                    'importance': importance,
                    'importance_pct': (importance / baseline_ndcg * 100) if baseline_ndcg > 0 else 0
                })
                
            except Exception as e:
                print(f"    ⚠️ Error processing feature {feature_to_drop}: {e}")
                continue
        
        # Create results DataFrame
        lofo_df = pd.DataFrame(lofo_results)
        lofo_df = lofo_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"  ✓ LOFO analysis completed for {len(lofo_df)} features")
        if len(lofo_df) > 0:
            print(f"  Top feature: {lofo_df.iloc[0]['feature']} (importance: {lofo_df.iloc[0]['importance']:.4f})")
        
        return lofo_df
    
    def calculate_shap_importance(self, scheme):
        """
        Calculate SHAP importance with warm SHAP data storage
        
        Args:
            scheme (str): Scheme name
            
        Returns:
            tuple: (shap_df, shap_data) - Results DataFrame and data for warm plots
        """
        print(f"\nCalculating SHAP importance for {scheme}...")
        
        config = self.active_methods['shap']
        model = self.base_experiment.base_models[scheme]
        test_df = self.datasets[scheme]['test']
        feature_columns = self.feature_columns_dict[scheme]
        
        # Sample data for SHAP analysis
        sample_size = config['sample_size']
        if len(test_df) > sample_size:
            test_sample = test_df.sample(n=sample_size, random_state=42)
            print(f"  Using sample of {sample_size} records (from {len(test_df)} total)")
        else:
            test_sample = test_df.copy()
            print(f"  Using all {len(test_sample)} records")
        
        # Prepare data
        X_sample = test_sample[feature_columns].values
        
        print(f"  Creating SHAP explainer...")
        
        # Create SHAP explainer for XGBoost
        explainer = shap.TreeExplainer(model)
        
        print(f"  Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance as mean absolute SHAP values
        shap_importance = np.abs(shap_values).mean(0)
        
        # Create results DataFrame
        shap_df = pd.DataFrame({
            'feature': feature_columns,
            'shap_importance': shap_importance,
            'shap_importance_abs': shap_importance  # Already absolute
        })
        
        # Sort by importance
        shap_df = shap_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
        
        print(f"  ✓ SHAP analysis completed for {len(shap_df)} features")
        print(f"  Top feature: {shap_df.iloc[0]['feature']} (SHAP importance: {shap_df.iloc[0]['shap_importance']:.4f})")
        
        # Store SHAP data for warm plots if enabled
        shap_data = None
        if config['save_warm_shap']:
            shap_data = {
                'values': shap_values,
                'base_value': explainer.expected_value,
                'data': X_sample,
                'feature_names': feature_columns,
                'sample_df': test_sample,
                'explainer': explainer  # Save explainer for warm plots
            }
            
            # Save SHAP data to file for later use
            scheme_dir = f"{self.output_dir}/{scheme}/{self.time_gap}_days"
            os.makedirs(scheme_dir, exist_ok=True)
            shap_file = f"{scheme_dir}/shap_data.pkl"
            try:
                with open(shap_file, 'wb') as f:
                    pickle.dump(shap_data, f)
                print(f"  💾 SHAP data saved for warm plots: {shap_file}")
            except Exception as e:
                print(f"  ⚠️ Could not save SHAP data: {e}")
        
        return shap_df, shap_data
    
    def run_feature_importance_analysis(self, force_retune=False):
        """
        Run complete feature importance analysis pipeline
        
        Args:
            force_retune (bool): Force hyperparameter retuning for base models
            
        Returns:
            dict: Comprehensive feature importance results
        """
        print(f"\n{'='*80}")
        print(f"FEATURE IMPORTANCE ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Race class: {self.race_class}")
        print(f"Level: {self.level}")
        print(f"Year: {self.year}")
        print(f"K value: {self.k_value}")
        print(f"Methods: {list(self.active_methods.keys())}")
        
        # Step 1: Train base models
        if self.base_experiment is None:
            self.train_base_models(force_retune=force_retune)
        
        # Step 2: Run feature importance analysis for each method and scheme
        for method in self.active_methods.keys():
            print(f"\n{'='*80}")
            print(f"{self.active_methods[method]['name'].upper()} ANALYSIS")
            print(f"{'='*80}")
            
            self.feature_importance_results[method] = {}
            
            for scheme in self.schemes:
                if scheme not in self.base_experiment.base_models:
                    print(f"⚠️ Skipping {scheme} - no trained model available")
                    continue
                
                try:
                    if method == 'permutation':
                        result_df = self.calculate_permutation_importance(scheme)
                    elif method == 'lofo':
                        result_df = self.calculate_lofo_importance(scheme)
                    elif method == 'shap':
                        result_df, shap_data = self.calculate_shap_importance(scheme)
                        if shap_data is not None:
                            if method not in self.shap_data_store:
                                self.shap_data_store[method] = {}
                            self.shap_data_store[method][scheme] = shap_data
                    else:
                        print(f"⚠️ Unknown method: {method}")
                        continue
                    
                    self.feature_importance_results[method][scheme] = result_df
                    
                    # Save individual results
                    if self.output_config['save_results']:
                        # Create scheme-specific subdirectory
                        scheme_dir = f"{self.output_dir}/{scheme}/{self.time_gap}_days"
                        os.makedirs(scheme_dir, exist_ok=True)
                        
                        result_file = f"{scheme_dir}/{method}_importance.csv"
                        result_df.to_csv(result_file, index=False)
                        print(f"  💾 Results saved to: {result_file}")
                        
                except Exception as e:
                    print(f"❌ Error calculating {method} importance for {scheme}: {e}")
                    continue
            
            print(f"\n✅ {self.active_methods[method]['name']} analysis completed for {len(self.feature_importance_results[method])} schemes")
        
        # Step 3: Calculate ensemble feature importance
        if self.output_config['create_ensemble_summary']:
            self.calculate_ensemble_feature_importance()
        
        # Step 4: Create visualizations
        if self.output_config['save_plots']:
            self.create_visualizations()
        
        # Step 5: Save comprehensive summary
        self.save_analysis_summary()
        
        print(f"\n{'='*80}")
        print(f"✅ FEATURE IMPORTANCE ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"📊 Methods completed: {len(self.feature_importance_results)}")
        print(f"🎯 Schemes analyzed: {len(self.schemes)}")
        print(f"💾 Results saved to: {self.output_dir}")
        
        return self.feature_importance_results
    
    def calculate_ensemble_feature_importance(self):
        """Calculate ensemble feature importance by averaging across schemes"""
        print(f"\n{'='*80}")
        print(f"ENSEMBLE FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        if len(self.base_experiment.base_models) < 2:
            print("⚠️ Not enough models for ensemble analysis")
            return
        
        for method, method_results in self.feature_importance_results.items():
            if not method_results:
                continue
                
            print(f"\nCombining {method} importance across schemes...")
            
            # Get common features across all schemes
            all_features = set()
            for scheme_results in method_results.values():
                all_features.update(scheme_results['feature'].tolist())
            
            # Calculate average importance for each feature
            ensemble_data = []
            importance_col = self._get_importance_column(method)
            
            for feature in all_features:
                importances = []
                for scheme, scheme_results in method_results.items():
                    feature_row = scheme_results[scheme_results['feature'] == feature]
                    if not feature_row.empty:
                        importances.append(feature_row.iloc[0][importance_col])
                
                if importances:  # Only include if feature exists in at least one scheme
                    ensemble_data.append({
                        'feature': feature,
                        'importance_mean': np.mean(importances),
                        'importance_std': np.std(importances),
                        'n_schemes': len(importances)
                    })
            
            if ensemble_data:
                ensemble_df = pd.DataFrame(ensemble_data)
                ensemble_df = ensemble_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
                self.ensemble_importance[method] = ensemble_df
                
                # Save ensemble results
                if self.output_config['save_results']:
                    ensemble_file = f"{self.output_dir}/ensemble_{method}_importance.csv"
                    ensemble_df.to_csv(ensemble_file, index=False)
                    print(f"    💾 Ensemble {method} results saved to: {ensemble_file}")
                
                print(f"    ✓ Combined {method} importance for {len(ensemble_df)} features")
        
        print(f"\n✅ Ensemble feature importance calculated for {len(self.ensemble_importance)} methods")
    
    def create_visualizations(self):
        """Create comprehensive visualizations for feature importance results"""
        print(f"\n{'='*80}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Method comparison plots
        if self.output_config['create_method_comparisons']:
            for method, method_results in self.feature_importance_results.items():
                if method_results:
                    print(f"\nCreating {method} comparison plots...")
                    self._create_method_comparison_plot(method, method_results)
        
        # 2. Ensemble summary plots
        if self.ensemble_importance and self.output_config['create_ensemble_summary']:
            print(f"\nCreating ensemble summary plots...")
            self._create_ensemble_summary_plot()
        
        # 3. SHAP detailed plots (if SHAP data available)
        if 'shap' in self.shap_data_store and self.active_methods.get('shap', {}).get('create_dependence_plots', False):
            print(f"\nCreating SHAP detailed plots...")
            self._create_shap_detailed_plots()
        
        print(f"\n✅ All visualizations created successfully!")
    
    def save_analysis_summary(self):
        """Save comprehensive analysis summary"""
        summary_file = f"{self.output_dir}/feature_importance_analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("FEATURE IMPORTANCE ANALYSIS - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Race Class: {self.race_class}\n")
            f.write(f"  Level: {self.level}\n")
            f.write(f"  Test Year: {self.year}\n")
            f.write(f"  K Value: {self.k_value}\n")
            f.write(f"  Schemes: {self.schemes}\n")
            f.write(f"  Methods: {list(self.active_methods.keys())}\n\n")
            
            # Add method results summary
            for method, method_results in self.feature_importance_results.items():
                f.write(f"{method.upper()} Results:\n")
                f.write(f"  Schemes analyzed: {len(method_results)}\n")
                if method_results:
                    total_features = sum(len(df) for df in method_results.values())
                    f.write(f"  Total features analyzed: {total_features}\n")
                f.write("\n")
            
            # Add ensemble summary if available
            if self.ensemble_importance:
                f.write("Ensemble Results:\n")
                for method, ensemble_df in self.ensemble_importance.items():
                    f.write(f"  {method}: {len(ensemble_df)} features\n")
                f.write("\n")
        
        print(f"💾 Comprehensive summary saved: {summary_file}")
    
    # Helper methods
    def _custom_ndcg_scorer(self, y_true, y_pred, groups):
        """Custom NDCG scorer that respects query groups"""
        scores = []
        start_idx = 0
        
        for group_size in groups:
            end_idx = start_idx + group_size
            group_true = y_true[start_idx:end_idx]
            group_pred = y_pred[start_idx:end_idx]
            
            # Calculate NDCG for this group
            if len(np.unique(group_true)) > 1:  # Only if there's variation in labels
                try:
                    ndcg = ndcg_score([group_true], [group_pred], k=self.k_value)
                    scores.append(ndcg)
                except:
                    scores.append(0.0)  # Default score if calculation fails
            else:
                scores.append(0.0)  # No variation in labels
            
            start_idx = end_idx
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_baseline_ndcg(self, model, test_df, feature_columns):
        """Calculate baseline NDCG with all features"""
        test_df_exp = test_df.copy()
        test_df_exp['race_id'] = test_df_exp['race'] + "_" + test_df_exp['date'].astype(str)
        test_df_exp = test_df_exp.sort_values('race_id').reset_index(drop=True)
        
        # Assign labels
        from ..core.models import assign_roster_label
        test_df_exp = test_df_exp.groupby('race_id', group_keys=False).apply(
            assign_roster_label, k_value=self.k_value, level=self.level
        )
        
        # Predict
        X_test = test_df_exp[feature_columns].values
        dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
        test_df_exp['pred_score'] = model.predict(dtest)
        
        # Calculate NDCG per race
        ndcg_scores = []
        for race_id, group in test_df_exp.groupby('race_id'):
            race_metrics = evaluate_race_predictions(group, k_values=[self.k_value], level=self.level)
            ndcg_score = race_metrics[f'NDCG@{self.k_value}']
            if not np.isnan(ndcg_score):
                ndcg_scores.append(ndcg_score)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _get_importance_column(self, method):
        """Get the importance column name for a method"""
        if method == 'permutation':
            return 'importance_mean'
        elif method == 'lofo':
            return 'importance'
        elif method == 'shap':
            return 'shap_importance'
        else:
            return 'importance_mean'
    
    def _create_method_comparison_plot(self, method, method_results):
        """Create comparison plot for a specific method across schemes"""
        if not method_results:
            return
        
        config = self.active_methods[method]
        top_n = config.get('top_features', 15)
        
        fig, axes = plt.subplots(1, len(method_results), figsize=(5*len(method_results), 8))
        if len(method_results) == 1:
            axes = [axes]
        
        fig.suptitle(f'{config["name"]} Comparison Across Schemes', fontsize=16, fontweight='bold')
        
        for idx, (scheme, results_df) in enumerate(method_results.items()):
            ax = axes[idx]
            
            # Get top features
            top_features = results_df.head(top_n)
            importance_col = self._get_importance_column(method)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_features[importance_col], 
                    color=config.get('color', '#1f77b4'), alpha=0.7)
            
            # Customize
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_features['feature']], 
                              fontsize=8)
            ax.set_xlabel('Importance Score', fontsize=10)
            ax.set_title(f'{scheme.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
        
        plt.tight_layout()
        
        if self.output_config['save_plots']:
            plot_file = f"{self.output_dir}/{method}_comparison.{self.output_config['plot_format']}"
            plt.savefig(plot_file, dpi=self.output_config['plot_dpi'], bbox_inches='tight')
            print(f"  📊 Plot saved: {plot_file}")
        
        plt.close()
    
    def _create_ensemble_summary_plot(self):
        """Create ensemble summary plot"""
        if not self.ensemble_importance:
            return
        
        n_methods = len(self.ensemble_importance)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 10))
        if n_methods == 1:
            axes = [axes]
        
        fig.suptitle('Ensemble Feature Importance Summary', fontsize=16, fontweight='bold')
        
        for idx, (method, results_df) in enumerate(self.ensemble_importance.items()):
            ax = axes[idx]
            
            # Get top features
            top_features = results_df.head(20)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            bars = ax.barh(y_pos, top_features['importance_mean'], 
                          xerr=top_features['importance_std'] if 'importance_std' in top_features.columns else None,
                          color=self.active_methods.get(method, {}).get('color', '#1f77b4'),
                          alpha=0.7, capsize=3)
            
            # Customize
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features['feature']], 
                              fontsize=8)
            ax.set_xlabel('Average Importance Score', fontsize=10)
            ax.set_title(f'{method.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
            # Add scheme count annotations
            if 'n_schemes' in top_features.columns:
                for i, (bar, n_schemes) in enumerate(zip(bars, top_features['n_schemes'])):
                    ax.text(bar.get_width() + 0.01*ax.get_xlim()[1], bar.get_y() + bar.get_height()/2, 
                           f'({n_schemes})', ha='left', va='center', fontsize=7, alpha=0.7)
        
        plt.tight_layout()
        
        if self.output_config['save_plots']:
            plot_file = f"{self.output_dir}/ensemble_importance_summary.{self.output_config['plot_format']}"
            plt.savefig(plot_file, dpi=self.output_config['plot_dpi'], bbox_inches='tight')
            print(f"  📊 Ensemble plot saved: {plot_file}")
        
        plt.close()
    
    def _create_shap_detailed_plots(self):
        """Create detailed SHAP plots including warm SHAP functionality"""
        if 'shap' not in self.shap_data_store:
            return
        
        config = self.active_methods['shap']
        max_features = config.get('max_dependence_features', 5)
        
        for scheme, shap_data in self.shap_data_store['shap'].items():
            print(f"  Creating detailed SHAP plots for {scheme}...")
            
            try:
                # 1. Summary Plot (Feature Importance + Directionality)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_data['values'], shap_data['data'], 
                                feature_names=shap_data['feature_names'],
                                max_display=20, show=False)
                plt.title(f'SHAP Summary Plot - {scheme.replace("_", " ").title()}', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                if self.output_config['save_plots']:
                    scheme_dir = f"{self.output_dir}/{scheme}/{self.time_gap}_days"
                    os.makedirs(scheme_dir, exist_ok=True)
                    plot_file = f"{scheme_dir}/shap_summary.{self.output_config['plot_format']}"
                    plt.savefig(plot_file, dpi=self.output_config['plot_dpi'], bbox_inches='tight')
                    print(f"    💾 SHAP summary plot saved: {plot_file}")
                
                plt.close()
                
                # 2. Bar Plot (Feature Importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_data['values'], shap_data['data'], 
                                feature_names=shap_data['feature_names'],
                                plot_type="bar", max_display=20, show=False)
                plt.title(f'SHAP Feature Importance - {scheme.replace("_", " ").title()}', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                if self.output_config['save_plots']:
                    scheme_dir = f"{self.output_dir}/{scheme}/{self.time_gap}_days"
                    os.makedirs(scheme_dir, exist_ok=True)
                    plot_file = f"{scheme_dir}/shap_bar.{self.output_config['plot_format']}"
                    plt.savefig(plot_file, dpi=self.output_config['plot_dpi'], bbox_inches='tight')
                    print(f"    💾 SHAP bar plot saved: {plot_file}")
                
                plt.close()
                
                # 3. Dependence Plots for Top Features
                if scheme in self.feature_importance_results['shap']:
                    top_features = self.feature_importance_results['shap'][scheme].head(max_features)['feature'].tolist()
                    
                    for feature in top_features:
                        if feature in shap_data['feature_names']:
                            feature_idx = shap_data['feature_names'].index(feature)
                            
                            plt.figure(figsize=(10, 6))
                            shap.dependence_plot(feature_idx, shap_data['values'], shap_data['data'],
                                               feature_names=shap_data['feature_names'], 
                                               show=False)
                            plt.title(f'SHAP Dependence Plot - {feature} ({scheme.replace("_", " ").title()})', 
                                     fontsize=12, fontweight='bold')
                            plt.tight_layout()
                            
                            if self.output_config['save_plots']:
                                scheme_dir = f"{self.output_dir}/{scheme}/{self.time_gap}_days"
                                os.makedirs(scheme_dir, exist_ok=True)
                                safe_feature_name = feature.replace('/', '_').replace(' ', '_')[:50]
                                plot_file = f"{scheme_dir}/shap_dependence_{safe_feature_name}.{self.output_config['plot_format']}"
                                plt.savefig(plot_file, dpi=self.output_config['plot_dpi'], bbox_inches='tight')
                                print(f"    💾 SHAP dependence plot saved: {plot_file}")
                            
                            plt.close()
                
                print(f"  ✅ SHAP detailed analysis completed for {scheme}")
                
            except Exception as e:
                print(f"  ❌ Error creating SHAP plots for {scheme}: {e}")
                continue
    
    def load_warm_shap_data(self, scheme):
        """
        Load saved SHAP data for warm SHAP plots
        
        Args:
            scheme (str): Scheme name
            
        Returns:
            dict: SHAP data for warm plots or None if not found
        """
        shap_file = f"{self.output_dir}/{scheme}/shap_data.pkl"
        
        if not os.path.exists(shap_file):
            print(f"⚠️ No saved SHAP data found for {scheme}")
            return None
        
        try:
            with open(shap_file, 'rb') as f:
                shap_data = pickle.load(f)
            print(f"✓ Warm SHAP data loaded for {scheme}")
            return shap_data
        except Exception as e:
            print(f"❌ Error loading SHAP data for {scheme}: {e}")
            return None
