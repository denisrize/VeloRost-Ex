#!/usr/bin/env python3
"""
Ensemble Experiments for Roster Ranking

This module implements ensemble learning approaches that combine predictions
from multiple scheme-based models using either simple averaging or meta-learning.
Uses BaseModelExperiment internally to handle base models and avoid code duplication.
"""

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'  # Force immediate output for job monitoring
import time
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from tqdm import tqdm
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from roster_ranker.utils import *
from roster_ranker.core import (
    tune_hyperparameters, train_model, evaluate_model, 
    get_feature_importance, prepare_data_for_training, get_hyperparameter_grid
)
from roster_ranker.core import calculate_summary_statistics, evaluate_race_predictions
from roster_ranker.data import load_or_create_dataset, load_and_merge_features
from roster_ranker.data import extract_race_ensemble_features
from .base_models import BaseModelExperiment
from roster_ranker.core import ndcg_at_k, recall_at_k

warnings.filterwarnings('ignore')

# Force unbuffered output for cluster jobs
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

# ============================================================================
# NEURAL NETWORK GATING MODELS
# ============================================================================

class GatingMLP(nn.Module):
    """
    Simple MLP for gating network classification
    """
    def __init__(self, input_dim=17, hidden_dims=[64, 32], num_classes=3, dropout=0.1, 
                 learnable_temperature=False, init_temperature=1.0):
        super(GatingMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add input normalization
        layers.append(nn.BatchNorm1d(input_dim))
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temperature, dtype=torch.float32))
            self.learnable_temperature = True
        else:
            self.register_buffer('temperature', torch.tensor(1.0, dtype=torch.float32))
            self.learnable_temperature = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        logits = self.network(x)
        # Apply temperature scaling
        return logits / torch.clamp(self.temperature, min=0.1)  # Clamp to avoid division by very small values


class GatingMLPWrapper:
    """
    Scikit-learn compatible wrapper for GatingMLP
    """
    def __init__(self, hidden_dims=[64, 32], lr=0.001, epochs=500, batch_size=32, 
                 dropout=0.1, weight_decay=1e-4, patience=40, random_state=42,
                 learnable_temperature=False, init_temperature=1.0, use_class_weights=True):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state
        self.learnable_temperature = learnable_temperature
        self.init_temperature = init_temperature
        self.use_class_weights = use_class_weights
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model with optional validation data for early stopping"""
        from sklearn.utils.class_weight import compute_class_weight
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        num_classes = len(self.classes_)
        
        # Create model
        self.model = GatingMLP(
            input_dim=self.n_features_in_,
            hidden_dims=self.hidden_dims,
            num_classes=num_classes,
            dropout=self.dropout,
            learnable_temperature=self.learnable_temperature,
            init_temperature=self.init_temperature
        ).to(self.device)
        
        # Prepare training data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Prepare validation data if provided
        val_dataloader = None
        if X_val is not None and y_val is not None:
            X_val = np.array(X_val, dtype=np.float32)
            y_val = np.array(y_val, dtype=np.int64)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup training with class weights
        if self.use_class_weights:
            # Compute class weights using sklearn
            class_weights = compute_class_weight('balanced', classes=self.classes_, y=y)
            class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
            train_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print(f"Using class weights for training: {dict(zip(self.classes_, class_weights))}")
        else:
            train_criterion = nn.CrossEntropyLoss()
        
        # Always use unweighted loss for validation (cleaner signal)
        val_criterion = nn.CrossEntropyLoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = train_criterion(outputs, batch_y)  # Use weighted loss for training
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(dataloader)
            
            # Validation phase (if validation data provided)
            if val_dataloader is not None:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X_val, batch_y_val in val_dataloader:
                        val_outputs = self.model(batch_X_val)
                        val_loss = val_criterion(val_outputs, batch_y_val)  # Use unweighted loss for validation
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_dataloader)
                loss_to_track = avg_val_loss  # Use validation loss for early stopping
                
                if epoch % 10 == 0:  # Print every 10 epochs
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                loss_to_track = avg_train_loss  # Use training loss if no validation data
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")
            
            # Early stopping based on validation loss (or training loss if no validation)
            if loss_to_track < best_loss:
                best_loss = loss_to_track
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model state
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def decision_function(self, X):
        """Get raw decision scores (before temperature scaling for compatibility)"""
        self.model.eval()
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Get raw logits from network (before temperature)
            if self.model.learnable_temperature:
                # For learnable temperature, get raw logits and multiply back by temperature
                outputs = self.model(X_tensor)  # This has temperature applied
                raw_logits = outputs * torch.clamp(self.model.temperature, min=0.1)
            else:
                # For fixed temperature, forward pass is just raw logits
                outputs = self.model(X_tensor)
                raw_logits = outputs
        
        return raw_logits.cpu().numpy()
    
    def score(self, X, y):
        """Get accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class EnsembleExperiment:
    """
    Manages ensemble experiments for roster or rider ranking
    
    Combines predictions from multiple scheme-based models using either simple 
    averaging or meta-learning. Uses BaseModelExperiment internally to handle 
    individual scheme models and avoid code duplication.
    """
    
    def __init__(self, race_class, base_models=None, schemes=None, ensemble_methods=None, custom_output_dir=None, custom_hyperparams_dir=None, year=2023, k_value=5, time_gap=None, level='roster', optimization_strategy='cluster_race_class', gating_model_type='logistic', exp_name='class_features'):
        """
        Initialize ensemble experiment
        
        Args:
            race_class (str): Race class ('all' or 'WT')
            schemes (list): List of schemes to use (default: ['time_lag', 'equal_weight', 'rank_norm'])
            ensemble_methods (list): List of ensemble methods (default: ['simple_average', 'meta_learning', 'static_moe'])
            custom_output_dir (str): Custom output directory (overrides default)
            year (int): Year for results (default: 2023)
            level (str): Level of ranking ('roster' or 'rider')
            optimization_strategy (str): Strategy for cluster performance optimization:
                - 'cluster_race_class': Use cluster+race_class combinations (default)
                - 'cluster_only': Use cluster-only combinations
            gating_model_type (str): Type of gating model for MoE methods ('logistic' or 'mlp')
        """
        self.race_class = race_class
        self.schemes = schemes if schemes else AVAILABLE_SCHEMES
        self.ensemble_methods = ensemble_methods or ['simple_average', 'meta_learning', 'static_moe', 'scheme_specialists', 'adaptive_feature_selection', 'hard_moe_gating', 'soft_moe_gating']
        self.year = year
        self.k_value = k_value
        self.time_gap = time_gap
        self.level = level
        self.optimization_strategy = optimization_strategy
        self.gating_model_type = gating_model_type
        self.exp_name = exp_name
        # Internal state for ensemble experiments
        if base_models:
            self.base_experiment = base_models
        else:
            self.base_experiment = self.setup_base_models(k_value=self.k_value)
        
        if custom_hyperparams_dir:
            self.hyperparams_dir = custom_hyperparams_dir
        else:
            self.hyperparams_dir = get_hyperparams_dir(race_class, k_value=self.k_value, level=level)

        self.ensemble_results = {}  # Method -> results mapping
        self.ensemble_models = {}   # Method -> model mapping (for meta-learning)
        self.feature_importance = {}  # Method -> feature importance mapping
        
        # Set up output directory
        if custom_output_dir:
            self.results_dir = custom_output_dir
        else:
            output_dir = get_output_dir('ensemble', race_class, level=level)
            self.results_dir = f"{output_dir}/{exp_name}"
        
        print(f"Ensemble results directory: {self.results_dir}")
        print(f"Ensemble hyperparameters directory: {self.hyperparams_dir}")
        
        if self.time_gap:
            self.results_dir = f"{self.results_dir}/{self.time_gap}_days"
            self.hyperparams_dir = f"{self.hyperparams_dir}/{self.time_gap}_days"
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.hyperparams_dir, exist_ok=True)
    
    def setup_base_models(self, force_retune=False, k_value=5):
        """
        Set up and train base models using BaseModelExperiment
        
        Args:
            force_retune (bool): Force hyperparameter retuning for base models
        """
        print("="*80)
        print("SETTING UP BASE MODELS")
        print("="*80)
        print(f"Race class: {self.race_class}")
        print(f"Schemes: {self.schemes}")

        # Initialize base model experiment
        self.base_experiment = BaseModelExperiment(
            race_class=self.race_class,
            schemes=self.schemes,
            year=self.year,
            level=self.level,
            optimization_strategy=self.optimization_strategy
        )
        
        # Train all base models
        print(f"\nTraining base models...")
        base_results = self.base_experiment.train_all_models(force_retune=force_retune, k_value=self.k_value)
        
        print(f"✅ Base models setup completed!")
        print(f"✅ Trained models: {list(self.base_experiment.base_models.keys())}")
        
        return base_results
    
    def get_base_predictions_for_year(self, year, one_year=False):
        """
        Get base model predictions for a specific year using BaseModelExperiment data
        
        Args:
            year (int): Year to get predictions for
            one_year (bool): If True, only use data from this specific year
        Returns:
            dict: Scheme -> predictions DataFrame mapping
        """
        if self.base_experiment is None:
            raise ValueError("Base models not set up. Call setup_base_models() first.")
        
        predictions = {}
        
        for scheme in self.schemes:
            # Load scheme data
            scheme_data = load_or_create_dataset(self.race_class, scheme, level=self.level, exp_name=self.exp_name, time_gap=self.time_gap)
            if one_year:
                year_data = scheme_data[scheme_data['year'] == year].copy()
            else:
                year_data = scheme_data[scheme_data['year'] <= year].copy()
            
            # Sort data consistently
            year_data['race_id'] = year_data['race'] + "_" + year_data['date'].astype(str)
            year_data = year_data.sort_values('race_id').reset_index(drop=True)
            
            # Prepare features (same as training)
            feature_columns = [col for col in year_data.columns if col not in EXCLUDE_COLS]
            
            # Create test matrix and predict
            X_test = year_data[feature_columns].values
            dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
            model = self.base_experiment.base_models[scheme]
            scores = model.predict(dtest)
            
            # Store predictions with race and team info
            rank_col, record_id = get_ranking_config(self.level)
            pred_df = year_data.copy()
            pred_df['pred_score'] = scores
            pred_df['scheme'] = scheme
            
            predictions[scheme] = pred_df

        return predictions
    
    def align_predictions(self, base_predictions):
        """
        Align predictions by race and team across all models
        
        Args:
            base_predictions (dict): Scheme -> predictions DataFrame mapping
            
        Returns:
            dict: Race key -> scheme -> scores mapping
        """
        aligned_data = defaultdict(dict)
        
        # Get all unique races from first scheme
        first_scheme = list(base_predictions.keys())[0]
        reference_df = base_predictions[first_scheme]
        unique_races = reference_df[['race', 'date']].drop_duplicates()
        rank_col, record_id = get_ranking_config(self.level)
        for _, race_row in unique_races.iterrows():
            race_key = (race_row['race'], race_row['date'])
            
            for scheme, pred_df in base_predictions.items():
                # Get race subset and sort by team consistently
                race_subset = pred_df[
                    (pred_df['race'] == race_row['race']) & 
                    (pred_df['date'] == race_row['date'])
                ].copy()
                
                race_subset = race_subset.sort_values(record_id).reset_index(drop=True)
                aligned_data[race_key][scheme] = race_subset['pred_score'].values
        
        return aligned_data
    
    def run_simple_average_ensemble(self, test_year=2023):
        """
        Run simple average ensemble method
        
        Args:
            test_year (int): Year to evaluate on
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print("\n" + "="*60)
        print("SIMPLE AVERAGE ENSEMBLE")
        print("="*60)
        
        # Get base model predictions for test year
        base_predictions = self.get_base_predictions_for_year(test_year, one_year=True)
        
        if not base_predictions:
            raise ValueError("No base predictions available")
        
        # Align predictions across models
        aligned_data = self.align_predictions(base_predictions)
        
        ensemble_results = []
        rank_col, record_id = get_ranking_config(self.level)
        for race_key, race_data in aligned_data.items():
            scores_matrix = np.array([race_data[scheme] for scheme in self.schemes if scheme in race_data])
            
            if len(scores_matrix) == 0:
                raise ValueError(f"No scores for race {race_key}")
            
            # Simple average of raw scores
            ensemble_scores = np.mean(scores_matrix, axis=0)
            
            # Create result dataframe
            race_df = list(base_predictions.values())[0]
            race_subset = race_df[
                (race_df['race'] == race_key[0]) & 
                (race_df['date'] == race_key[1])
            ].copy()
            race_subset = race_subset.sort_values(record_id).reset_index(drop=True)
            
            race_subset['ensemble_score'] = ensemble_scores
            ensemble_results.append(race_subset[[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score']])
        
        ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        
        # Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Simple average ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {ensemble_eval[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {ensemble_eval[f'Recall@{self.k_value}'].mean():.4f}")
        
        # Store results
        self.ensemble_results['simple_average'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval)
        }
        
        return ensemble_eval
    
    def prepare_meta_features_for_year(self, year, one_year=False):
        """
        Prepare meta-features (base predictions + race context) for a specific year
        
        Args:
            year (int): Year to prepare features for
            one_year (bool): If True, only use data from this specific year
            
        Returns:
            tuple: (X_meta, y_meta, meta_groups)
        """
        # Get base model predictions for training data
        base_predictions = self.get_base_predictions_for_year(year, one_year=one_year)
        
        meta_features = []
        meta_labels = []
        meta_groups = []
        
        # Use first scheme's data as reference
        reference_scheme = list(base_predictions.keys())[0]
        reference_data = base_predictions[reference_scheme]
        
        # Group by race
        for (race_name, race_date), race_ref_data in reference_data.groupby(['race', 'date']):
            race_ref_data = race_ref_data.copy()
            
            if len(race_ref_data) == 0:
                continue
            
            # Sort teams consistently
            race_ref_data = race_ref_data.sort_values('team').reset_index(drop=True)
            
            # Get base model predictions for this race
            race_base_features = []
            
            for scheme in self.schemes:
                if scheme in base_predictions:
                    scheme_preds = base_predictions[scheme]
                    race_preds = scheme_preds[
                        (scheme_preds['race'] == race_name) & 
                        (scheme_preds['date'] == race_date)
                    ].copy()
                    
                    if len(race_preds) > 0:
                        # Align with reference data
                        aligned_preds = pd.merge(
                            race_ref_data[['race', 'date', 'team']], 
                            race_preds[['race', 'date', 'team', 'pred_score']], 
                            on=['race', 'date', 'team'], 
                            how='left'
                        )
                        race_base_features.append(aligned_preds['pred_score'].values)
            
            # Extract race features
            race_features = extract_race_ensemble_features(race_ref_data)
            
            # Combine base predictions with race features
            race_base_matrix = np.column_stack(race_base_features)
            
            # Expand race features to match number of teams
            n_teams = len(race_ref_data)
            race_features_expanded = np.tile(race_features, (n_teams, 1))
            
            # Combined meta features
            combined_features = np.hstack([race_base_matrix, race_features_expanded])
            
            # Labels using consistent relevance scoring
            ranks = race_ref_data['team_rank'].values
            relevance_labels = [max(6 - rank, 0) if rank <= 5 else 0 for rank in ranks]
            
            meta_features.append(combined_features)
            meta_labels.extend(relevance_labels)
            meta_groups.append(len(race_ref_data))
        
        # Stack all features
        X_meta = np.vstack(meta_features)
        y_meta = np.array(meta_labels)
        
        return X_meta, y_meta, meta_groups

    def run_meta_learning_ensemble(self, train_year=2021, val_year=2022, test_year=2023):
        """
        Run meta-learning ensemble method
        
        Args:
            train_year (int): Year for meta-model training
            val_year (int): Year for meta-model validation
            test_year (int): Year for final testing
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print("\n" + "="*60)
        print("META-LEARNING ENSEMBLE")
        print("="*60)
        
        # Prepare meta-training data
        print(f"Preparing meta-training data from {train_year}...")
        X_meta_train, y_meta_train, meta_train_groups = self.prepare_meta_features_for_year(train_year)
        
        if X_meta_train is None:
            raise ValueError("No valid meta-training data found")
        
        # Prepare meta-validation data
        print(f"Preparing meta-validation data from {val_year}...")
        X_meta_val, y_meta_val, meta_val_groups = self.prepare_meta_features_for_year(val_year, one_year=True)
        
        if X_meta_val is None:
            raise ValueError("No valid meta-validation data found")
        
        # Tune meta-model hyperparameters
        print("Tuning meta-model hyperparameters with early stopping...")
        
        param_grid = get_hyperparameter_grid()
        import itertools
        keys, values = zip(*param_grid.items())
        all_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Calculate total combinations
        total_combinations = len(all_param_combinations)
        print(f"Testing {total_combinations} parameter combinations with early stopping...")
        
        # Build DMatrices with meta-feature names
        trained_schemes = [scheme for scheme in self.schemes if scheme in self.base_experiment.base_models]
        meta_feature_names = [f'base_pred_{scheme}' for scheme in trained_schemes]
        context_feature_names = [
            'cluster_flat', 'cluster_hills_flat_finish', 'cluster_hills_uphill_finish', 
            'cluster_mountains_flat_finish', 'cluster_mountains_uphill_finish',
            'race_class_ordinal', 'team_count', 'is_stage_race'
        ]
        meta_feature_names.extend(context_feature_names)
        
        dmeta_train = xgb.DMatrix(X_meta_train, label=y_meta_train, feature_names=meta_feature_names)
        dmeta_train.set_group(meta_train_groups)
        
        dmeta_val = xgb.DMatrix(X_meta_val, label=y_meta_val, feature_names=meta_feature_names)
        dmeta_val.set_group(meta_val_groups)
        
        best_ndcg = 0
        best_params = None
        best_num_rounds = 0
        
        # Early stopping configuration
        max_boost_rounds = 1000  # Set high limit
        early_stopping_rounds = 100  # Stop if no improvement for 100 rounds
        
        print(f"Early stopping config: max_rounds={max_boost_rounds}, early_stop={early_stopping_rounds}")
        
        for i, hyperparams_dict in enumerate(tqdm(all_param_combinations, desc='Meta-Model Hyperparameter Tuning')):  
            try:
                hyperparams_dict = hyperparams_dict.copy()
                hyperparams_dict['objective'] = 'rank:ndcg'
                hyperparams_dict['eval_metric'] = f'ndcg@{self.k_value}'
                
                # Train meta-model with early stopping
                evals = [(dmeta_train, 'train'), (dmeta_val, 'validation')]
                meta_model = xgb.train(
                    hyperparams_dict, 
                    dmeta_train, 
                    num_boost_round=max_boost_rounds,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False  # Suppress XGBoost output during tuning
                )
                
                # Get the optimal number of rounds from early stopping
                optimal_rounds = meta_model.best_iteration + 1  # XGBoost uses 0-based indexing
                
                # Evaluate on validation using the early-stopped model
                val_predictions = meta_model.predict(dmeta_val, iteration_range=(0, meta_model.best_iteration + 1))
                
                # Calculate NDCG@5 per race
                ndcg_scores = []
                start_idx = 0
                
                for group_size in meta_val_groups:
                    end_idx = start_idx + group_size
                    
                    # Get predictions and labels for this race
                    race_preds = val_predictions[start_idx:end_idx]
                    race_labels = y_meta_val[start_idx:end_idx]
                    
                    # Create dummy team IDs for NDCG calculation
                    team_ids = [f"team_{i}" for i in range(len(race_preds))]
                    
                    # Sort by prediction score (descending)
                    pred_order = [team_ids[i] for i in np.argsort(race_preds)[::-1]]
                    
                    # Sort by actual relevance (descending) 
                    actual_order = [team_ids[i] for i in np.argsort(race_labels)[::-1]]
                    
                    # Calculate NDCG@5
                    from roster_ranker.core import ndcg_at_k
                    ndcg5 = ndcg_at_k(pred_order, actual_order, 5)
                    if not np.isnan(ndcg5):
                        ndcg_scores.append(ndcg5)
                    
                    start_idx = end_idx
                
                if len(ndcg_scores) == 0:
                    print(f"Warning: No valid NDCG scores for meta-model combination {i+1}")
                    continue
                
                # Mean NDCG@5 across all validation races
                mean_ndcg = np.mean(ndcg_scores)
                
                if mean_ndcg > best_ndcg:
                    best_ndcg = mean_ndcg
                    best_params = hyperparams_dict.copy()
                    best_num_rounds = optimal_rounds
                    print(f"New best meta-model NDCG@{self.k_value}: {best_ndcg:.4f} (rounds: {best_num_rounds}, combination {i+1}/{total_combinations})")
                    
            except Exception as e:
                print(f"Error with meta-model combination {i+1}: {e}")
                continue
        
        if best_params is None:
            raise ValueError("Meta-model hyperparameter tuning failed")
        
        # Add the optimal number of rounds to best params for final training
        best_params['optimal_num_boost_round'] = best_num_rounds
        
        # Train final meta-model on combined data
        print(f"Training final meta-model on combined {train_year}+{val_year} data...")
        print(f"Using {best_num_rounds} boosting rounds from hyperparameter tuning...")
        
        # Combine training and validation data
        X_meta_combined = np.vstack([X_meta_train, X_meta_val])
        y_meta_combined = np.concatenate([y_meta_train, y_meta_val])
        meta_groups_combined = meta_train_groups + meta_val_groups
        
        dmeta_final = xgb.DMatrix(X_meta_combined, label=y_meta_combined, feature_names=meta_feature_names)
        dmeta_final.set_group(meta_groups_combined)
        
        # Use optimal number of boost rounds found during hyperparameter tuning
        final_params = best_params.copy()
        optimal_rounds = final_params.pop('optimal_num_boost_round', 300)
        
        meta_model = xgb.train(final_params, dmeta_final, num_boost_round=optimal_rounds)
        
        print(f"✓ Meta-model trained on {len(meta_groups_combined)} races")
        print(f"✓ Best meta-model NDCG@{self.k_value}: {best_ndcg:.4f}")
        print(f"✓ Optimal boosting rounds: {optimal_rounds}")
        
        # Get test predictions
        print(f"Generating ensemble predictions for {test_year}...")
        X_meta_test, _, meta_groups_test = self.prepare_meta_features_for_year(test_year, one_year=True)
        
        if X_meta_test is None:
            raise ValueError("No valid meta-test data found")

        # Predict with meta-model
        dmeta_test = xgb.DMatrix(X_meta_test, feature_names=meta_feature_names)
        ensemble_scores = meta_model.predict(dmeta_test)
        
        # Get reference data to construct result dataframe
        test_predictions = self.get_base_predictions_for_year(test_year, one_year=True)
        reference_data = list(test_predictions.values())[0]
        
        # Build result dataframe by grouping races
        ensemble_results = []
        score_idx = 0
        
        for (race_name, race_date), race_subset in reference_data.groupby(['race', 'date']):
            race_subset = race_subset.copy()
            
            # Sort teams consistently
            rank_col, record_id = get_ranking_config(self.level)
            race_subset = race_subset.sort_values(record_id).reset_index(drop=True)
            
            # Extract ensemble scores for this race
            race_size = len(race_subset)
            race_scores = ensemble_scores[score_idx:score_idx + race_size]
            
            # Add scores to race data
            race_subset['ensemble_score'] = race_scores
            ensemble_results.append(race_subset[[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score']])
            
            score_idx += race_size
        
        if ensemble_results:
            ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        else:
            ensemble_df = pd.DataFrame(columns=[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score'])

        # Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Meta-learning ensemble completed!")
        print(f"   Mean NDCG@5: {ensemble_eval['NDCG@5'].mean():.4f}")
        print(f"   Mean Recall@5: {ensemble_eval['Recall@5'].mean():.4f}")
        
        # Extract feature importance (use the same meta_feature_names we defined earlier)
        try:
            meta_importance_df = get_feature_importance(meta_model, meta_feature_names, importance_type='gain')
            # Convert DataFrame to dictionary and get ALL features (not just top 10)
            all_meta_importance = dict(zip(meta_importance_df['feature'], meta_importance_df['importance']))
            
            # Print top 10 features for logging
            print(f"Top 10 meta-model features:")
            for i, (_, row) in enumerate(meta_importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
                
        except Exception as e:
            print(f"⚠ Could not extract meta-model feature importance: {e}")
            raise e
        
        # Store results
        self.ensemble_results['meta_learning'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval),
            'meta_model': meta_model,
            'meta_hyperparams': best_params
        }
        self.ensemble_models['meta_learning'] = meta_model
        self.feature_importance['meta_learning'] = all_meta_importance
        
        return ensemble_eval

    def run_static_moe_ensemble(self, test_year=2023):
        """
        Run static Mixture of Experts ensemble method
        
        Uses cluster-specific expert selection based on pre-computed validation performance
        from the base model training phase. Each race cluster uses the scheme that 
        performs best on that cluster type.
        
        Args:
            test_year (int): Year for final testing
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print("\n" + "="*60)
        print("STATIC MIXTURE OF EXPERTS ENSEMBLE")
        print("="*60)
        
        # Step 1: Use pre-computed cluster performance from base experiment
        if not hasattr(self.base_experiment, 'cluster_performance') or not self.base_experiment.cluster_performance:
            raise ValueError("No cluster performance data available from base experiment. "
                           "Make sure base models have been trained with cluster evaluation.")
        
        cluster_performance = self.base_experiment.cluster_performance
        print(f"✓ Using pre-computed cluster performance from base experiment")
        print(f"✓ Available clusters: {list(cluster_performance.keys())}")
        
        # Step 2: Select best expert for each cluster+race_class based on validation performance
        print(f"\nStep 2: Selecting best expert for each cluster + race class...")
        cluster_expert_selection = {}
        
        for cluster_race_key, scheme_scores in cluster_performance.items():
                
            # Find best performing scheme for this cluster+race_class combination
            best_scheme = max(scheme_scores.items(), key=lambda x: x[1])
            cluster_expert_selection[cluster_race_key] = best_scheme[0]
            
            # Format display key
            if isinstance(cluster_race_key, tuple):
                display_key = f"{cluster_race_key[0]} + {cluster_race_key[1]}"
            else:
                display_key = str(cluster_race_key)
            
            print(f"   {display_key:<35} → {best_scheme[0]} (NDCG@{self.k_value}: {best_scheme[1]:.4f})")
            
            # Show all scheme performances for this cluster+race_class
            sorted_schemes = sorted(scheme_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (scheme, score) in enumerate(sorted_schemes):
                marker = "🏆" if i == 0 else "  "
                print(f"     {marker} {scheme:<15}: {score:.4f}")
        
        print(f"\n🎯 Expert Selection Summary:")
        for cluster_race_key, expert in cluster_expert_selection.items():
            if isinstance(cluster_race_key, tuple):
                display_key = f"{cluster_race_key[0]} + {cluster_race_key[1]}"
            else:
                display_key = str(cluster_race_key)
            print(f"   {display_key} → {expert}")
        
        # Step 3: Apply static MoE to test data
        print(f"\nApplying static MoE to test data from {test_year}...")
        test_predictions = self.get_base_predictions_for_year(test_year, one_year=True)
        
        if not test_predictions:
            raise ValueError("No test predictions available")
        
        # Build ensemble predictions using cluster-specific experts
        ensemble_results = []
        
        # Get reference data for iteration
        reference_test_data = list(test_predictions.values())[0]
        
        for (race_name, race_date), race_group in reference_test_data.groupby(['race', 'date']):
            race_group = race_group.copy()
            race_cluster = race_group['cluster'].iloc[0]
            race_class = race_group['race_class'].iloc[0]
            
            # Get expert scheme based on optimization strategy
            if self.optimization_strategy == 'cluster_race_class':
                lookup_key = (race_cluster, race_class)
            else:  # cluster_only
                lookup_key = race_cluster
            
            expert_scheme = cluster_expert_selection.get(lookup_key)
            
            if expert_scheme is None:
                # Fallback: use the first available scheme if exact match not found
                available_schemes = list(test_predictions.keys())
                expert_scheme = available_schemes[0]
                print(f"⚠ No expert found for {lookup_key}, using fallback: {expert_scheme}")

            # Get expert predictions for this race
            expert_data = test_predictions[expert_scheme]
            expert_race_data = expert_data[
                (expert_data['race'] == race_name) & 
                (expert_data['date'] == race_date)
            ].copy()
            

            # Sort teams consistently and add ensemble scores
            rank_col, record_id = get_ranking_config(self.level)
            expert_race_data = expert_race_data.sort_values(record_id).reset_index(drop=True)
            expert_race_data['ensemble_score'] = expert_race_data['pred_score']
            expert_race_data['selected_expert'] = expert_scheme
            
            ensemble_results.append(
                expert_race_data[[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score', 'selected_expert']]
            )
        
        ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        
        # Step 4: Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Static MoE ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {ensemble_eval[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {ensemble_eval[f'Recall@{self.k_value}'].mean():.4f}")
        
        # Expert usage statistics
        expert_usage = ensemble_df['selected_expert'].value_counts()
        print(f"\n📊 Expert Usage:")
        for expert, count in expert_usage.items():
            print(f"   {expert}: {count} races ({count/len(ensemble_df)*100:.1f}%)")
        
        # Store results with expert selection info
        self.ensemble_results['static_moe'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval),
            'expert_selection': cluster_expert_selection,
            'cluster_performance': cluster_performance,  # Use the base experiment's cluster performance
            'expert_usage': expert_usage.to_dict()
        }
        
        return ensemble_eval

    def run_scheme_specialists_ensemble(self, test_year=2023, selection_metric='ndcg@3'):
        """
        Run scheme specialists ensemble method
        
        Uses race type × scheme performance matrix to assign each race type to its 
        best performing scheme, then trains specialist models for each scheme on 
        only its assigned race types with dedicated hyperparameter tuning.
        
        Args:
            test_year (int): Year for final testing
            selection_metric (str): Metric for expert selection ('ndcg@3', 'ndcg@5', 'recall@3', 'recall@5')
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print("\n" + "="*60)
        print("SCHEME SPECIALISTS ENSEMBLE")
        print("="*60)
        print(f"Selection metric: {selection_metric}")
        
        # Step 1: Load race type × scheme performance matrix
        if not hasattr(self.base_experiment, 'race_type_scheme_matrix') or not self.base_experiment.race_type_scheme_matrix:
            raise ValueError("No race type × scheme matrix available from base experiment. "
                           "Make sure base models have been trained with race type matrix computation.")
        
        matrix = self.base_experiment.race_type_scheme_matrix
        print(f"✓ Using race type × scheme performance matrix from base experiment")
        print(f"✓ Available race types: {list(matrix.keys())}")
        
        # Validate selection metric
        valid_metrics = ['ndcg@3', 'ndcg@5', 'recall@3', 'recall@5']
        if selection_metric not in valid_metrics:
            raise ValueError(f"Invalid selection metric: {selection_metric}. Must be one of {valid_metrics}")
        
        # Step 2: Assign race types to best performing schemes
        print(f"\nStep 2: Assigning race types to specialist schemes (using {selection_metric})...")
        race_type_assignments = {}
        scheme_assignments = defaultdict(list)  # scheme -> list of race types
        
        for race_type, scheme_scores in matrix.items():
            
            # Find best performing scheme for this race type based on selected metric
            best_scheme = max(scheme_scores.items(), key=lambda x: x[1][selection_metric])
            race_type_assignments[race_type] = best_scheme[0]
            scheme_assignments[best_scheme[0]].append(race_type)
            
            print(f"   {race_type:<25} → {best_scheme[0]} ({selection_metric}: {best_scheme[1][selection_metric]:.4f})")
        
        print(f"\n🎯 Scheme Specialist Assignments:")
        for scheme, race_types in scheme_assignments.items():
            print(f"   {scheme:<15} → {', '.join(race_types)}")
        
        # Step 3: Train specialist models with dedicated hyperparameter tuning
        print(f"\n Step 3: Training specialist models with hyperparameter tuning...")
        specialist_models = {}
        specialist_hyperparameters = {}
        
        # Create specialist hyperparameters directory
        specialist_hyperparams_dir = f"{self.hyperparams_dir}/specialist_hyperparams_{selection_metric}"
        os.makedirs(specialist_hyperparams_dir, exist_ok=True)
        
        # Load full datasets for training specialists
        for scheme in scheme_assignments.keys():
            assigned_race_types = scheme_assignments[scheme]
            if not assigned_race_types:
                continue
                
            print(f"\n  Training {scheme} specialist for race types: {', '.join(assigned_race_types)}")
            
            # Load scheme data
            scheme_data = load_or_create_dataset(self.race_class, scheme, level=self.level)
            
            # Split specialist data into train/validation for hyperparameter tuning
            specialist_train_data = scheme_data[
                (scheme_data['year'] < test_year - 1) & 
                (scheme_data['cluster'].isin(assigned_race_types))
            ].copy()
            
            specialist_val_data = scheme_data[
                (scheme_data['year'] == test_year - 1) & 
                (scheme_data['cluster'].isin(assigned_race_types))
            ].copy()

            print(f"    ✓ Specialist train data: {len(specialist_train_data)} records")
            print(f"    ✓ Specialist validation data: {len(specialist_val_data)} records")
            print(f"    ✓ Race types: {specialist_train_data['cluster'].value_counts().to_dict()}")
            
            # Prepare features
            feature_columns = [col for col in specialist_train_data.columns if col not in EXCLUDE_COLS]
            
            # Check for saved specialist hyperparameters
            specialist_hyperparams_file = f"{specialist_hyperparams_dir}/{scheme}_specialist_hyperparams.json"
            
            if os.path.exists(specialist_hyperparams_file):
                print(f"    ✓ Loading saved specialist hyperparameters...")
                try:
                    with open(specialist_hyperparams_file, 'r') as f:
                        specialist_best_params = json.load(f)
                    print(f"    ✓ Using saved specialist hyperparameters: {specialist_best_params}")
                except Exception as e:
                    print(f"    ⚠ Could not load saved hyperparameters: {e}, tuning new ones...")
                    specialist_best_params = None
            else:
                specialist_best_params = None
            
            if specialist_best_params is None:
                # Tune hyperparameters specifically for this specialist's data
                print(f"    🔧 Tuning hyperparameters for {scheme} specialist...")
                specialist_best_params, specialist_best_ndcg = tune_hyperparameters(
                    specialist_train_data, specialist_val_data, feature_columns
                )
                print(f"    ✓ Specialist best params: {specialist_best_params}")
                print(f"    ✓ Specialist best validation NDCG@{self.k_value}: {specialist_best_ndcg:.4f}")
                
                # Save specialist hyperparameters
                try:
                    with open(specialist_hyperparams_file, 'w') as f:
                        json.dump(specialist_best_params, f, indent=2)
                    print(f"    ✓ Specialist hyperparameters saved")
                except Exception as e:
                    print(f"    ⚠ Could not save specialist hyperparameters: {e}")
            
            # Combine train + validation for final specialist model
            specialist_full_data = pd.concat([specialist_train_data, specialist_val_data], ignore_index=True)
            print(f"    ✓ Final specialist training data: {len(specialist_full_data)} records")
            
            # Train specialist model with tuned hyperparameters
            specialist_model = train_model(
                specialist_full_data, feature_columns, specialist_best_params, 
                f"{scheme}_specialist"
            )
            specialist_models[scheme] = specialist_model
            specialist_hyperparameters[scheme] = specialist_best_params
            print(f"    ✓ {scheme} specialist trained successfully")
        
        print(f"\n✅ Trained {len(specialist_models)} specialist models")
        
        # Step 4: Apply specialists to test data
        print(f"\nStep 4: Applying specialists to test data from {test_year}...")
        
        ensemble_results = []
        specialist_usage = defaultdict(int)
        
        # Process each scheme's test data
        for scheme in self.schemes:
            if scheme not in specialist_models:
                print(f"⚠ No specialist model for {scheme}, skipping...")
                continue
            
            # Load test data for this scheme
            scheme_data = load_or_create_dataset(self.race_class, scheme, level=self.level, time_gap=self.time_gap, exp_name=self.exp_name)
            test_data = scheme_data[scheme_data['year'] == test_year].copy()
            
            if len(test_data) == 0:
                continue
            
            # Filter to only race types assigned to this specialist
            assigned_race_types = scheme_assignments[scheme]
            specialist_test_data = test_data[test_data['cluster'].isin(assigned_race_types)].copy()
            
            if len(specialist_test_data) == 0:
                continue
            
            print(f"  {scheme} specialist: {len(specialist_test_data)} test records")
            
            # Generate predictions using specialist model
            feature_columns = [col for col in specialist_test_data.columns if col not in EXCLUDE_COLS]
            specialist_test_data = specialist_test_data.sort_values(['race', 'date', 'team']).reset_index(drop=True)
            
            X_test = specialist_test_data[feature_columns].values
            dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
            specialist_scores = specialist_models[scheme].predict(dtest)
            
            # Create results dataframe
            specialist_results = specialist_test_data[['race', 'date', 'team', 'cluster', 'team_rank']].copy()
            specialist_results['ensemble_score'] = specialist_scores
            specialist_results['selected_specialist'] = scheme
            
            ensemble_results.append(specialist_results)
            specialist_usage[scheme] += len(specialist_results.groupby(['race', 'date']))
        
        if ensemble_results:
            ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        else:
            ensemble_df = pd.DataFrame(columns=['race', 'date', 'team', 'cluster', 'ensemble_score', 'team_rank', 'selected_specialist'])
        
        # Step 5: Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Scheme specialists ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {ensemble_eval[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {ensemble_eval[f'Recall@{self.k_value}'].mean():.4f}")
        
        # Specialist usage statistics
        print(f"\n📊 Specialist Usage:")
        total_races = sum(specialist_usage.values())
        for specialist, count in specialist_usage.items():
            percentage = (count / total_races) * 100 if total_races > 0 else 0
            race_types = ', '.join(scheme_assignments.get(specialist, []))
            print(f"   {specialist:<15}: {count:3d} races ({percentage:5.1f}%) - {race_types}")
        
        # Store results
        self.ensemble_results['scheme_specialists'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval),
            'race_type_assignments': race_type_assignments,
            'scheme_assignments': dict(scheme_assignments),
            'race_type_scheme_matrix': matrix,
            'specialist_usage': specialist_usage,
            'specialist_models': specialist_models,
            'specialist_hyperparameters': specialist_hyperparameters,
            'selection_metric': selection_metric
        }
        
        return ensemble_eval

    def evaluate_ensemble_predictions(self, ensemble_predictions):
        """
        Evaluate ensemble predictions using core metrics functions
        
        Args:
            ensemble_predictions (pd.DataFrame): Predictions with ensemble_score and team_rank
            
        Returns:
            pd.DataFrame: Evaluation results per race
        """
        # Add race_id for grouping and sort consistently
        ensemble_predictions = ensemble_predictions.copy()
        ensemble_predictions['race_id'] = ensemble_predictions['race'] + "_" + ensemble_predictions['date'].astype(str)
        ensemble_predictions = ensemble_predictions.sort_values('race_id').reset_index(drop=True)
        
        results = []
        rank_col, record_id = get_ranking_config(self.level)
        # Use groupby to ensure consistent race iteration order
        for race_id, pred_race in ensemble_predictions.groupby('race_id'):
            # Sort teams within each race to ensure alignment
            pred_race = pred_race.sort_values(record_id).reset_index(drop=True)
            
            # Create evaluation DataFrame with required columns
            race_combined = pred_race[[rank_col, record_id, 'ensemble_score']].copy()
            race_combined['pred_score'] = race_combined['ensemble_score']
            
            # Use core metrics function to evaluate this race
            race_metrics = evaluate_race_predictions(race_combined, k_values=[3, 5, 10], level=self.level)
            
            # Add race metadata
            race_result = {
                'race': pred_race['race'].iloc[0],
                'date': pred_race['date'].iloc[0],
                'cluster': pred_race['cluster'].iloc[0],
                'race_class': pred_race['race_class'].iloc[0],
                'num_teams': len(race_combined),
                **race_metrics
            }
            
            results.append(race_result)
        
        return pd.DataFrame(results)

    def run_all_ensemble_methods(self, force_retune_base=False):
        """
        Run all ensemble methods
        
        Args:
            force_retune_base (bool): Force hyperparameter retuning for base models
            
        Returns:
            dict: Results for all ensemble methods
        """
        print("="*80)
        print("ENSEMBLE EXPERIMENT PIPELINE")
        print("="*80)
        print(f"Race class: {self.race_class}")
        print(f"Ensemble methods: {self.ensemble_methods}")
        print(f"Time gap: {self.time_gap}")
    
        # Step 1: Run each ensemble method
        all_results = {}
        
        for method in self.ensemble_methods:
            try:
                print(f"\n{'='*60}")
                print(f"RUNNING {method.upper()} ENSEMBLE")
                print(f"{'='*60}")
                
                if method == 'simple_average':
                    result = self.run_simple_average_ensemble()
                elif method == 'meta_learning':
                    result = self.run_meta_learning_ensemble()
                elif method == 'static_moe':
                    result = self.run_static_moe_ensemble()
                elif method == 'scheme_specialists':
                    result = self.run_scheme_specialists_ensemble()
                elif method == 'adaptive_feature_selection':
                    result = self.run_adaptive_feature_selection_ensemble()
                elif method == 'hard_moe_gating':
                    result = self.run_hard_moe_gating_ensemble(gating_model_type=self.gating_model_type)
                elif method == 'soft_moe_gating':
                    result = self.run_soft_moe_gating_ensemble(gating_model_type=self.gating_model_type)
                else:
                    print(f"⚠ Unknown ensemble method: {method}")
                    continue
                
                all_results[method] = result
                
            except Exception as e:
                print(f"❌ Error running {method} ensemble: {e}")
                raise e
        
        # Step 2: Save comprehensive results
        self.save_all_results()
        
        print(f"\n{'='*80}")
        print("ENSEMBLE EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        print(f"✅ Race class: {self.race_class}")
        print(f"✅ Successful ensemble methods: {list(all_results.keys())}")
        
        return all_results

    def save_all_results(self):
        """
        Save comprehensive results for all ensemble methods
        """
        print("\n" + "="*60)
        print("SAVING COMPREHENSIVE RESULTS")
        print("="*60)
        
        # Create comprehensive summary
        summary = {
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'race_class': self.race_class,
            'schemes': self.schemes,
            'ensemble_methods': self.ensemble_methods,
            'base_models': {},
            'ensemble_results': {},
            'feature_importance': {}
        }
        
        # Base model statistics - simplified
        if self.base_experiment and self.base_experiment.base_results:
            for scheme, results_df in self.base_experiment.base_results.items():
                scheme_stats = calculate_summary_statistics(results_df)
                # Extract only essential statistics
                simplified_stats = {}
                for metric, stats_dict in scheme_stats.items():
                    if isinstance(stats_dict, dict):
                        simplified_stats[metric] = {
                            'mean': round(stats_dict.get('mean', 0.0), 4),
                            'std': round(stats_dict.get('std', 0.0), 4),
                            'count': stats_dict.get('count', 0)
                        }
                
                # Add feature importance summary for base model
                if hasattr(self.base_experiment, 'base_feature_importance') and scheme in self.base_experiment.base_feature_importance:
                    feature_count = len(self.base_experiment.base_feature_importance[scheme])
                    simplified_stats['feature_importance_count'] = feature_count
                
                summary['base_models'][scheme] = simplified_stats
        
        # Ensemble statistics - simplified
        for method, method_results in self.ensemble_results.items():
            ensemble_summary = {}
            
            if 'summary_stats' in method_results:
                stats = method_results['summary_stats']
                simplified_stats = {}
                for metric, stats_dict in stats.items():
                    if isinstance(stats_dict, dict):
                        simplified_stats[metric] = {
                            'mean': round(stats_dict.get('mean', 0.0), 4),
                            'std': round(stats_dict.get('std', 0.0), 4),
                            'count': stats_dict.get('count', 0)
                        }
                ensemble_summary.update(simplified_stats)
            
            # Add hyperparameters for meta-learning
            if method == 'meta_learning' and 'meta_hyperparams' in method_results:
                ensemble_summary['meta_hyperparams'] = method_results['meta_hyperparams']
            
            summary['ensemble_results'][method] = ensemble_summary
        
        # Feature importance - all features and counts
        summary['feature_importance_summary'] = {}
        if self.feature_importance:
            for method, importance_dict in self.feature_importance.items():
                if isinstance(importance_dict, dict):
                    # Since feature_importance is now a clean dict of {feature_name: float_value}
                    # we just need to ensure all values are JSON-serializable
                    serializable_features = {}
                    for feature, importance in importance_dict.items():
                        try:
                            serializable_features[str(feature)] = float(importance)
                        except (ValueError, TypeError):
                            print(f"⚠ Could not convert feature importance for {feature}: {importance}")
                            serializable_features[str(feature)] = 0.0
                    
                    # Sort by importance and get top 10 for summary
                    sorted_features = sorted(serializable_features.items(), key=lambda x: x[1], reverse=True)
                    top_10_features = dict(sorted_features[:10])
                    
                    summary['feature_importance_summary'][method] = {
                        'top_10_features': top_10_features,  # Top 10 for summary display
                        'total_features': len(serializable_features)
                    }
        
        # Save simplified summary
        if self.ensemble_methods == ['simple_average', 'meta_learning', 'static_moe', 'scheme_specialists', 'adaptive_feature_selection']:
            method = 'all'
        elif self.ensemble_methods == ['simple_average']:
            method = 'simple_average'
        elif self.ensemble_methods == ['meta_learning']:
            method = 'meta_learning'
        elif self.ensemble_methods == ['static_moe']:
            method = 'static_moe'
            method += f"_{self.optimization_strategy}"
        elif self.ensemble_methods == ['scheme_specialists']:
            method = 'scheme_specialists'
        elif self.ensemble_methods == ['adaptive_feature_selection']:
            method = 'afs'
            method += f"_{self.optimization_strategy}"
        elif self.ensemble_methods == ['hard_moe_gating']:
            method = 'hard_moe' + f"_{self.gating_model_type}"
        elif self.ensemble_methods == ['soft_moe_gating']:
            method = 'soft_moe' + f"_{self.gating_model_type}"
        else:
            method = 'mixed'

        summary_file = f'{self.results_dir}/{method}_results_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save individual method results as CSV files
        for method, method_results in self.ensemble_results.items():
            method_dir = f'{self.results_dir}/{method}'
            os.makedirs(method_dir, exist_ok=True)
            
            # Save evaluation results
            if 'evaluation' in method_results and method_results['evaluation'] is not None:
                eval_file = f'{method_dir}/evaluation_results.csv'
                method_results['evaluation'].to_csv(eval_file, index=False)
            
            # Save predictions
            if 'predictions' in method_results and method_results['predictions'] is not None:
                pred_file = f'{method_dir}/predictions.csv'
                method_results['predictions'].to_csv(pred_file, index=False)
        
        # Save base model results as CSV files
        if self.base_experiment and self.base_experiment.base_results:
            base_models_dir = f'{self.results_dir}/base_models'
            os.makedirs(base_models_dir, exist_ok=True)
            
            for scheme, results_df in self.base_experiment.base_results.items():
                base_file = f'{base_models_dir}/{scheme}_results.csv'
                results_df.to_csv(base_file, index=False)
                
                # Save base model feature importance if available
                if hasattr(self.base_experiment, 'base_feature_importance') and scheme in self.base_experiment.base_feature_importance:
                    base_importance_file = f'{base_models_dir}/{scheme}_feature_importance.csv'
                    self.base_experiment.base_feature_importance[scheme].to_csv(base_importance_file, index=False)
                    print(f"✓ Base model feature importance saved: {base_importance_file}")
        
        # Save detailed feature importance as separate files
        if self.feature_importance:
            importance_dir = f'{self.results_dir}/feature_importance'
            os.makedirs(importance_dir, exist_ok=True)
            
            for method, importance_dict in self.feature_importance.items():
                if isinstance(importance_dict, dict):
                    importance_file = f'{importance_dir}/{method}_feature_importance.csv'
                    importance_df = pd.DataFrame([
                        {'feature': feature, 'importance': importance}
                        for feature, importance in importance_dict.items()
                    ])
                    importance_df.to_csv(importance_file, index=False)
        
        print(f"✓ Simplified summary saved to: {summary_file}")
        print(f"✓ Detailed results saved to CSV files in method subdirectories")
        print(f"✓ Base model results saved to: {self.results_dir}/base_models/")
        print(f"✓ Base model feature importance saved to: {self.results_dir}/base_models/")
        print(f"✓ Ensemble feature importance saved to: {self.results_dir}/feature_importance/")

    def get_best_ensemble_method(self, metric='NDCG@5'):
        """
        Get the best performing ensemble method
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            str: Best ensemble method name
        """
        if not self.ensemble_results:
            raise ValueError("No ensemble experiments have been run yet")
        
        best_score = -1
        best_method = None
        
        for method, method_results in self.ensemble_results.items():
            if 'evaluation' in method_results and method_results['evaluation'] is not None:
                eval_df = method_results['evaluation']
                score = eval_df[metric].mean() if metric in eval_df.columns else 0
                
                if score > best_score:
                    best_score = score
                    best_method = method
        
        return best_method

    def create_adaptive_feature_selection_dataset(self, race_class, save_dataset=True):
        """
        Create unified dataset where each race uses features from its optimal scheme
        based on cluster+race_class performance matrix from base models
        
        Args:
            race_class (str): Race class ('all' or 'WT')
            save_dataset (bool): Whether to save the unified dataset
            
        Returns:
            pd.DataFrame: Unified dataset with optimal features per race
        """
        print("\n" + "="*60)
        print("CREATING ADAPTIVE FEATURE SELECTION DATASET")
        print("="*60)
        
        # Step 1: Get cluster performance data from base experiment
        if not hasattr(self.base_experiment, 'cluster_performance') or not self.base_experiment.cluster_performance:
            raise ValueError("No cluster performance data available from base experiment. "
                           "Make sure base models have been trained with cluster evaluation.")
        
        cluster_performance = self.base_experiment.cluster_performance
        print(f"✓ Using cluster performance from base experiment")
        print(f"✓ Available combinations: {len(cluster_performance)} ({self.optimization_strategy})")
        
        # Step 2: Create expert selection mapping
        cluster_expert_selection = {}
        for cluster_race_key, scheme_scores in cluster_performance.items():

            # Find best performing scheme for this combination
            best_scheme = max(scheme_scores.items(), key=lambda x: x[1])
            cluster_expert_selection[cluster_race_key] = best_scheme[0]
            
            # Format display key
            if isinstance(cluster_race_key, tuple):
                display_key = f"{cluster_race_key[0]} + {cluster_race_key[1]}"
            else:
                display_key = str(cluster_race_key)
            print(f"   {display_key:<35} → {best_scheme[0]} (NDCG@{self.k_value}: {best_scheme[1]:.4f})")
        
        # Step 3: Load all scheme datasets
        print(f"\nLoading scheme datasets...")
        scheme_datasets = {}
        for scheme in self.schemes:
            scheme_data = load_or_create_dataset(race_class, scheme, level=self.level, exp_name=self.exp_name, time_gap=self.time_gap)
            scheme_data['source_scheme'] = scheme  # Track which scheme each row came from
            scheme_datasets[scheme] = scheme_data
            print(f"   ✓ {scheme}: {len(scheme_data)} records")
        
        # Step 4: Build unified dataset with optimal features
        print(f"\nBuilding unified dataset with optimal features...")
        unified_records = []
        scheme_usage_count = defaultdict(int)
        
        # Get reference dataset to find all unique combinations
        reference_scheme = list(scheme_datasets.keys())[0]
        reference_data = scheme_datasets[reference_scheme]
        
        if self.optimization_strategy == 'cluster_race_class':
            unique_combinations = reference_data[['cluster', 'race_class']].drop_duplicates()
            combination_col = 'cluster+race_class'
        else:  # cluster_only
            unique_combinations = reference_data[['cluster']].drop_duplicates()
            combination_col = 'cluster'
        
        print(f"Processing {len(unique_combinations)} unique {combination_col} combinations...")
        
        for _, combo_row in unique_combinations.iterrows():
            race_cluster = combo_row['cluster']
            
            if self.optimization_strategy == 'cluster_race_class':
                race_class_val = combo_row['race_class']
                lookup_key = (race_cluster, race_class_val)
                display_key = f"{race_cluster} + {race_class_val}"
                
                # Filter data for this cluster+race_class combination
                filter_condition = (
                    (reference_data['cluster'] == race_cluster) & 
                    (reference_data['race_class'] == race_class_val)
                )
                cluster_key_value = f"{race_cluster}|{race_class_val}"
            else:  # cluster_only
                lookup_key = race_cluster
                display_key = f"{race_cluster}"
                
                # Filter data for this cluster only
                filter_condition = (reference_data['cluster'] == race_cluster)
                cluster_key_value = race_cluster
            
            # Determine optimal scheme for this combination
            optimal_scheme = cluster_expert_selection.get(lookup_key)
            
            if optimal_scheme is None:
                print(f"⚠ No optimal scheme found for {lookup_key}, skipping...")
                continue
            
            # Filter optimal scheme dataset for this combination
            optimal_scheme_data = scheme_datasets[optimal_scheme]
            cluster_data = optimal_scheme_data[filter_condition].copy()
            
            if len(cluster_data) > 0:
                # Add metadata about scheme selection
                cluster_data['selected_scheme'] = optimal_scheme
                cluster_data['cluster_race_key'] = cluster_key_value
                unified_records.append(cluster_data)
                scheme_usage_count[optimal_scheme] += len(cluster_data)
                
                print(f"   {display_key:<35} → {optimal_scheme:<15} ({len(cluster_data):>4} records)")
            else:
                print(f"⚠ No data found for {display_key} in {optimal_scheme}")
        print(f"✓ Processed all combinations efficiently using vectorized filtering")
        
        # Step 5: Combine all records
        if unified_records:
            unified_df = pd.concat(unified_records, ignore_index=True)
        else:
            raise ValueError("No records found for unified dataset")
        
        print(f"\n✅ Unified dataset created!")
        print(f"   Total records: {len(unified_df)}")
        print(f"   Unique races: {unified_df[['race', 'date']].drop_duplicates().shape[0]}")
        print(f"   Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")

        # Scheme usage statistics
        print(f"\n📊 Scheme Usage in Unified Dataset:")
        total_records = sum(scheme_usage_count.values())
        for scheme, count in sorted(scheme_usage_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_records) * 100
            print(f"   {scheme:<15}: {count:>6} records ({percentage:>5.1f}%)")
        
        # Step 6: Save dataset if requested
        if save_dataset:
            from roster_ranker.utils import get_data_dir
            dataset_dir = get_data_dir('roster_datasets')
            filename = f'roster_dataset_{race_class}_adaptive_feature_selection.csv'
            output_path = f"{dataset_dir}/{filename}"
            
            try:
                unified_df.to_csv(output_path, index=False)
                print(f"\n✓ Unified dataset saved to: {output_path}")
            except Exception as e:
                print(f"⚠ Could not save unified dataset: {e}")
        
        return unified_df

    def run_adaptive_feature_selection_ensemble(self, test_year=2023, force_retune=False):
        """
        Run Adaptive Feature Selection ensemble method
        
        Creates a unified dataset where each race uses features from its optimal scheme,
        then trains a single model on this expert-selected feature dataset.
        
        Args:
            test_year (int): Year for final testing
            force_retune (bool): Force hyperparameter retuning
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print("\n" + "="*60)
        print("ADAPTIVE FEATURE SELECTION ENSEMBLE")
        print("="*60)
        
        # Step 1: Create unified dataset with optimal features
        unified_dataset = self.create_adaptive_feature_selection_dataset(self.race_class)
        
        # Step 2: Split data for training
        print(f"\nSplitting unified dataset...")
        train_df = unified_dataset[unified_dataset['year'] < test_year - 1].copy()
        val_df = unified_dataset[unified_dataset['year'] == test_year - 1].copy()
        test_df = unified_dataset[unified_dataset['year'] == test_year].copy()
        
        print(f"✓ Train: {len(train_df)} records")
        print(f"✓ Validation: {len(val_df)} records")
        print(f"✓ Test: {len(test_df)} records")
        
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            raise ValueError("Insufficient data in one or more splits")
        
        # Step 3: Prepare features (exclude metadata columns)
        exclude_cols = EXCLUDE_COLS + ['source_scheme', 'selected_scheme', 'cluster_race_key']
        feature_columns = [col for col in unified_dataset.columns if col not in exclude_cols]
        print(f"✓ Feature columns: {len(feature_columns)} features")
        
        # Step 4: Hyperparameter tuning
        print(f"\nTuning hyperparameters for unified model...")
        
        # Check for saved hyperparameters
        hyperparams_file = f"{self.results_dir}/afs_hyperparams_{self.optimization_strategy}.json"
        if os.path.exists(hyperparams_file) and not force_retune:
            print(f"✓ Loading saved hyperparameters...")
            try:
                with open(hyperparams_file, 'r') as f:
                    best_params = json.load(f)
                print(f"✓ Using saved hyperparameters: {best_params}")
            except Exception as e:
                print(f"⚠ Could not load saved hyperparameters: {e}, tuning new ones...")
                best_params = None
        else:
            best_params = None
        
        if best_params is None:
            best_params, best_ndcg = tune_hyperparameters(train_df, val_df, feature_columns, k_value=self.k_value, level=self.level)
            print(f"✓ Best hyperparameters: {best_params}")
            print(f"✓ Best validation NDCG@{self.k_value}: {best_ndcg:.4f}")
            
            # Save hyperparameters
            try:
                os.makedirs(self.results_dir, exist_ok=True)
                with open(hyperparams_file, 'w') as f:
                    json.dump(best_params, f, indent=2)
                print(f"✓ Hyperparameters saved")
            except Exception as e:
                print(f"⚠ Could not save hyperparameters: {e}")
        
        # Step 5: Train final model
        print(f"\nTraining unified model...")
        # Combine train + validation for final model
        full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        
        model = train_model(full_train_df, feature_columns, best_params, "adaptive_feature_selection_model", k_value=self.k_value, level=self.level)
        print(f"✓ Unified model trained")
        
        # Step 6: Extract feature importance
        print(f"\n6. EXTRACTING FEATURE IMPORTANCE...")
        print("-" * 50)
        
        try:
            feature_importance_df = get_feature_importance(model, feature_columns, importance_type='gain')
            print(f"✓ Feature importance extracted ({len(feature_importance_df)} features)")
            
            # Print top 10 features for logging
            print(f"Top 10 most important features:")
            for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.1f}")
                
        except Exception as e:
            print(f"⚠ Feature importance extraction failed: {e}")
            feature_importance_df = None
        
        # Step 7: Evaluate on test set
        print(f"\n7. EVALUATING ON TEST SET...")
        print("-" * 50)
        test_results = evaluate_model(model, test_df, feature_columns, "adaptive_feature_selection_model", k_value=self.k_value, level=self.level)
        print(f"✓ Test evaluation completed: {len(test_results)} races")
        
        # Step 8: Add scheme selection info to results
        test_results_with_schemes = test_results.copy()
        
        # Map back the selected scheme for each race
        scheme_mapping = test_df.groupby(['race', 'date'])['selected_scheme'].first().to_dict()
        test_results_with_schemes['selected_scheme'] = test_results_with_schemes.apply(
            lambda row: scheme_mapping.get((row['race'], row['date']), 'unknown'), axis=1
        )
        
        print(f"✅ Adaptive Feature Selection ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {test_results[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {test_results[f'Recall@{self.k_value}'].mean():.4f}")
        
        # Scheme usage in test set
        test_scheme_usage = test_results_with_schemes['selected_scheme'].value_counts()
        print(f"\n📊 Test Set Scheme Usage:")
        for scheme, count in test_scheme_usage.items():
            print(f"   {scheme}: {count} races ({count/len(test_results_with_schemes)*100:.1f}%)")
        
        # Feature importance summary
        if feature_importance_df is not None:
            print(f"\n📈 Feature Importance Summary:")
            print(f"   Total features: {len(feature_importance_df)}")
            print(f"   Top feature: {feature_importance_df.iloc[0]['feature']} ({feature_importance_df.iloc[0]['importance']:.1f})")

            # Convert feature importance to dictionary for storage
            feature_importance_dict = {}
            for _, row in feature_importance_df.iterrows():
                feature_importance_dict[row['feature']] = row['importance']
            self.feature_importance['adaptive_feature_selection'] = feature_importance_dict
        
        # Store results
        self.ensemble_results['adaptive_feature_selection'] = {
            'predictions': test_results_with_schemes,
            'evaluation': test_results,
            'summary_stats': calculate_summary_statistics(test_results),
            'model': model,
            'hyperparameters': best_params,
            'feature_columns': feature_columns,
            'feature_importance': feature_importance_df,
            'test_scheme_usage': test_scheme_usage.to_dict()
        }
        
        return test_results

    def extract_gating_features(self, race_data):
        """
        Extract features for gating network based on race context
        
        Args:
            race_data (pd.DataFrame): Race data for feature extraction
            
        Returns:
            np.array: Gating network features
        """
        
        # Get race information 
        cluster = race_data['cluster'].iloc[0]
        race_class = race_data['race_class'].iloc[0]
        
        # One-hot encode cluster (6 categories)
        cluster_encoding = [1 if cluster == c else 0 for c in CLUSTERS]
        
        # One-hot encode race class (WT,Pro,1,2)
        race_class_encoding = [1 if race_class == race_class_ else 0 for race_class_ in RACE_CLASSES]
        
        # StageRace flag
        stage_race_flag = [race_data['StageRace'].iloc[0]] 
        
        # Stage number / total stages (0 for one-day races)
        stage_ratio = [race_data['stage_ratio'].iloc[0]] 

        # profile score, distance, verticalMeters
        race_attributes = [race_data['profileScore'].iloc[0], race_data['distance'].iloc[0], race_data['verticalMeters'].iloc[0]] 

        # Number of riders in the race
        num_riders = [len(race_data)]

        # Combine all features
        gating_features = cluster_encoding + race_class_encoding + stage_race_flag + stage_ratio + race_attributes + num_riders
        return np.array(gating_features, dtype=np.float32)

    def determine_winning_expert(self, race_results, base_predictions, race_key):
        """
        Determine the winning expert (best NDCG@k) for a specific race
        
        Args:
            race_results (pd.DataFrame): Actual race results
            base_predictions (dict): Predictions from all base models
            race_key (tuple): (race, date) identifier
            
        Returns:
            str: Winning scheme name
        """
        
        best_ndcg = -1
        best_recall = -1
        winning_expert = self.schemes[0]  # default
        record_id = get_record_id(self.level)
        
        # Get actual finishing order for this race
        race_actual = race_results[
            (race_results['race'] == race_key[0]) & 
            (race_results['date'] == race_key[1])
        ].copy()
        
        if len(race_actual) == 0:
            return winning_expert
            
        rank_col = get_rank_col(self.level)
        actual_order = race_actual.sort_values(rank_col)[record_id].tolist()
        
        # Test each scheme's performance
        for scheme in self.schemes:
            
            # Check if base_predictions is in aligned format (race_key -> scheme -> array)
            # or in DataFrame format (scheme -> DataFrame)
            if race_key in base_predictions:
                # Aligned format: base_predictions[race_key][scheme] = numpy array
                race_pred_scores = base_predictions[race_key][scheme]
                if len(race_pred_scores) == 0:
                    continue
                
                # For aligned format, we need to get the record IDs from race_results
                # and create the prediction order based on scores
                race_records = race_actual.sort_values(record_id)[record_id].tolist()
                
                # Create (score, record_id) pairs and sort by score descending
                score_record_pairs = list(zip(race_pred_scores, race_records))
                score_record_pairs.sort(key=lambda x: x[0], reverse=True)
                pred_order = [pair[1] for pair in score_record_pairs]
            else:
                # DataFrame format: base_predictions[scheme] = DataFrame
                pred_df = base_predictions[scheme]
                race_preds = pred_df[
                    (pred_df['race'] == race_key[0]) & 
                    (pred_df['date'] == race_key[1])
                ].copy()

                if len(race_preds) == 0:
                    continue
                    
                # Sort by prediction score (descending) for DataFrame
                pred_order = race_preds.sort_values('pred_score', ascending=False)[record_id].tolist()
            
            # Calculate NDCG@k
            ndcg = ndcg_at_k(pred_order, actual_order, self.k_value)
            recall = recall_at_k(pred_order, actual_order, self.k_value)

            if ndcg > best_ndcg or (ndcg == best_ndcg and recall > best_recall):
                best_ndcg = ndcg
                best_recall = recall
                winning_expert = scheme

        return winning_expert

    def train_gating_network(self, train_data, val_data, base_predictions_train, base_predictions_val, force_retune=False, model_type='logistic'):
        """
        Train gating network classifier to predict winning expert
        
        Args:
            train_data (pd.DataFrame): Training race data
            val_data (pd.DataFrame): Validation race data  
            base_predictions_train (dict): Base model predictions for training
            base_predictions_val (dict): Base model predictions for validation
            force_retune (bool): Force hyperparameter retuning
            model_type (str): Type of model to use ('logistic' or 'mlp')
            
        Returns:
            tuple: (classifier, scaler, best_params, val_accuracy)
        """
        print(f"\n{'='*60}")
        print("TRAINING GATING NETWORK")
        print(f"{'='*60}")
        
        # Extract gating features and labels for training
        X_train = []
        y_train = []
        
        train_races = train_data.groupby(['race', 'date'])
        
        print(f"Extracting training features from {len(train_races)} races...")
        for race_key, race_group in tqdm(train_races):
            # Extract gating features
            gating_features = self.extract_gating_features(race_group)
            X_train.append(gating_features)
            
            # Determine winning expert
            winning_expert = self.determine_winning_expert(race_group, base_predictions_train, race_key)
            expert_idx = self.schemes.index(winning_expert)
            y_train.append(expert_idx)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Extract validation features and labels
        X_val = []
        y_val = []
        
        val_races = val_data.groupby(['race', 'date'])
        
        print(f"Extracting validation features from {len(val_races)} races...")
        for race_key, race_group in tqdm(val_races):
            gating_features = self.extract_gating_features(race_group)
            X_val.append(gating_features)
            
            winning_expert = self.determine_winning_expert(race_group, base_predictions_val, race_key)
            expert_idx = self.schemes.index(winning_expert)
            y_val.append(expert_idx)
        
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Scale features since we now have mixed scales (one-hot + continuous)
        scaler = StandardScaler()  # Zero mean, unit variance
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"Training set: {len(X_train)} races, {X_train.shape[1]} features")
        print(f"Validation set: {len(X_val)} races")
        print(f"Class distribution train: {np.bincount(y_train)}")
        print(f"Class distribution val: {np.bincount(y_val)}")
        print(f"Feature scaling applied:")
        print(f"  Original X_train range: {X_train.min():.2f} to {X_train.max():.2f}")
        print(f"  Scaled X_train range: {X_train_scaled.min():.2f} to {X_train_scaled.max():.2f}")
        print(f"  Feature dimensions: {X_train.shape[1]} features")
        
        # Use scaled features for training
        X_train = X_train_scaled
        X_val = X_val_scaled

        # Check for saved hyperparameters (with model type prefix)
        hyperparams_file = f"{self.results_dir}/{model_type}_gating_network_hyperparams.json"
        best_params = None
        
        if os.path.exists(hyperparams_file) and not force_retune:
            print(f"✓ Loading saved {model_type} gating network hyperparameters...")
            try:
                with open(hyperparams_file, 'r') as f:
                    best_params = json.load(f)
                print(f"✓ Using saved hyperparameters: {best_params}")
            except Exception as e:
                print(f"⚠ Could not load saved hyperparameters: {e}, tuning new ones...")
                best_params = None
        
        if best_params is None:
            print(f"Tuning {model_type} gating network hyperparameters...")
            print(f"Using temporal validation (train on training data, validate on validation data)")
            
            if model_type == 'logistic':
                # Get logistic regression hyperparameter grid from config
                param_grid = get_logistic_regression_hyperparameter_grid()
            
            elif model_type == 'mlp':
                # Get MLP hyperparameter grid from config
                param_grid = get_mlp_hyperparameter_grid()
            
            print(f"Generated {len(param_grid)} hyperparameter combinations for {model_type} gating network")
            
            # Temporal validation: train on training data, validate on separate validation data
            best_score = -1
            
            for i, params in enumerate(tqdm(param_grid, desc=f"Tuning {model_type} gating network")):
                try:
                    # Create model based on type
                    if model_type == 'logistic':
                        clf = LogisticRegression(random_state=42, **params)
                    elif model_type == 'mlp':
                        clf = GatingMLPWrapper(random_state=42, use_class_weights=False, **params)
                    
                    # Train on training data
                    clf.fit(X_train, y_train, X_val, y_val)
                    params['best_epoch'] = clf.best_epoch

                    # Evaluate on temporally separate validation set (no data leakage)
                    val_score = clf.score(X_val, y_val)
                    val_predictions = clf.predict(X_val)
                    val_balanced_score = balanced_accuracy_score(y_val, val_predictions)
                    
                    # if i % 10 == 0 or val_balanced_score > best_score:  # Progress updates
                    print('---------Validation---------')
                    print(f"Params {i+1}/{len(param_grid)}: {params}")
                    print(f"Prediction Class distribution (val): {np.bincount(val_predictions)}")
                    print(f"Actual Class distribution (val): {np.bincount(y_val)}")
                    print(f"Validation score: {val_score:.4f}")
                    print(f"Validation balanced accuracy: {val_balanced_score:.4f}")
                
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params
                        print(f"  ★ New best validation score: {best_score:.4f}")
                
                except Exception as e:
                    print(f"Error with params {params}: {e}")
                    continue
            
            print(f"✓ Best {model_type} hyperparameters: {best_params}")
            print(f"✓ Best validation accuracy: {best_score:.4f}")
            
            # Save hyperparameters with model type prefix
            try:
                os.makedirs(self.results_dir, exist_ok=True)
                hyperparams_file_with_type = f"{self.results_dir}/{model_type}_gating_network_hyperparams.json"
                with open(hyperparams_file_with_type, 'w') as f:
                    json.dump(best_params, f, indent=2)
                print(f"✓ {model_type} gating network hyperparameters saved")
            except Exception as e:
                print(f"⚠ Could not save hyperparameters: {e}")
        
        # Train classifier for validation accuracy assessment
        if 'best_epoch' in best_params:
            best_params.pop('best_epoch')
        if model_type == 'logistic':
            temp_classifier = LogisticRegression(random_state=42, **best_params)
        elif model_type == 'mlp':
            temp_classifier = GatingMLPWrapper(random_state=42, use_class_weights=False, **best_params)
        
        temp_classifier.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_predictions = temp_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        print(f"✓ Validation accuracy: {val_accuracy:.4f}")

        # Train final classifier on combined train+validation data for best performance
        # print(f"Training final {model_type} gating network on combined train+validation data...")
        # X_combined = np.vstack([X_train, X_val])
        # y_combined = np.concatenate([y_train, y_val])
        
        # if model_type == 'logistic':
        #     final_classifier = LogisticRegression(random_state=42, **best_params)
        # elif model_type == 'mlp':
        #     final_classifier = GatingMLPWrapper(random_state=42, **best_params)
        
        # final_classifier.fit(X_combined, y_combined)
        
        # print(f"✓ Final gating network trained on {len(X_combined)} races")
        
        return temp_classifier, best_params, val_accuracy, scaler

    def apply_temperature_scaling(self, logits, temperature):
        """Apply temperature scaling to logits before softmax"""
        return logits / temperature

    def normalize_expert_scores(self, aligned_data, method='z_score'):
        """
        Normalize expert scores per race to ensure fair combination across different schemes
        
        Args:
            aligned_data (dict): Race-aligned predictions from different schemes
            method (str): Normalization method ('z_score' or 'min_max')
            normalization_params (dict): Not used in per-race normalization (kept for compatibility)
            
        Returns:
            tuple: (normalized_data, None) - normalization_params is None since we normalize per race
        """
        print(f"Normalizing expert scores using per-race {method} method...")
        
        normalized_data = {}
        
        # Apply per-race normalization
        for race_key, race_data in aligned_data.items():
            normalized_race_data = {}
            
            for scheme in self.schemes:
                if scheme in race_data:
                    scores = np.array(race_data[scheme])
                    
                    # Skip normalization if we have fewer than 2 scores or all scores are identical
                    if len(scores) < 2 or np.std(scores) < 1e-8:
                        normalized_race_data[scheme] = scores
                        continue
                    
                    # Compute per-race normalization parameters
                    if method == 'z_score':
                        mean_score = np.mean(scores)
                        std_score = np.std(scores) + 1e-8  # Add small epsilon to avoid division by zero
                        normalized_scores = (scores - mean_score) / std_score
                    elif method == 'min_max':
                        min_score = np.min(scores)
                        max_score = np.max(scores)
                        score_range = max_score - min_score + 1e-8  # Add small epsilon to avoid division by zero
                        normalized_scores = (scores - min_score) / score_range
                    else:
                        # Default to no normalization for unknown methods
                        normalized_scores = scores
                    
                    normalized_race_data[scheme] = normalized_scores
            
            normalized_data[race_key] = normalized_race_data
        
        print(f"✓ Expert scores normalized per-race across {len(aligned_data)} races")
        return normalized_data

    def optimize_temperature(self, classifier, X_val, y_val):
        """
        Optimize temperature parameter for better calibration
        
        Args:
            classifier: Trained classifier
            X_val: Validation features (no scaling needed for one-hot encoded features)
            y_val: Validation labels
            
        Returns:
            float: Optimal temperature
        """
        print(f"Optimizing temperature scaling...")
        
        # Get raw logits (decision function)
        logits = classifier.decision_function(X_val)
        if logits.ndim == 1:  # Binary case, make it 2D
            logits = np.column_stack([-logits, logits])
        
        best_temp = 1.0
        best_loss = float('inf')
        
        # Grid search over temperature values
        temp_range = np.concatenate([ np.arange(0.05,1.0,0.05), np.arange(1.0,5.5,0.5)])
        
        for temp in temp_range:
            # Apply temperature scaling
            scaled_logits = self.apply_temperature_scaling(logits, temp)
            
            # Convert to probabilities
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Calculate cross-entropy loss
            loss = log_loss(y_val, probabilities)
            
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        
        print(f"✓ Optimal temperature: {best_temp:.3f} (validation loss: {best_loss:.4f})")
        return best_temp

    def run_hard_moe_gating_ensemble(self, test_year=2023, gating_model_type='logistic'):
        """
        Run hard MoE gating ensemble - selects single expert with highest weight
        
        Args:
            test_year (int): Year to evaluate on
            gating_model_type (str): Type of gating model ('logistic' or 'mlp')
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print(f"\n{'='*60}")
        print("HARD MOE GATING ENSEMBLE")
        print(f"{'='*60}")
        
        # Get base model predictions
        all_predictions = self.get_base_predictions_for_year(test_year) 
        base_predictions_train = {}
        base_predictions_val = {}
        base_predictions_test = {}
        
        for scheme, data in all_predictions.items():
            base_predictions_train[scheme] = data[data['year'] < test_year - 1]
            base_predictions_val[scheme] = data[data['year'] == test_year - 1]
            base_predictions_test[scheme] = data[data['year'] == test_year]

        if not all([base_predictions_train, base_predictions_val, base_predictions_test]):
            raise ValueError("Missing base model predictions for MoE gating")
        
        # Get race data for training gating network
        race_data = load_and_merge_features(self.race_class, 'leader')
        race_data = race_data[ID_COLS+RACE_FEATURES_ALL]
        # fill na with mean
        race_data[['distance', 'verticalMeters', 'profileScore']] = race_data[['distance', 'verticalMeters', 'profileScore']].fillna(race_data[['distance', 'verticalMeters', 'profileScore']].mean())
        train_data = race_data[race_data['year'] < test_year - 1].copy()
        val_data = race_data[race_data['year'] == test_year - 1].copy()
        test_data = race_data[race_data['year'] == test_year].copy()
        
        # Train gating network (now returns optimal temperature computed correctly)
        classifier, best_params, val_accuracy, scaler = self.train_gating_network(
            train_data, val_data, base_predictions_train, base_predictions_val, model_type=gating_model_type
        )
        
        # Apply hard gating on test set
        print(f"\nApplying hard MoE gating on test set...")
        
        # First, compute normalization parameters from train+val data to avoid data leakage
        print(f"Computing normalization parameters from train+validation data...")
        train_val_predictions = {}
        for scheme in self.schemes:
            if scheme in base_predictions_train and scheme in base_predictions_val:
                train_val_predictions[scheme] = pd.concat([
                    base_predictions_train[scheme], 
                    base_predictions_val[scheme]
                ], ignore_index=True)
        
        train_val_aligned = self.align_predictions(train_val_predictions)
        # Per-race normalization doesn't need to store parameters from train/val data
        
        # Apply per-race normalization to test data
        aligned_data = self.align_predictions(base_predictions_test)
        normalized_data = self.normalize_expert_scores(aligned_data, method='z_score')
        ensemble_results = []
        gating_decisions = []
        
        record_id = get_record_id(self.level)
        rank_col = get_rank_col(self.level)
        
        for race_key, race_data in tqdm(normalized_data.items()):
            # Get race context for gating
            race_context = test_data[
                (test_data['race'] == race_key[0]) & 
                (test_data['date'] == race_key[1])
            ]
            
            if len(race_context) == 0:
                continue
            
            # Extract gating features
            gating_features = self.extract_gating_features(race_context)
            gating_features = gating_features.reshape(1, -1)
            gating_features = scaler.transform(gating_features)  # Apply same scaling as training
            
            # Get expert probabilities with temperature scaling
            logits = classifier.decision_function(gating_features)
            if logits.ndim == 1:
                logits = np.column_stack([-logits, logits])
            

            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            expert_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            expert_probs = expert_probs.flatten()
            
            # Hard gating: select expert with highest probability
            selected_expert_idx = np.argmax(expert_probs)
            selected_scheme = self.schemes[selected_expert_idx]
            
            # Use selected expert's predictions
            if selected_scheme in race_data:
                ensemble_scores = race_data[selected_scheme]
                
                # Create result dataframe
                race_df = list(base_predictions_test.values())[0]
                race_subset = race_df[
                    (race_df['race'] == race_key[0]) & 
                    (race_df['date'] == race_key[1])
                ].copy()
                race_subset = race_subset.sort_values(record_id).reset_index(drop=True)
                
                race_subset['ensemble_score'] = ensemble_scores
                race_subset['selected_expert'] = selected_scheme
                race_subset['expert_confidence'] = expert_probs[selected_expert_idx]
                
                ensemble_results.append(race_subset[[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score', 'selected_expert', 'expert_confidence']])
                
                gating_decisions.append({
                    'race': race_key[0],
                    'date': race_key[1], 
                    'selected_expert': selected_scheme,
                    'confidence': expert_probs[selected_expert_idx],
                    'expert_probs': dict(zip(self.schemes, expert_probs))
                })
        
        ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        
        # Expert scores already normalized globally, no need for per-race normalization
        
        # Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Hard MoE gating ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {ensemble_eval[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {ensemble_eval[f'Recall@{self.k_value}'].mean():.4f}")
        print(f"   Gating network accuracy: {val_accuracy:.4f}")
        
        # Expert usage statistics
        expert_usage = ensemble_df['selected_expert'].value_counts()
        print(f"\n📊 Expert Usage:")
        for expert, count in expert_usage.items():
            print(f"   {expert}: {count} races ({count/len(ensemble_df)*100:.1f}%)")
        
        # Store results
        self.ensemble_results['hard_moe_gating'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval),
            'gating_classifier': classifier,
            'gating_hyperparams': best_params,
            'gating_accuracy': val_accuracy,
            'expert_usage': expert_usage.to_dict(),
            'gating_decisions': gating_decisions,
        }
        
        return ensemble_eval

    def run_soft_moe_gating_ensemble(self, test_year=2023, gating_model_type='logistic'):
        """
        Run soft MoE gating ensemble - weighted combination based on expert probabilities
        
        Args:
            test_year (int): Year to evaluate on
            enable_post_hoc_tuning (bool): Enable post-hoc fine-tuning of gating weights
            gating_model_type (str): Type of gating model ('logistic' or 'mlp')
            
        Returns:
            pd.DataFrame: Ensemble results
        """
        print(f"\n{'='*60}")
        print("SOFT MOE GATING ENSEMBLE")
        print(f"{'='*60}")
        
        # Get base model predictions
        all_predictions = self.get_base_predictions_for_year(test_year) 
        base_predictions_train = {}
        base_predictions_val = {}
        base_predictions_test = {}
        for scheme, data in all_predictions.items():
            base_predictions_train[scheme] = data[data['year'] < test_year - 1]
            base_predictions_val[scheme] = data[data['year'] == test_year - 1]
            base_predictions_test[scheme] = data[data['year'] == test_year]

        if not all([base_predictions_train, base_predictions_val, base_predictions_test]):
            raise ValueError("Missing base model predictions for MoE gating")
        
        # Get race data for training gating network
        race_data = load_and_merge_features(self.race_class, 'leader')
        race_data = race_data[ID_COLS+RACE_FEATURES_ALL]
        # fill na with mean
        race_data[['distance', 'verticalMeters', 'profileScore']] = race_data[['distance', 'verticalMeters', 'profileScore']].fillna(race_data[['distance', 'verticalMeters', 'profileScore']].mean())
        train_data = race_data[race_data['year'] < test_year - 1].copy()
        val_data = race_data[race_data['year'] == test_year - 1].copy()
        test_data = race_data[race_data['year'] == test_year].copy()
        
        # First, compute normalization parameters from train+val data to avoid data leakage
        print(f"Computing normalization parameters from train+validation data...")
        train_val_predictions = {}
        for scheme in self.schemes:
            if scheme in base_predictions_train and scheme in base_predictions_val:
                train_val_predictions[scheme] = pd.concat([
                    base_predictions_train[scheme], 
                    base_predictions_val[scheme]
                ], ignore_index=True)
        
        train_val_data = pd.concat([train_data, val_data], ignore_index=True)
        train_val_aligned = self.align_predictions(train_val_predictions)
        test_aligned = self.align_predictions(base_predictions_test)

        # Train gating network (now returns optimal temperature computed correctly)
        classifier, best_params, val_accuracy, scaler = self.train_gating_network(
            train_val_data, test_data, train_val_aligned, test_aligned, model_type=gating_model_type
        )

        # Apply soft gating on test set
        print(f"\nApplying soft MoE gating on test set...")
                
        # Apply per-race normalization to test data
        normalized_data = self.normalize_expert_scores(test_aligned, method='z_score')
        ensemble_results = []
        gating_decisions = []
        
        record_id = get_record_id(self.level)
        rank_col = get_rank_col(self.level)
        
        for race_key, race_data in tqdm(normalized_data.items()):
            # Get race context for gating
            race_context = test_data[
                (test_data['race'] == race_key[0]) & 
                (test_data['date'] == race_key[1])
            ]
            
            if len(race_context) == 0:
                continue
            
            # Extract gating features
            gating_features = self.extract_gating_features(race_context)
            gating_features = gating_features.reshape(1, -1)
            gating_features = scaler.transform(gating_features)  # Apply same scaling as training
            
            # Get expert probabilities with temperature scaling
            logits = classifier.decision_function(gating_features)
            if logits.ndim == 1:
                logits = np.column_stack([-logits, logits])
            
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            expert_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            expert_probs = expert_probs.flatten()

            # Soft gating: weighted combination of globally normalized scores
            ensemble_scores = np.zeros(len(race_data[self.schemes[0]]))
            total_weight = 0
            
            for i, scheme in enumerate(self.schemes):
                if scheme in race_data:
                    weight = expert_probs[i] if i < len(expert_probs) else 0
                    ensemble_scores += weight * race_data[scheme]  # race_data already contains normalized scores
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_scores = ensemble_scores / total_weight
            
            # Create result dataframe
            race_df = list(base_predictions_test.values())[0]
            race_subset = race_df[
                (race_df['race'] == race_key[0]) & 
                (race_df['date'] == race_key[1])
            ].copy()
            race_subset = race_subset.sort_values(record_id).reset_index(drop=True)
            
            race_subset['ensemble_score'] = ensemble_scores
            
            # Add expert weights as additional columns
            for i, scheme in enumerate(self.schemes):
                weight = expert_probs[i] if i < len(expert_probs) else 0
                race_subset[f'{scheme}_weight'] = weight
            
            ensemble_results.append(race_subset[[rank_col, record_id, 'race', 'date', 'cluster', 'race_class', 'ensemble_score'] + [f'{s}_weight' for s in self.schemes]])
            
            gating_decisions.append({
                'race': race_key[0],
                'date': race_key[1],
                'expert_weights': dict(zip(self.schemes, expert_probs)),
                'entropy': -np.sum(expert_probs * np.log(expert_probs + 1e-8))  # Measure of uncertainty
            })
        
        ensemble_df = pd.concat(ensemble_results, ignore_index=True)
        
        # Evaluate ensemble predictions
        ensemble_eval = self.evaluate_ensemble_predictions(ensemble_df)
        
        print(f"✅ Soft MoE gating ensemble completed!")
        print(f"   Mean NDCG@{self.k_value}: {ensemble_eval[f'NDCG@{self.k_value}'].mean():.4f}")
        print(f"   Mean Recall@{self.k_value}: {ensemble_eval[f'Recall@{self.k_value}'].mean():.4f}")
        print(f"   Gating network accuracy: {val_accuracy:.4f}")
        
        # Weight usage statistics
        weight_stats = {}
        for scheme in self.schemes:
            weight_col = f'{scheme}_weight'
            if weight_col in ensemble_df.columns:
                weight_stats[scheme] = {
                    'mean': ensemble_df[weight_col].mean(),
                    'std': ensemble_df[weight_col].std(),
                    'min': ensemble_df[weight_col].min(),
                    'max': ensemble_df[weight_col].max()
                }
        
        print(f"\n📊 Expert Weight Statistics:")
        for scheme, stats in weight_stats.items():
            print(f"   {scheme}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # Store results
        self.ensemble_results['soft_moe_gating'] = {
            'predictions': ensemble_df,
            'evaluation': ensemble_eval,
            'summary_stats': calculate_summary_statistics(ensemble_eval),
            'gating_classifier': classifier,
            'gating_hyperparams': best_params,
            'gating_accuracy': val_accuracy,
            'weight_statistics': weight_stats,
            'gating_decisions': gating_decisions,
        }
        
        return ensemble_eval

    def post_hoc_finetune_gating(self, classifier, temperature, scaler, base_predictions_val, val_data, num_epochs=50):
        """
        Post-hoc fine-tuning of gating weights for ranking objective
        
        Args:
            classifier: Trained gating classifier
            temperature: Optimal temperature
            scaler: Feature scaler used during training
            base_predictions_val: Validation predictions
            val_data: Validation data
            num_epochs: Number of fine-tuning epochs
            
        Returns:
            np.array: Fine-tuned expert weights
        """
        print(f"Starting post-hoc fine-tuning for {num_epochs} epochs...")
        
        # Initialize weights to uniform
        expert_weights = np.ones(len(self.schemes)) / len(self.schemes)
        learning_rate = 0.01
        best_weights = expert_weights.copy()
        best_ndcg = -1
        
        aligned_val_data = self.align_predictions(base_predictions_val)
        # Use same normalization approach as in main method
        normalized_val_data = self.normalize_expert_scores(aligned_val_data, method='z_score')
        
        for epoch in range(num_epochs):
            epoch_ndcg_scores = []
            gradients = np.zeros(len(self.schemes))
            
            for race_key, race_data in normalized_val_data.items():
                # Get race context
                race_context = val_data[
                    (val_data['race'] == race_key[0]) & 
                    (val_data['date'] == race_key[1])
                ]
                
                if len(race_context) == 0:
                    continue
                
                # Get gating probabilities
                gating_features = self.extract_gating_features(race_context)
                gating_features = gating_features.reshape(1, -1)
                gating_features = scaler.transform(gating_features)  # Apply same scaling as training
                
                logits = classifier.decision_function(gating_features)
                if logits.ndim == 1:
                    logits = np.column_stack([-logits, logits])
                
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                base_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                base_probs = base_probs.flatten()
                
                # Apply current expert weights
                weighted_probs = base_probs * expert_weights
                weighted_probs = weighted_probs / np.sum(weighted_probs)
                
                # Compute ensemble prediction using globally normalized scores
                ensemble_scores = np.zeros(len(race_data[self.schemes[0]]))
                for i, scheme in enumerate(self.schemes):
                    if scheme in race_data:
                        ensemble_scores += weighted_probs[i] * race_data[scheme]  # race_data already normalized
                
                # Get actual ranking for NDCG calculation
                rank_col = get_rank_col(self.level)
                record_id = get_record_id(self.level)
                
                race_actual = val_data[
                    (val_data['race'] == race_key[0]) & 
                    (val_data['date'] == race_key[1])
                ]
                
                if len(race_actual) == 0:
                    continue
                
                actual_order = race_actual.sort_values(rank_col)[record_id].tolist()
                
                # Create dummy team IDs and get predicted order
                team_ids = [f"team_{i}" for i in range(len(ensemble_scores))]
                pred_order = [team_ids[i] for i in np.argsort(ensemble_scores)[::-1]]
                
                # Calculate NDCG
                from roster_ranker.core import ndcg_at_k
                ndcg = ndcg_at_k(pred_order, actual_order, self.k_value)
                epoch_ndcg_scores.append(ndcg)
                
                # Compute approximate gradients (simple finite differences)
                eps = 1e-4
                for i in range(len(self.schemes)):
                    # Perturb weight
                    perturbed_weights = expert_weights.copy()
                    perturbed_weights[i] += eps
                    perturbed_weights = perturbed_weights / np.sum(perturbed_weights)
                    
                    # Compute perturbed prediction
                    perturbed_probs = base_probs * perturbed_weights
                    perturbed_probs = perturbed_probs / np.sum(perturbed_probs)
                    
                    perturbed_scores = np.zeros(len(race_data[self.schemes[0]]))
                    for j, scheme in enumerate(self.schemes):
                        if scheme in race_data:
                            perturbed_scores += perturbed_probs[j] * race_data[scheme]  # race_data already normalized
                    
                    # Get perturbed NDCG
                    perturbed_pred_order = [team_ids[k] for k in np.argsort(perturbed_scores)[::-1]]
                    perturbed_ndcg = ndcg_at_k(perturbed_pred_order, actual_order, self.k_value)
                    
                    # Gradient approximation
                    gradient = (perturbed_ndcg - ndcg) / eps
                    gradients[i] += gradient
            
            # Update weights
            if len(epoch_ndcg_scores) > 0:
                mean_epoch_ndcg = np.mean(epoch_ndcg_scores)
                
                # Apply gradients
                expert_weights += learning_rate * gradients / len(normalized_val_data)
                expert_weights = np.maximum(expert_weights, 1e-8)  # Keep positive
                expert_weights = expert_weights / np.sum(expert_weights)  # Normalize
                
                # Track best weights
                if mean_epoch_ndcg > best_ndcg:
                    best_ndcg = mean_epoch_ndcg
                    best_weights = expert_weights.copy()
                
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch}: NDCG={mean_epoch_ndcg:.4f}, Weights={expert_weights}")
        
        print(f"✓ Post-hoc fine-tuning completed")
        print(f"✓ Best validation NDCG: {best_ndcg:.4f}")
        print(f"✓ Final expert weights: {best_weights}")
        
        return best_weights


def run_ensemble_experiment(race_class, ensemble_methods=None, force_retune=False, gating_model_type='logistic'):
    """
    Run complete ensemble experiment pipeline using EnsembleExperiment class
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        ensemble_methods (list): Ensemble methods to run
        force_retune (bool): Whether to force hyperparameter retuning
        gating_model_type (str): Type of gating model for MoE methods ('logistic' or 'mlp')
    
    Returns:
        EnsembleExperiment: The experiment instance with results
    """
    # Create and run ensemble experiment
    ensemble_exp = EnsembleExperiment(
        race_class=race_class,
        ensemble_methods=ensemble_methods,
        gating_model_type=gating_model_type
    )
    
    # Run all ensemble methods
    results = ensemble_exp.run_all_ensemble_methods(force_retune_base=force_retune)
    
    return ensemble_exp