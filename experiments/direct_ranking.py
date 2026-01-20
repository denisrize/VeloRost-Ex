import os
import json
import time
import pandas as pd
from ..utils.config import *
from ..core.metrics import ndcg_at_k, recall_at_k
from ..data.loaders import map_class
import numpy as np
from tqdm import tqdm


class DirectRankingExperiment:

    def __init__(
        self,
        race_class=None,
        schemes=None,
        k_grid=None,
        k_value=10,
        lambdas=None,
        baseline=False,
        specific_race=None,
        top_n=None,
        lambda_grid=None,
        max_teammates=8
    ):
        self.race_class = race_class
        self.schemes = schemes if schemes is not None else ['equal_weight']
        self.k_grid = k_grid if k_grid is not None else DIRECT_RANKING_K_GRID
        self.k_value = k_value
        self.lambdas = lambdas if lambdas is not None else DIRECT_RANKING_LAMBDA_GRID
        self.baseline = baseline
        self.specific_race = specific_race
        self.top_n = top_n
        self.lambda_grid = lambda_grid if lambda_grid is not None else DIRECT_RANKING_LAMBDA_GRID
        self.max_teammates = max_teammates

    def _calculate_ensemble_teammate_scores(self, grp, k_pen, schemes, top_n=None):
        """
        Calculate teammate scores averaged across all schemes.
        
        Args:
            grp: Race group data
            k_pen: K penalty parameter
            schemes: List of schemes to average across
            top_n: Number of top teammates to include (default: None)
        Returns:
            pd.Series: Average teammate scores for each rider
        """
        teammate_avg_perf = []
        for idx, row in grp.iterrows():
            teammates = grp[(grp['team'] == row['team']) & (grp['rider'] != row['rider'])]
            if len(teammates) > 0:
                # Calculate teammate scores for each scheme
                scheme_scores = []
                
                # First scheme (base columns)
                teammate_perfs_base = teammates['race_cluster_teammate_mu'] - k_pen * teammates['race_cluster_teammate_sigma']
                if top_n is not None:
                    teammate_perfs_base = teammate_perfs_base.nlargest(top_n)
                scheme_scores.append(teammate_perfs_base.mean())
                
                # Additional schemes (suffixed columns)
                for s in schemes[1:]:
                    mu_col = f'race_cluster_teammate_mu_{s}'
                    sigma_col = f'race_cluster_teammate_sigma_{s}'
                    teammate_perfs_scheme = teammates[mu_col] - k_pen * teammates[sigma_col]
                    if top_n is not None:
                        teammate_perfs_scheme = teammate_perfs_scheme.nlargest(top_n)
                    scheme_scores.append(teammate_perfs_scheme.mean())
                
                # Average across all schemes
                avg_teammate_perf = np.mean(scheme_scores)
            else:
                avg_teammate_perf = 0.0
            teammate_avg_perf.append(avg_teammate_perf)
        return pd.Series(teammate_avg_perf, index=grp.index)

    def run_direct_ranking_evaluation(
            self,
            race_class,
            year,
            k_grid      = None,          
            k_value   = None,
            schemes = None,  
            lambdas=None, 
            baseline = None,  
            specific_race=None,
            top_n=None
    ):
        """
        Enhanced direct ranking evaluation with lambda parameter for teammate contribution.
        Always averages teammate scores across all provided schemes.
        
        Performance formula: 
        - Baseline mode: rider_score = (mu - k_penalty * sigma)
        - Full mode: rider_score = (mu - k_penalty * sigma) + lambda * avg_teammate_scores
        
        Args:
            race_class (str): Race class to evaluate ('all', 'WT', etc.)
            year (int): Target year for testing (year-1 used for validation)
            k_grid (range): Range of k_penalty values to test (default: 1-29)
            k_value (int): K value for NDCG@k and Top-k accuracy evaluation (default: 10)
            schemes (list): List of schemes to average teammate scores across (default: ['rank_norm'])
            lambda_grid (list): Lambda values to test for teammate contribution (default: 0.0-1.0, step 0.05)
            baseline (bool): If True, only use leader skill without teammates (default: False)
            specific_race (str): Race name to evaluate (default: None)
            top_n (int): Number of top teammates to include (default: None)
        Returns:
            str: Output directory path containing results
        
        The function performs hyperparameter optimization:
        - Baseline mode: Only k_penalty tuning using validation set
        - Full mode: Nested k_penalty and lambda tuning using validation set
        
        Teammate scores are always averaged across all schemes in the list (ignored in baseline mode).
        """
        
        race_class = race_class if race_class is not None else self.race_class
        k_grid = k_grid if k_grid is not None else self.k_grid
        k_value = k_value if k_value is not None else self.k_value
        schemes = schemes if schemes is not None else self.schemes
        lambdas = lambdas if lambdas is not None else self.lambdas
        baseline = baseline if baseline is not None else self.baseline
        specific_race = specific_race if specific_race is not None else self.specific_race
        top_n = top_n if top_n is not None else self.top_n

        scheme_names = '_'.join(schemes)
        
        # Load race results data
        race_results_path = get_data_dir('riders_race_results')
        race_results = pd.read_csv(race_results_path)
        race_results['date'] = pd.to_datetime(race_results['date'])
        race_results['classification'] = race_results['classification'].astype(str)
        
        merge_on = ['race', 'date', 'rider', 'cluster','classification','rank_number']

        output_path = get_output_dir('direct_ranking', race_class=race_class)
        if baseline:
            output_path = os.path.join(output_path, 'baseline')
        elif len(schemes) > 1:
            output_path = output_path.replace('direct_ranking', f'direct_ranking_ensemble')
        else:
            output_path = os.path.join(output_path, schemes[0])

        os.makedirs(output_path, exist_ok=True)
        print(f'Output path: {output_path}')
        
        leader_power_path = get_leader_power_path(race_class)
        true_skill_features_leader = pd.read_csv(leader_power_path)
        true_skill_features_leader['date'] = pd.to_datetime(true_skill_features_leader['date'])
        true_skill_features_leader['year'] = true_skill_features_leader['date'].dt.year
        true_skill_features_leader['classification'] = true_skill_features_leader['classification'].astype(str)

        true_skill_features_leader = pd.merge(
            race_results[merge_on], 
            true_skill_features_leader, 
            on=merge_on, 
            how='inner'
        )
        merge_on.append('team')
        
        if baseline:
            # Baseline mode: only use leader features
            print("Baseline mode: using only leader features")
            true_skill_features = true_skill_features_leader.copy()
        else:
            # Load teammates data for all schemes
            scheme_data = {}
            for s in schemes:
                team_power_path = get_team_power_path(race_class, s)
                scheme_df = pd.read_csv(team_power_path)
                scheme_df['date'] = pd.to_datetime(scheme_df['date'])
                scheme_df['classification'] = scheme_df['classification'].astype(str)
                scheme_data[s] = scheme_df
            print(f"Loaded teammate data for {len(schemes)} schemes: {schemes}")
            
            print("Computing team performance features...")

            # Merge leader features with each scheme data separately
            merged_schemes = {}
            for s, scheme_df in scheme_data.items():
                print(f'Merging features of leader {true_skill_features_leader.shape} and teammates for scheme {s} {scheme_df.shape}')
                merged_df = pd.merge(
                    scheme_df[merge_on + RAW_TRUE_SKILL_TEAMMATE_FEATURES],     
                    true_skill_features_leader[merge_on+RAW_TRUE_SKILL_LEADER_FEATURES], 
                    on=merge_on, how='inner'
                )
                merged_schemes[s] = merged_df
                print(f'Merged features shape for scheme {s}: {merged_df.shape}')
            
            # Use the first scheme as base and add teammate columns from other schemes
            base_scheme = list(merged_schemes.keys())[0]
            true_skill_features = merged_schemes[base_scheme].copy()
            
            # Add teammate features from other schemes with suffix
            for s in list(merged_schemes.keys())[1:]:
                scheme_df = merged_schemes[s]
                # Add teammate features with scheme suffix
                for col in ['race_cluster_teammate_mu', 'race_cluster_teammate_sigma']:
                    if col in scheme_df.columns:
                        true_skill_features[f'{col}_{s}'] = scheme_df[col]
        
        print(f'Final features shape: {true_skill_features.shape}')

        # split into validation and test sets
        true_skill_features['year'] = true_skill_features['date'].dt.year
        # true_skill_features = true_skill_features[true_skill_features['cluster'] != 'General Classification'].copy()
        if specific_race is not None:
            true_skill_features = true_skill_features[true_skill_features['race'].str.contains(specific_race)]
        val_df = true_skill_features[true_skill_features['year'] == year - 1]
        test_df = true_skill_features[true_skill_features['year'] == year]

        print(f'Validation set shape: {val_df.shape}')
        print(f'Test set shape: {test_df.shape}')
        val_results, test_results = [], []

        # ------- hyper‑parameter grid‑search on the validation set -------------
        for k_pen in tqdm(k_grid, desc='k_penalty tuning', total=len(k_grid)):
            if baseline:
                # Baseline mode: no lambda tuning, only k_penalty
                best_lambda = 0.0
                val_scores = []
                
                # Calculate performance for each race in validation set
                for (race_name, race_date), grp in val_df.groupby(['race','date']):
                    # Calculate leader performance only
                    leader_perf = grp['race_cluster_leader_mu'] - k_pen * grp['race_cluster_leader_sigma']
                    combined_perf = leader_perf  # No teammate component
                    
                    # Sort by performance to get predictions
                    grp_with_perf = grp.copy()
                    grp_with_perf['combined_perf'] = combined_perf
                    pred = grp_with_perf.sort_values('combined_perf', ascending=False)['rider']
                    true = grp.sort_values('rank_number')['rider']
                    
                    # Calculate NDCG for this race
                    race_ndcg = ndcg_at_k(list(pred), list(true), k_value)
                    val_scores.append(race_ndcg)
                
                best_val_score = np.mean(val_scores)
            else:
                # Full mode: nested lambda tuning for this k_penalty
                best_lambda = 0.0
                best_val_score = -np.inf
                
                for lam in tqdm(lambdas, desc='lambda tuning', total=len(lambdas)):
                    val_scores = []
                    
                    # Calculate performance for each race in validation set
                    for (race_name, race_date), grp in val_df.groupby(['race','date']):
                        # Calculate leader performance
                        leader_perf = grp['race_cluster_leader_mu'] - k_pen * grp['race_cluster_leader_sigma']
                        
                        # Calculate average teammate performance across schemes
                        teammate_avg_perf = self._calculate_ensemble_teammate_scores(grp, k_pen, schemes, top_n=top_n)
                        
                        # Combined performance: leader + lambda * avg_teammates
                        combined_perf = leader_perf + lam * teammate_avg_perf
                        
                        # Sort by performance to get predictions
                        grp_with_perf = grp.copy()
                        grp_with_perf['combined_perf'] = combined_perf
                        pred = grp_with_perf.sort_values('combined_perf', ascending=False)['rider']
                        true = grp.sort_values('rank_number')['rider']
                        
                        # Calculate NDCG for this race
                        race_ndcg = ndcg_at_k(list(pred), list(true), k_value)
                        val_scores.append(race_ndcg)
                    
                    # Average NDCG across all validation races for this k_penalty, lambda combination
                    mean_val_score = np.mean(val_scores)
                    
                    # Track best lambda for this k_penalty
                    if mean_val_score > best_val_score:
                        best_val_score = mean_val_score
                        best_lambda = lam
                        print(f'New best lambda: {best_lambda} for k_penalty: {k_pen} with mean_val_score: {mean_val_score}')
            
            # Now evaluate on validation set with best lambda for detailed results
            for (race_name, race_date), grp in val_df.groupby(['race','date']):
                # Calculate leader performance
                leader_perf = grp['race_cluster_leader_mu'] - k_pen * grp['race_cluster_leader_sigma']
                
                if baseline:
                    # Baseline mode: only leader performance
                    combined_perf = leader_perf
                else:
                    # Full mode: leader + teammate performance
                    teammate_avg_perf = self._calculate_ensemble_teammate_scores(grp, k_pen, schemes, top_n=top_n)
                    combined_perf = leader_perf + best_lambda * teammate_avg_perf
                
                grp_with_perf = grp.copy()
                grp_with_perf['combined_perf'] = combined_perf
                pred = grp_with_perf.sort_values('combined_perf', ascending=False)['rider']
                true = grp.sort_values('rank_number')['rider']

                res_row = {
                    'race'    : race_name,
                    'date':    race_date,
                    'classification': grp['classification'].iloc[0],
                    'cluster': grp['cluster'].iloc[0],
                    'k_penalty'  : k_pen,
                    'lambda': best_lambda if not baseline else 0.0,
                    'scheme': 'baseline' if baseline else scheme_names,
                    'VAL_NDCG@10'    : round(ndcg_at_k(list(pred), list(true), k_value),4),
                    'VAL_Top-10 Accuracy' : round(recall_at_k(list(pred), list(true), k_value), 4),
                }
                val_results.append(res_row)

            # ------------- evaluate on TEST with best k_penalty and lambda* ----------------------------
            for (race_name, race_date), grp in test_df.groupby(['race','date']):
                # Calculate leader performance
                leader_perf = grp['race_cluster_leader_mu'] - k_pen * grp['race_cluster_leader_sigma']
                
                if baseline:
                    # Baseline mode: only leader performance
                    combined_perf = leader_perf
                else:
                    # Full mode: leader + teammate performance
                    teammate_avg_perf = self._calculate_ensemble_teammate_scores(grp, k_pen, schemes, top_n=top_n)
                    combined_perf = leader_perf + best_lambda * teammate_avg_perf
                
                grp_with_perf = grp.copy()
                grp_with_perf['combined_perf'] = combined_perf
                pred = grp_with_perf.sort_values('combined_perf', ascending=False)['rider']
                true = grp.sort_values('rank_number')['rider']
                ndcg = round(ndcg_at_k(list(pred), list(true), k_value),4)
                top_10_accuracy = round(recall_at_k(list(pred), list(true), k_value), 4)
                res_row = {
                    'race': race_name,
                    'date':    race_date,
                    'classification': grp['classification'].iloc[0],
                    'cluster': grp['cluster'].iloc[0],
                    'k_penalty'  : k_pen,
                    'lambda': best_lambda if not baseline else 0.0,
                    'scheme': 'baseline' if baseline else scheme_names,
                    'TEST_NDCG@10': ndcg,
                    'TEST_Top-10 Accuracy' : top_10_accuracy,
                }
                test_results.append(res_row)

        val_results_df = pd.DataFrame(val_results)
        test_results_df = pd.DataFrame(test_results)

        if specific_race is not None:
            output_path = os.path.join(output_path, specific_race)
            os.makedirs(output_path, exist_ok=True)
        if top_n is not None:
            output_path = os.path.join(output_path, f'top_{top_n}')
            os.makedirs(output_path, exist_ok=True)
        if k_value is not None:
            output_path = os.path.join(output_path, f'k_{k_value}')
            os.makedirs(output_path, exist_ok=True)

        val_results_df.to_csv(os.path.join(output_path, 'val_results.csv'), index=False)
        test_results_df.to_csv(os.path.join(output_path, 'test_results.csv'), index=False)

        # Add summary statistics
        val_summary_stats = val_results_df.groupby(['k_penalty', 'lambda']).agg({'VAL_NDCG@10': 'mean', 'VAL_Top-10 Accuracy': 'mean'}).reset_index()
        test_summary_stats = test_results_df.groupby(['k_penalty', 'lambda']).agg({'TEST_NDCG@10': 'mean', 'TEST_Top-10 Accuracy': 'mean'}).reset_index()
        val_class_summary_stats = val_results_df.groupby(['k_penalty', 'lambda', 'classification']).agg({'VAL_NDCG@10': 'mean', 'VAL_Top-10 Accuracy': 'mean'}).reset_index()
        test_class_summary_stats = test_results_df.groupby(['k_penalty', 'lambda', 'classification']).agg({'TEST_NDCG@10': 'mean', 'TEST_Top-10 Accuracy': 'mean'}).reset_index()
        val_cluster_summary_stats = val_results_df.groupby(['k_penalty', 'lambda', 'cluster']).agg({'VAL_NDCG@10': 'mean', 'VAL_Top-10 Accuracy': 'mean'}).reset_index()
        test_cluster_summary_stats = test_results_df.groupby(['k_penalty', 'lambda', 'cluster']).agg({'TEST_NDCG@10': 'mean', 'TEST_Top-10 Accuracy': 'mean'}).reset_index()
        
        # save results
        val_summary_stats.to_csv(os.path.join(output_path, 'val_summary_stats.csv'), index=False)
        test_summary_stats.to_csv(os.path.join(output_path, 'test_summary_stats.csv'), index=False)
        val_class_summary_stats.to_csv(os.path.join(output_path, 'val_class_summary_stats.csv'), index=False)
        test_class_summary_stats.to_csv(os.path.join(output_path, 'test_class_summary_stats.csv'), index=False)
        val_cluster_summary_stats.to_csv(os.path.join(output_path, 'val_cluster_summary_stats.csv'), index=False)
        test_cluster_summary_stats.to_csv(os.path.join(output_path, 'test_cluster_summary_stats.csv'), index=False)

        print(f"Direct ranking evaluation completed for {race_class} in {year} with k_value {k_value}")
        print(f'Top 5 (k_penalty, lambda) combinations for validation set:')
        print(val_summary_stats.sort_values('VAL_NDCG@10', ascending=False).head(5))
        print(f'Top 5 (k_penalty, lambda) combinations for test set:')
        print(test_summary_stats.sort_values('TEST_NDCG@10', ascending=False).head(5))
        print(f'Top 5 (k_penalty, lambda) combinations for validation set by classification:')
        print(val_class_summary_stats.sort_values('VAL_NDCG@10', ascending=False).head(5))
        print(f'Top 5 (k_penalty, lambda) combinations for test set by classification:')
        print(test_class_summary_stats.sort_values('TEST_NDCG@10', ascending=False).head(5))
        print(f'Top 5 (k_penalty, lambda) combinations for validation set by cluster:')
        print(val_cluster_summary_stats.sort_values('VAL_NDCG@10', ascending=False).head(5))
        print(f'Top 5 (k_penalty, lambda) combinations for test set by cluster:')
        print(test_cluster_summary_stats.sort_values('TEST_NDCG@10', ascending=False).head(5))

        return output_path

    def _score_race_group(self, grp, lam, k_pen, scheme, k_value, top_n):
        """Return (ndcg, topk) for one race group (race,date)."""
        leader_perf = grp['race_cluster_leader_mu'] - k_pen * grp['race_cluster_leader_sigma']
        teammate_avg_perf = self._calculate_ensemble_teammate_scores(grp, k_pen, [scheme], top_n=top_n)
        combined = leader_perf + lam * teammate_avg_perf

        pred = grp.assign(combined_perf=combined).sort_values('combined_perf', ascending=False)['rider']
        true = grp.sort_values('rank_number')['rider']
        return (
            ndcg_at_k(list(pred), list(true), k_value),
            recall_at_k(list(pred), list(true), k_value)
        )

    def _tune_lambda_on_subset(self, df_subset, k_pen, scheme, lambda_grid, k_value, top_n):
        """Return (best_lambda, val_ndcg, val_topk, n_val_races)."""
        if df_subset.empty:
            return 0.0, np.nan, np.nan, 0

        # group by race to get per-query metrics
        race_groups = list(df_subset.groupby(['race', 'date']))
        n_val = len(race_groups)

        best_lam, best_ndcg, best_topk = 0.0, -1.0, 0.0
        for lam in lambda_grid:
            ndcgs, tops = [], []
            for _, grp in race_groups:
                nd, tp = self._score_race_group(grp, lam, k_pen, scheme, k_value, top_n)
                ndcgs.append(nd); tops.append(tp)
            mean_ndcg = float(np.mean(ndcgs)) if ndcgs else np.nan
            mean_top  = float(np.mean(tops))  if tops  else np.nan

            if mean_ndcg > best_ndcg:
                best_ndcg, best_topk, best_lam = mean_ndcg, mean_top, lam
        return best_lam, best_ndcg, best_topk, n_val

    def _eval_on_subset(self, df_subset, lam, k_pen, scheme, k_value, top_n):
        """Return (test_ndcg, test_topk, n_test_races) using fixed lam."""
        if df_subset.empty:
            return np.nan, np.nan, 0
        race_groups = list(df_subset.groupby(['race', 'date']))
        ndcgs, tops = [], []
        for _, grp in race_groups:
            nd, tp = self._score_race_group(grp, lam, k_pen, scheme, k_value, top_n)
            ndcgs.append(nd); tops.append(tp)
        return float(np.mean(ndcgs)), float(np.mean(tops)), len(race_groups)

    def tune_lambda_by_context(
            self,
            race_class: str,
            year: int,
            scheme: str,
            k_grid      = None,          # 1 … 29
            lambda_grid = None,  # 0.00 … 1.00
            k_value: int = None,
            top_n=None
    ):
        """
        Context-aware λ tuning with fixed k_pen:
          1) Tune λ on the VALIDATION set (year-1) separately for each race_class and profile.
          2) Evaluate the selected λ on the TEST set (year).

        Saves:
          - best_lambda_per_class.csv  (cols: race_class, tuned_lambda, val/test metrics, counts, meta)
          - best_lambda_per_profile.csv
        """
        race_class = race_class if race_class is not None else self.race_class
        k_grid = k_grid if k_grid is not None else self.k_grid
        lambda_grid = lambda_grid if lambda_grid is not None else self.lambda_grid
        k_value = k_value if k_value is not None else self.k_value
        top_n = top_n if top_n is not None else self.top_n

        # ---------------- paths & IO ----------------
        output_path = get_output_dir('direct_ranking')
        output_path = os.path.join(output_path, scheme)
        os.makedirs(output_path, exist_ok=True)

        print(f'[λ-by-context] scheme={scheme}  year={year}')
        print(f'λ grid: {lambda_grid}')
        print(f'Output dir: {output_path}')

        # ---------------- load & merge ----------------
        race_results_path = get_data_dir('riders_race_results')
        race_results = pd.read_csv(race_results_path)
        race_results['date'] = pd.to_datetime(race_results['date'])
        race_results['classification'] = race_results['classification'].astype(str)

        leader_power_path = get_leader_power_path(race_class)
        leader_df = pd.read_csv(leader_power_path)
        leader_df['date'] = pd.to_datetime(leader_df['date'])
        leader_df['classification'] = leader_df['classification'].astype(str)

        merge_on = ['race', 'date', 'rider', 'cluster', 'classification', 'rank_number']
        leader_df = pd.merge(race_results[merge_on], leader_df, on=merge_on, how='inner')
        merge_on.append('team')

        team_power_path = get_team_power_path(race_class, scheme)
        team_df = pd.read_csv(team_power_path)
        team_df['date'] = pd.to_datetime(team_df['date'])
        team_df['classification'] = team_df['classification'].astype(str)

        full_df = pd.merge(
            team_df[merge_on + RAW_TRUE_SKILL_TEAMMATE_FEATURES],
            leader_df[merge_on + RAW_TRUE_SKILL_LEADER_FEATURES],
            on=merge_on, how='inner'
        )
        full_df['race_class'] = full_df['classification'].apply(map_class)
        full_df['year'] = full_df['date'].dt.year

        # split
        val_df  = full_df[full_df['year'] == (year - 1)].copy()
        test_df = full_df[full_df['year'] == year].copy()

        print(f'Validation set shape: {val_df.shape}')
        print(f'Validation set race class distribution: {val_df["race_class"].value_counts()}')
        print(f'Test set shape: {test_df.shape}')
        print(f'Test set race class distribution: {test_df["race_class"].value_counts()}')

        # ---------------- tune by race_class ----------------
        rows_class = []
        for rc, val_subset in tqdm(val_df.groupby('race_class'), desc='Tuning by race_class'):
            for k_pen in k_grid:
                # tune on validation
                tuned_lambda, val_ndcg, val_topk, n_val = self._tune_lambda_on_subset(
                    val_subset, k_pen, scheme, lambda_grid, k_value, top_n
                )
                # evaluate on test (same context)
                test_subset = test_df[test_df['race_class'] == rc]
                test_ndcg, test_topk, n_test = self._eval_on_subset(
                    test_subset, tuned_lambda, k_pen, scheme, k_value, top_n
                )

                rows_class.append({
                    'race_class': rc,
                    'tuned_lambda': tuned_lambda,
                    'VAL_NDCG@{}'.format(k_value): round(val_ndcg, 4) if pd.notna(val_ndcg) else np.nan,
                    'VAL_Top-{}'.format(k_value):  round(val_topk, 4) if pd.notna(val_topk) else np.nan,
                    'VAL_races': n_val,
                    'TEST_NDCG@{}'.format(k_value): round(test_ndcg, 4) if pd.notna(test_ndcg) else np.nan,
                    'TEST_Top-{}'.format(k_value):  round(test_topk, 4) if pd.notna(test_topk) else np.nan,
                    'TEST_races': n_test,
                    'k_penalty': k_pen,
                    'scheme': scheme,
                    'year_test': year
                })

        df_class = pd.DataFrame(rows_class).sort_values('race_class')
        df_class.to_csv(os.path.join(output_path, 'results_by_class_best_penalty.csv'), index=False)

        # ---------------- tune by cluster ----------------
        rows_cluster = []
        for clus, val_subset in tqdm(val_df.groupby('cluster'), desc='Tuning by cluster'):
            for k_pen in tqdm(k_grid, desc='Tuning by k_penalty'):
                tuned_lambda, val_ndcg, val_topk, n_val = self._tune_lambda_on_subset(
                    val_subset, k_pen, scheme, lambda_grid, k_value, top_n
                )
                test_subset = test_df[test_df['cluster'] == clus]
                test_ndcg, test_topk, n_test = self._eval_on_subset(
                    test_subset, tuned_lambda, k_pen, scheme, k_value, top_n
                )
                print(f'Tuned lambda: {tuned_lambda} for k_penalty: {k_pen} for cluster: {clus}')
                print(f'Validation NDCG: {val_ndcg}')
                print(f'Validation Top-10 Accuracy: {val_topk}')
                print(f'Test NDCG: {test_ndcg}')
                print(f'Test Top-10 Accuracy: {test_topk}')
                print('------------------------------------------------\n\n')

                rows_cluster.append({
                    'cluster': clus,
                    'tuned_lambda': tuned_lambda,
                    'VAL_NDCG@{}'.format(k_value): round(val_ndcg, 4) if pd.notna(val_ndcg) else np.nan,
                    'VAL_Top-{}'.format(k_value):  round(val_topk, 4) if pd.notna(val_topk) else np.nan,
                    'VAL_races': n_val,
                    'TEST_NDCG@{}'.format(k_value): round(test_ndcg, 4) if pd.notna(test_ndcg) else np.nan,
                    'TEST_Top-{}'.format(k_value):  round(test_topk, 4) if pd.notna(test_topk) else np.nan,
                    'TEST_races': n_test,
                    'k_penalty': k_pen,
                    'scheme': scheme,
                    'year_test': year
                })

        df_cluster = pd.DataFrame(rows_cluster).sort_values('cluster')
        df_cluster.to_csv(os.path.join(output_path, f'{scheme}_by_cluster_tuned_penalty.csv'), index=False)

        # print(f"[λ-by-context] Saved:\n  {os.path.join(output_path, 'results_by_class_best_penalty.csv')}\n  {os.path.join(output_path, 'results_by_cluster_best_penalty.csv')}")
        return output_path

    def _calculate_teammate_marginal_scores(self, grp, k_pen, num_teammates=None):
        """
        Calculate marginal teammate scores by adding teammates incrementally.
        
        Args:
            grp: Race group data
            k_pen: K penalty parameter  
            num_teammates: Number of top teammates to include (None = all teammates)
        
        Returns:
            pd.Series: Average teammate scores for each rider using top N teammates
        """
        teammate_avg_perf = []
        
        for idx, row in grp.iterrows():
            teammates = grp[(grp['team'] == row['team']) & (grp['rider'] != row['rider'])]
            
            if len(teammates) > 0:
                # Calculate individual teammate performances
                teammate_perfs = teammates['race_cluster_teammate_mu'] - k_pen * teammates['race_cluster_teammate_sigma']
                
                # Sort teammates by performance (best first)
                teammate_perfs_sorted = teammate_perfs.sort_values(ascending=False)
                
                # Take only the top N teammates if specified
                if num_teammates is not None:
                    teammate_perfs_sorted = teammate_perfs_sorted.head(num_teammates)
                
                # Calculate average of selected teammates
                if len(teammate_perfs_sorted) > 0:
                    avg_teammate_perf = teammate_perfs_sorted.mean()
                else:
                    avg_teammate_perf = 0.0
            else:
                avg_teammate_perf = 0.0
                
            teammate_avg_perf.append(avg_teammate_perf)
            
        return pd.Series(teammate_avg_perf, index=grp.index)

    def run_marginal_teammate_contribution_analysis(
            self,
            race_class,
            year,
            scheme,
            k_penalty,
            lambda_value,
            k_value=None,
            max_teammates=None
    ):
        """
        Analyze the marginal contribution of teammates by incrementally adding them
        based on their skill scores and evaluating performance on the test set.
        
        This function evaluates how performance changes as we add teammates one by one,
        starting with the best teammate and progressively adding the next best ones.
        
        Args:
            race_class (str): Race class to evaluate ('all', 'WT', etc.)
            year (int): Target year for testing
            scheme (str): Teammate scoring scheme to use
            k_penalty (float): K penalty parameter (fixed, no tuning)
            lambda_value (float): Lambda value for teammate contribution (fixed, no tuning)
            k_value (int): K value for NDCG@k and Top-k accuracy evaluation (default: 10)
            max_teammates (int): Maximum number of teammates to analyze (default: 8)
        
        Returns:
            str: Output directory path containing results
        """
        
        race_class = race_class if race_class is not None else self.race_class
        k_value = k_value if k_value is not None else self.k_value
        max_teammates = max_teammates if max_teammates is not None else self.max_teammates

        print(f"Starting Marginal Teammate Contribution Analysis")
        print(f"Race class: {race_class}, Year: {year}, Scheme: {scheme}")
        print(f"K penalty: {k_penalty}, Lambda: {lambda_value}")
        print(f"Max teammates: {max_teammates}")
        
        # Load race results data
        race_results_path = get_data_dir('riders_race_results')
        race_results = pd.read_csv(race_results_path)
        race_results['date'] = pd.to_datetime(race_results['date'])
        race_results['classification'] = race_results['classification'].astype(str)
        
        merge_on = ['race', 'date', 'rider', 'cluster','classification','rank_number']

        # Setup output directory
        output_path = get_output_dir('direct_ranking', race_class=race_class)
        output_path = os.path.join(output_path, 'marginal_contribution', scheme)
        os.makedirs(output_path, exist_ok=True)

        # Load leader features
        leader_power_path = get_leader_power_path(race_class)
        true_skill_features_leader = pd.read_csv(leader_power_path)
        true_skill_features_leader['date'] = pd.to_datetime(true_skill_features_leader['date'])
        true_skill_features_leader['year'] = true_skill_features_leader['date'].dt.year
        true_skill_features_leader['classification'] = true_skill_features_leader['classification'].astype(str)

        true_skill_features_leader = pd.merge(
            race_results[merge_on], 
            true_skill_features_leader, 
            on=merge_on, 
            how='inner'
        )
        merge_on.append('team')
        
        # Load teammate data
        team_power_path = get_team_power_path(race_class, scheme)
        team_df = pd.read_csv(team_power_path)
        team_df['date'] = pd.to_datetime(team_df['date'])
        team_df['classification'] = team_df['classification'].astype(str)
        
        print(f"Merging leader features {true_skill_features_leader.shape} with teammate features {team_df.shape}")
        
        # Merge leader and teammate features
        true_skill_features = pd.merge(
            team_df[merge_on + RAW_TRUE_SKILL_TEAMMATE_FEATURES], 
            true_skill_features_leader[merge_on + RAW_TRUE_SKILL_LEADER_FEATURES], 
            on=merge_on, 
            how='inner'
        )
        
        print(f'Final features shape: {true_skill_features.shape}')

        # Filter to test set only
        true_skill_features['year'] = true_skill_features['date'].dt.year
        test_df = true_skill_features[true_skill_features['year'] == year].copy()
        
        print(f"Test set shape: {test_df.shape}")
        
        # Results storage
        all_results = []
        
        # Evaluate for different numbers of teammates (0 to max_teammates)
        teammate_counts = list(range(0, max_teammates + 1))
        
        print(f"Evaluating marginal contribution for teammate counts: {teammate_counts}")
        
        for num_teammates in tqdm(teammate_counts, desc='Teammate counts'):
            print(f"\nEvaluating with {num_teammates} teammates...")
            
            race_results = []
            
            # Evaluate each race in test set
            for (race_name, race_date), grp in test_df.groupby(['race','date']):
                # Calculate leader performance
                leader_perf = grp['race_cluster_leader_mu'] - k_penalty * grp['race_cluster_leader_sigma']
                
                if num_teammates == 0:
                    # Baseline: no teammates (lambda_value * 0)
                    combined_perf = leader_perf
                    avg_teammates_used = 0
                else:
                    # Calculate teammate performance with specified number of teammates
                    teammate_avg_perf = self._calculate_teammate_marginal_scores(grp, k_penalty, num_teammates)
                    combined_perf = leader_perf + lambda_value * teammate_avg_perf
                    
                    # Calculate average number of teammates actually used in this race
                    actual_teammates_used = []
                    for idx, row in grp.iterrows():
                        teammates = grp[(grp['team'] == row['team']) & (grp['rider'] != row['rider'])]
                        actual_used = min(len(teammates), num_teammates)
                        actual_teammates_used.append(actual_used)
                    avg_teammates_used = np.mean(actual_teammates_used)
                
                # Sort by performance to get predictions
                grp_with_perf = grp.copy()
                grp_with_perf['combined_perf'] = combined_perf
                pred = grp_with_perf.sort_values('combined_perf', ascending=False)['rider']
                true = grp.sort_values('rank_number')['rider']
                
                # Calculate metrics
                ndcg = ndcg_at_k(list(pred), list(true), k_value)
                top_k_accuracy = recall_at_k(list(pred), list(true), k_value)
                
                race_result = {
                    'race': race_name,
                    'date': race_date,
                    'classification': grp['classification'].iloc[0],
                    'cluster': grp['cluster'].iloc[0],
                    'num_teammates': num_teammates,
                    'avg_teammates_used': round(avg_teammates_used, 2),
                    'k_penalty': k_penalty,
                    'lambda': lambda_value,
                    'scheme': scheme,
                    'NDCG@10': round(ndcg, 4),
                    'Top-10 Accuracy': round(top_k_accuracy, 4),
                    'total_riders': len(grp)
                }
                race_results.append(race_result)
            
            all_results.extend(race_results)
            
            # Print progress summary
            race_results_df = pd.DataFrame(race_results)
            mean_ndcg = race_results_df['NDCG@10'].mean()
            mean_accuracy = race_results_df['Top-10 Accuracy'].mean()
            mean_teammates_used = race_results_df['avg_teammates_used'].mean()
            
            print(f"  {num_teammates} teammates -> NDCG@10: {mean_ndcg:.4f}, "
                  f"Top-10 Acc: {mean_accuracy:.4f}, Avg used: {mean_teammates_used:.2f}")
        
        # Convert to DataFrame and save detailed results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(output_path, 'marginal_contribution_detailed.csv'), index=False)
        
        # Create summary statistics
        summary_stats = results_df.groupby(['num_teammates']).agg({
            'NDCG@10': ['mean', 'std', 'count'],
            'Top-10 Accuracy': ['mean', 'std'],
            'avg_teammates_used': 'mean'
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()
        
        summary_stats.to_csv(os.path.join(output_path, 'marginal_contribution_summary.csv'), index=False)
        
        # Create summary by classification
        class_summary_stats = results_df.groupby(['num_teammates', 'classification']).agg({
            'NDCG@10': 'mean',
            'Top-10 Accuracy': 'mean',
            'avg_teammates_used': 'mean'
        }).round(4).reset_index()
        
        class_summary_stats.to_csv(os.path.join(output_path, 'marginal_contribution_by_classification.csv'), index=False)
        
        # Create summary by cluster
        cluster_summary_stats = results_df.groupby(['num_teammates', 'cluster']).agg({
            'NDCG@10': 'mean',
            'Top-10 Accuracy': 'mean', 
            'avg_teammates_used': 'mean'
        }).round(4).reset_index()
        
        cluster_summary_stats.to_csv(os.path.join(output_path, 'marginal_contribution_by_cluster.csv'), index=False)
        
        # Calculate marginal improvements
        marginal_summary = summary_stats[['num_teammates', 'NDCG@10_mean', 'Top-10 Accuracy_mean']].copy()
        marginal_summary['NDCG@10_marginal'] = marginal_summary['NDCG@10_mean'].diff()
        marginal_summary['Top-10 Accuracy_marginal'] = marginal_summary['Top-10 Accuracy_mean'].diff()
        marginal_summary['NDCG@10_marginal'] = marginal_summary['NDCG@10_marginal'].fillna(marginal_summary['NDCG@10_mean'])
        marginal_summary['Top-10 Accuracy_marginal'] = marginal_summary['Top-10 Accuracy_marginal'].fillna(marginal_summary['Top-10 Accuracy_mean'])
        
        marginal_summary.to_csv(os.path.join(output_path, 'marginal_improvements.csv'), index=False)
        
        print(f"\nMarginal Teammate Contribution Analysis completed!")
        print(f"Results saved to: {output_path}")
        
        print(f"\nSummary of marginal improvements:")
        print("=" * 60)
        print(f"{'Teammates':<12} {'NDCG@10':<12} {'Marginal':<12} {'Top-10 Acc':<12} {'Marginal':<12}")
        print("=" * 60)
        
        for _, row in marginal_summary.iterrows():
            print(f"{int(row['num_teammates']):<12} "
                  f"{row['NDCG@10_mean']:<12.4f} "
                  f"{row['NDCG@10_marginal']:<12.4f} "
                  f"{row['Top-10 Accuracy_mean']:<12.4f} "
                  f"{row['Top-10 Accuracy_marginal']:<12.4f}")
        
        # Find the point of diminishing returns
        ndcg_marginals = marginal_summary['NDCG@10_marginal'].iloc[1:]  # Skip baseline (0 teammates)
        
        if len(ndcg_marginals) > 0:
            # Find the number of teammates where marginal improvement drops below threshold
            threshold = 0.001  # Threshold for diminishing returns
            diminishing_point = None
            
            for i, marginal in enumerate(ndcg_marginals):
                if marginal < threshold:
                    diminishing_point = i + 1  # +1 because we skipped baseline
                    break
            
            if diminishing_point:
                print(f"\nDiminishing returns point: {diminishing_point} teammates")
                print(f"(marginal NDCG@10 improvement < {threshold})")
            else:
                print(f"\nNo clear diminishing returns point found with threshold {threshold}")
        
        # Show best performance
        best_perf = summary_stats.loc[summary_stats['NDCG@10_mean'].idxmax()]
        print(f"\nBest performance: {int(best_perf['num_teammates'])} teammates")
        print(f"NDCG@10: {best_perf['NDCG@10_mean']:.4f}")
        print(f"Top-10 Accuracy: {best_perf['Top-10 Accuracy_mean']:.4f}")
        
        return output_path