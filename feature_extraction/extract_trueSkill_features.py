import pandas as pd
import numpy as np
from trueskill import Rating, rate, setup
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from roster_ranker.feature_extraction.preprocess import *
from scipy.stats import norm, zscore
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from roster_ranker.utils import *
from roster_ranker.data import *
from datetime import timedelta

def create_leader_data_set(data_set, ranking_by_race_class_induv, race_class='all', time_gap=None, previous_snapshot_df=None):
    data_set = data_set.sort_values('date')
    
    # Load previous snapshots and initialize ratings if provided
    if previous_snapshot_df is not None:
        print(f'Loading previous snapshot data for incremental calculation...', flush=True)
        
        # Ensure date column is datetime
        previous_snapshot_df['date'] = pd.to_datetime(previous_snapshot_df['date'])
        
        # Get the last date from previous snapshot
        last_snapshot_year = previous_snapshot_df['date'].max().year
        if time_gap != 'season_start':
            last_snapshot_date = previous_snapshot_df['date'].max() - timedelta(days=time_gap)
        else:
            last_snapshot_date = pd.to_datetime(f'{last_snapshot_year}-01-01')
        print(f'Last snapshot date: {last_snapshot_date}', flush=True)
        
        # Filter data_set to only include races AFTER the last snapshot date
        data_set = data_set[data_set['date'] > last_snapshot_date].copy()
        print(f'Processing {len(data_set)} new race entries after {last_snapshot_date}', flush=True)

        # Initialize ratings from the last snapshot for each rider in each cluster
        indiv_ratings = ranking_by_race_class_induv['induv_rating']
        last_update_induv = ranking_by_race_class_induv['induv_last_update']
        
        # Get the last rating for each rider in each cluster
        total_riders_loaded = 0
        for cluster in previous_snapshot_df['cluster'].unique():
            cluster_snapshot = previous_snapshot_df[previous_snapshot_df['cluster'] == cluster].copy()
            
            # Get the last snapshot for each rider in this cluster
            last_rider_snapshot = cluster_snapshot.sort_values('date').groupby('rider').last()
            
            # Populate the rating dict for this cluster
            for rider, row in last_rider_snapshot.iterrows():
                indiv_ratings[cluster][rider] = Rating(mu=row['race_cluster_leader_mu'], 
                                                       sigma=row['race_cluster_leader_sigma'])
                if not pd.isna(row['race_cluster_last_update_leader']):
                    last_update_induv[cluster][rider] = row['date'] - pd.Timedelta(days=row['race_cluster_last_update_leader'])
                else:
                    last_update_induv[cluster][rider] = last_snapshot_date
                total_riders_loaded += 1
                
        # Also initialize GC ratings for all riders
        gc_snapshot = previous_snapshot_df.sort_values('date').groupby('rider').last()
        gc_riders_loaded = 0
        for rider, row in gc_snapshot.iterrows():
            indiv_ratings['General Classification'][rider] = Rating(mu=row['gc_leader_mu'], 
                                                                    sigma=row['gc_leader_sigma'])
            if not pd.isna(row['gc_last_update_leader']):
                last_update_induv['General Classification'][rider] = row['date'] - pd.Timedelta(days=row['gc_last_update_leader'])
            else:
                last_update_induv['General Classification'][rider] = last_snapshot_date
            gc_riders_loaded += 1
        
        print(f'Loaded ratings for clusters: {list(previous_snapshot_df["cluster"].unique())}', flush=True)
        print(f'Total riders loaded: {total_riders_loaded} (cluster-specific), {gc_riders_loaded} (GC)', flush=True)
        
        # Add previous snapshots to the list
        race_ratings_snapshot_list = [previous_snapshot_df]
    else:
        race_ratings_snapshot_list = []
        indiv_ratings = ranking_by_race_class_induv['induv_rating']
        last_update_induv = ranking_by_race_class_induv['induv_last_update']
    
    # Get groups as a list of tuples: ((race, date), group)
    grouped_races = list(data_set.groupby(['race', 'date']))

    # Sort groups by date (the second element of the key)
    grouped_races = sorted(grouped_races, key=lambda x: x[0][1])

    # Get the list of all clusters that appear in the training data.
    all_clusters = data_set['cluster'].unique()

    setup(backend='mpmath')

    print(f'Running Leader trueSkill features for {race_class} Race Class, and race clusters {all_clusters}', flush=True)
    
    # Time gap support: track last update by cluster for batch processing
    if time_gap is not None:
        if previous_snapshot_df is not None:
            # Initialize last_update_by_cluster from previous snapshot
            last_update_by_cluster = defaultdict(lambda: {'last_update': pd.to_datetime('2022-12-31')})
            for cluster in previous_snapshot_df['cluster'].unique():
                cluster_dates = previous_snapshot_df[previous_snapshot_df['cluster'] == cluster]['date']
                if len(cluster_dates) > 0:
                    last_update_by_cluster[cluster]['last_update'] = cluster_dates.max() - timedelta(days=time_gap) if time_gap != 'season_start' else pd.to_datetime(f'{last_snapshot_year}-01-01')
        else:
            last_update_by_cluster = defaultdict(lambda: {'last_update': pd.to_datetime('2016-12-31')})
        
    for (race, race_date), riders_group in tqdm(grouped_races, desc='Running Races: ', total=len(grouped_races)):
        
        if race_date.year <= last_snapshot_year:
            continue
        # Get cluster value from the riders_group (assumed common for the race)
        cluster = riders_group['cluster'].iloc[0]
        cluster_indiv_rating = indiv_ratings[cluster]
        race_class = riders_group['classification'].iloc[0]
        race_date = riders_group['date'].iloc[0]
        
        # Time gap processing: update ratings up to decision point
        if time_gap is not None:
            if time_gap != 'season_start':
                decision_date = race_date - pd.Timedelta(days=time_gap)
            else:
                season = race_date.year
                decision_date = pd.to_datetime(f'{season}-01-01')

            # Get races to update for this cluster up to decision point
            last_update_date = last_update_by_cluster[cluster]['last_update']
            races_to_update = data_set[(data_set['cluster'] == cluster) & 
                                    (data_set['date'] <= decision_date) &
                                    (data_set['date'] > last_update_date)].copy()
            
            if not races_to_update.empty:
                for update_race_name, update_race_df in races_to_update.groupby('race'):
                    current_race_date = pd.to_datetime(update_race_df['date'].iloc[0])
                    
                    # Filter out riders with DNF or DF ranks from skill updates
                    riders_for_updates = update_race_df[~update_race_df['rank'].isin(['DNF', 'DF'])].copy()
                    
                    if len(riders_for_updates) == 0:
                        continue
                        
                    # Individual rating updates
                    indiv_teams = []
                    indiv_ranks = []
                    for rider, rider_df in riders_for_updates.groupby('rider'):
                        indiv_teams.append([cluster_indiv_rating.get(rider, Rating(mu=25.0, sigma=8.333333))])
                        indiv_ranks.append(rider_df['rank_number'].min())

                    new_indiv_teams = rate(indiv_teams, ranks=indiv_ranks)

                    for updated_rating, (rider, _) in zip(new_indiv_teams, riders_for_updates.groupby('rider')):
                        cluster_indiv_rating[rider] = updated_rating[0]
                        last_update_induv[cluster][rider] = current_race_date

                # Update last update date for this cluster
                last_update_by_cluster[cluster]['last_update'] = decision_date
                
        # Create a snapshot for each rider.
        # Instead of storing only the ratings for the current cluster,
        # we store the ratings for all clusters in a single record.
        snapshot_records = []
        for rider in riders_group['rider'].unique():
            induv_obj = cluster_indiv_rating.get(rider, Rating(mu=25.0, sigma=8.333333))
            induv_obj_gc = indiv_ratings['General Classification'].get(rider, Rating(mu=25.0, sigma=8.333333))
            record = {
                'race': race,
                'date': race_date,
                'classification': race_class,
                'rider': rider,
                'team': riders_group[riders_group['rider'] == rider]['team'].iloc[0],
                'rank_number': riders_group[riders_group['rider'] == rider]['rank_number'].iloc[0],
                'team_rank': riders_group[riders_group['rider'] == rider]['final_team_rank'].iloc[0],
                'cluster': cluster,  # the race's own cluster, if needed
                'race_cluster_leader_mu': round(induv_obj.mu, 2),
                'race_cluster_leader_sigma': round(induv_obj.sigma, 2),
                'gc_leader_mu': round(induv_obj_gc.mu, 2),
                'gc_leader_sigma': round(induv_obj_gc.sigma, 2),
            }
            last_update = last_update_induv[cluster].get(rider, None)
            if last_update is None:
                record[f'race_cluster_last_update_leader'] = np.nan
            else:
                record[f'race_cluster_last_update_leader'] = (race_date - last_update).days
            last_update_gc = last_update_induv['General Classification'].get(rider, None)
            if last_update_gc is None:
                record['gc_last_update_leader'] = np.nan
            else:
                record['gc_last_update_leader'] = (race_date - last_update_gc).days
            snapshot_records.append(record)
        
        # Append the DataFrame for the current race snapshot.
        race_ratings_snapshot_list.append(pd.DataFrame(snapshot_records))

        # Only update ratings if time_gap is None (real-time updates)
        if time_gap is None:
            #########################
            # 1) INDIVIDUAL UPDATES
            #########################
            # Filter out riders with DNF or DF ranks from skill updates
            riders_for_updates = riders_group[~riders_group['rank'].isin(['DNF', 'DF'])].copy()
            
            if len(riders_for_updates) == 0:
                # No valid riders to update, skip this race
                continue
                
            # single-rider teams
            indiv_teams = []
            indiv_ranks = []
            for rider, rider_df in riders_for_updates.groupby('rider'):
                indiv_teams.append([cluster_indiv_rating.get(rider, Rating(mu=25.0, sigma=8.333333))])
                indiv_ranks.append(rider_df['rank_number'].min())

            new_indiv_teams  = rate(indiv_teams, ranks=indiv_ranks)

            for updated_rating, (rider, _) in zip(new_indiv_teams, riders_for_updates.groupby('rider')):
                cluster_indiv_rating[rider] = updated_rating[0]
                last_update_induv[cluster][rider] = race_date

    # Combine all the race ratings snapshots into one DataFrame.
    ratings_snapshot_df = pd.concat(race_ratings_snapshot_list, ignore_index=True)
    return ratings_snapshot_df

def create_leader_data_set_by_cluster_class(data_set, race_class='all', time_gap=None):
    """
    Create leader dataset with TrueSkill ratings for each (cluster, class) combination.
    Similar to create_leader_data_set but maintains separate ratings for each cluster-class pair.
    """
    data_set = data_set.sort_values('date')
    # Get groups as a list of tuples: ((race, date), group)
    grouped_races = list(data_set.groupby(['race', 'date']))

    # Sort groups by date
    grouped_races = sorted(grouped_races, key=lambda x: x[0][1])

    # Get unique clusters and race classes
    all_clusters = data_set['cluster'].unique()
    all_race_classes = data_set['race_class'].unique()

    setup(backend='mpmath')

    print(f'Running {race_class} Race Class with cluster-class pairs...', flush=True)
    
    # Initialize ratings and last updates for each cluster-class pair
    indiv_ratings = {}
    last_update_induv = {}
    for cluster in all_clusters:
        indiv_ratings[cluster] = {}
        last_update_induv[cluster] = {}
        for rc in all_race_classes:
            indiv_ratings[cluster][rc] = {}
            last_update_induv[cluster][rc] = {}

    # To store the snapshot of rider ratings for each race
    race_ratings_snapshot_list = []
    
    # Time gap support: track last update by cluster-class pair for batch processing
    if time_gap is not None:
        last_update_by_cluster_class = defaultdict(lambda: defaultdict(lambda: {'last_update': pd.to_datetime('2016-12-31')}))
        
    for (race, race_date), riders_group in tqdm(grouped_races, desc='Running Races: ', total=len(grouped_races)):
        
        # Get cluster and class values from the riders_group
        cluster = riders_group['cluster'].iloc[0]
        race_class_value = riders_group['race_class'].iloc[0]
        race_date = riders_group['date'].iloc[0]
        
        # Time gap processing: update ratings up to decision point
        if time_gap is not None:
            if time_gap != 'season_start':
                decision_date = race_date - pd.Timedelta(days=time_gap)
            else:
                season = race_date.year
                decision_date = pd.to_datetime(f'{season}-01-01')

            # Get races to update for this cluster-class pair up to decision point
            last_update_date = last_update_by_cluster_class[cluster][race_class_value]['last_update']
            races_to_update = data_set[(data_set['cluster'] == cluster) & 
                                    (data_set['race_class'] == race_class_value) &
                                    (data_set['date'] <= decision_date) &
                                    (data_set['date'] > last_update_date)].copy()
            
            if not races_to_update.empty:
                for update_race_name, update_race_df in races_to_update.groupby('race'):
                    current_race_date = pd.to_datetime(update_race_df['date'].iloc[0])
                    
                    # Filter out riders with DNF or DF ranks from skill updates
                    riders_for_updates = update_race_df[~update_race_df['rank'].isin(['DNF', 'DF'])].copy()
                    
                    if len(riders_for_updates) == 0:
                        continue
                        
                    # Individual rating updates
                    indiv_teams = []
                    indiv_ranks = []
                    for rider, rider_df in riders_for_updates.groupby('rider'):
                        indiv_teams.append([indiv_ratings[cluster][race_class_value].get(rider, Rating(mu=25.0, sigma=8.333333))])
                        indiv_ranks.append(rider_df['rank_number'].min())

                    new_indiv_teams = rate(indiv_teams, ranks=indiv_ranks)

                    for updated_rating, (rider, _) in zip(new_indiv_teams, riders_for_updates.groupby('rider')):
                        indiv_ratings[cluster][race_class_value][rider] = updated_rating[0]
                        last_update_induv[cluster][race_class_value][rider] = current_race_date

                # Update last update date for this cluster-class pair
                last_update_by_cluster_class[cluster][race_class_value]['last_update'] = decision_date
        
        # Create a snapshot for each rider
        snapshot_records = []
        for rider in riders_group['rider'].unique():
            # Get rating for current cluster-class pair
            induv_obj = indiv_ratings[cluster][race_class_value].get(rider, Rating(mu=25.0, sigma=8.333333))
            induv_obj_gc = indiv_ratings['General Classification'][race_class_value].get(rider, Rating(mu=25.0, sigma=8.333333))
            record = {
                'race': race,
                'date': race_date,
                'race_class': race_class_value,
                'classification': riders_group['classification'].iloc[0],
                'rider': rider,
                'team': riders_group[riders_group['rider'] == rider]['team'].iloc[0],
                'rank_number': riders_group[riders_group['rider'] == rider]['rank_number'].iloc[0],
                'team_rank': riders_group[riders_group['rider'] == rider]['final_team_rank'].iloc[0],
                'cluster': cluster,
                'race_cluster_class_leader_mu': round(induv_obj.mu, 2),
                'race_cluster_class_leader_sigma': round(induv_obj.sigma, 2),
                'gc_class_leader_mu': round(induv_obj_gc.mu, 2),
                'gc_class_leader_sigma': round(induv_obj_gc.sigma, 2),
            }
            
            # Add last update for current cluster-class pair
            last_update = last_update_induv[cluster][race_class_value].get(rider, None)
            if last_update is None:
                record['race_cluster_class_last_update_leader'] = np.nan
            else:
                record['race_cluster_class_last_update_leader'] = (race_date - last_update).days

            # Add last update for General Classification
            last_update_gc = last_update_induv['General Classification'][race_class_value].get(rider, None)
            if last_update_gc is None:
                record['gc_class_last_update_leader'] = np.nan
            else:
                record['gc_class_last_update_leader'] = (race_date - last_update_gc).days

            snapshot_records.append(record)
        
        # Append the DataFrame for the current race snapshot
        race_ratings_snapshot_list.append(pd.DataFrame(snapshot_records))

        # Only update ratings if time_gap is None (real-time updates)
        if time_gap is None:
            #########################
            # 1) INDIVIDUAL UPDATES
            #########################
            # Filter out riders with DNF or DF ranks from skill updates
            riders_for_updates = riders_group[~riders_group['rank'].isin(['DNF', 'DF'])].copy()
            
            if len(riders_for_updates) == 0:
                continue
                
            # single-rider teams
            indiv_teams = []
            indiv_ranks = []
            for rider, rider_df in riders_for_updates.groupby('rider'):
                indiv_teams.append([indiv_ratings[cluster][race_class_value].get(rider, Rating(mu=25.0, sigma=8.333333))])
                indiv_ranks.append(rider_df['rank_number'].min())

            new_indiv_teams = rate(indiv_teams, ranks=indiv_ranks)

            for updated_rating, (rider, _) in zip(new_indiv_teams, riders_for_updates.groupby('rider')):
                indiv_ratings[cluster][race_class_value][rider] = updated_rating[0]
                last_update_induv[cluster][race_class_value][rider] = race_date

    # Combine all the race ratings snapshots into one DataFrame
    ratings_snapshot_df = pd.concat(race_ratings_snapshot_list, ignore_index=True)
    return ratings_snapshot_df

def build_weights(team_df, race_df, scheme):
    """
    team_df : rows for one team
    race_df : rows for all riders in the race  (needed for race‑wide max)
    scheme  : 'rank', 'gap', or 'topK'
    """

    rider_ids = team_df['rider'].tolist()
        
    if scheme=='rank_norm':
        N = len(race_df)       # number of finishers
        norm_rank = lambda r: (r-1)/(N-1)
        raw = [1-norm_rank(team_df.loc[team_df['rider']==r, 'rank_number'].iloc[0])
               for r in rider_ids] 
        
    elif scheme=='time_lag':

        rider_ids = team_df['rider'].tolist()
        # 1. Compute z-scores for the whole peloton
        gaps = race_df['log2_timeLag_stage_sec'].values
        if np.std(gaps) == 0:                      # bunch sprint, all gaps zero
            return tuple(1/len(rider_ids) for _ in rider_ids)

        # Clip outliers at 95th percentile (or 99th)
        # upper_bound = np.percentile(gaps, 99)  # You can adjust this
        # gaps_clipped = np.clip(gaps, None, upper_bound)
    
        z_all = zscore(gaps)                       # mean 0, std 1
        z_map = dict(zip(race_df['rider'], z_all))

        # 2. Raw credit = 1 - Φ(z)
        raw = [1.0 - norm.cdf(z_map[r]) for r in rider_ids]

    elif scheme=='time_lag_perc':
        rider_ids = team_df['rider'].tolist()
    
        # Step 1: Calculate raw percentage increases for ALL riders in race
        winner_idx = race_df['rank_number'].idxmin()
        winner_time = race_df.loc[winner_idx, 'time_sec']
        
        all_percent_increases = []
        for _, row in race_df.iterrows():
            percent_inc = (row['time_sec'] - winner_time) / winner_time
            all_percent_increases.append(percent_inc)
        
        # Step 2: Race-wide normalization parameters
        max_increase = np.percentile(all_percent_increases, 95)  # Use 95th percentile to handle outliers
        
        # Step 3: Calculate team weights
        raw = []
        for rider in rider_ids:
            rider_time = team_df.loc[team_df['rider'] == rider, 'time_sec'].iloc[0]
            percent_increase = (rider_time - winner_time) / winner_time
            
            # Normalize by race context, then invert
            normalized_increase = min(percent_increase / max_increase, 1.0)  # Cap at 1.0
            weight = 1.0 - normalized_increase
            
            # Round to 3 decimal places to prevent floating point precision issues
            weight = round(weight, 3)
            raw.append(weight)

    elif scheme=='time_lag_norm':
        # 1. Compute z-scores for the whole peloton
        rider_ids = team_df['rider'].tolist()

        # Compute z-scores for the log-transformed time gaps.
        gaps = race_df['log2_timeLag_stage_sec'].values
        if np.std(gaps) == 0:  # e.g. in a bunch sprint where all gaps are zero.
            return tuple(1/len(rider_ids) for _ in rider_ids)

        z_all = zscore(gaps)  # Standardized values, mean=0, std=1
        z_map = dict(zip(race_df['rider'], z_all))
        
        # Compute the minimum and maximum z-values.
        z_values = np.array(list(z_map.values()))
        z_min = z_values.min()
        z_max = z_values.max()
        
        # If z_max equals z_min, return uniform weights.
        if z_max == z_min:
            raw = [1/len(rider_ids)] * len(rider_ids)
        else:
            raw = [(z_max - z_map[r]) / (z_max - z_min) for r in rider_ids]
    else:
        # Equal weights
        raw = [1.0 / len(rider_ids) for _ in rider_ids]

    s = sum(raw); 
    return tuple(round(w/s, 3) if s else 1/len(raw) for w in raw)

def create_data_set_team_weights(data_set, ranking_by_race_class_team, race_class='all', scheme='time_lag', helpers_only=False, time_gap=None, previous_snapshot_df=None):
    data_set = data_set.sort_values('date')
    
    # Load previous snapshots and initialize ratings if provided
    if previous_snapshot_df is not None:
        print(f'Loading previous snapshot data for incremental calculation...', flush=True)
        
        # Ensure date column is datetime
        previous_snapshot_df['date'] = pd.to_datetime(previous_snapshot_df['date'])
        
        # Get the last date from previous snapshot
        last_snapshot_year = previous_snapshot_df['date'].max().year
        if time_gap != 'season_start':
            last_snapshot_date = previous_snapshot_df['date'].max() - timedelta(days=time_gap)
        else:
            last_snapshot_date = pd.to_datetime(f'{last_snapshot_year}-01-01')
        print(f'Last snapshot date: {last_snapshot_date}', flush=True)
        
        # Filter data_set to only include races AFTER the last snapshot date
        data_set = data_set[data_set['date'] > last_snapshot_date].copy()
        print(f'Processing {len(data_set)} new race entries after {last_snapshot_date}', flush=True)
        
        # Initialize ratings from the last snapshot for each rider in each cluster
        team_ratings = ranking_by_race_class_team['teammate_rating']
        last_update_team = ranking_by_race_class_team['teammate_last_update']
        
        # Get the last rating for each rider in each cluster
        total_riders_loaded = 0
        for cluster in previous_snapshot_df['cluster'].unique():
            cluster_snapshot = previous_snapshot_df[previous_snapshot_df['cluster'] == cluster].copy()
            
            # Get the last snapshot for each rider in this cluster
            last_rider_snapshot = cluster_snapshot.sort_values('date').groupby('rider').last()
            
            # Populate the rating dict for this cluster
            for rider, row in last_rider_snapshot.iterrows():
                team_ratings[cluster][rider] = Rating(mu=row['race_cluster_teammate_mu'], 
                                                      sigma=row['race_cluster_teammate_sigma'])
                if not pd.isna(row['race_cluster_last_update_teammate']):
                    last_update_team[cluster][rider] = row['date'] - pd.Timedelta(days=row['race_cluster_last_update_teammate'])
                else:
                    last_update_team[cluster][rider] = last_snapshot_date
                total_riders_loaded += 1
                
        # Also initialize GC ratings for all riders
        gc_snapshot = previous_snapshot_df.sort_values('date').groupby('rider').last()
        gc_riders_loaded = 0
        for rider, row in gc_snapshot.iterrows():
            team_ratings['General Classification'][rider] = Rating(mu=row['gc_teammate_mu'], 
                                                                   sigma=row['gc_teammate_sigma'])
            if not pd.isna(row['gc_last_update_teammate']):
                last_update_team['General Classification'][rider] = row['date'] - pd.Timedelta(days=row['gc_last_update_teammate'])
            else:
                last_update_team['General Classification'][rider] = last_snapshot_date
            gc_riders_loaded += 1
        
        print(f'Loaded ratings for clusters: {list(previous_snapshot_df["cluster"].unique())}', flush=True)
        print(f'Total riders loaded: {total_riders_loaded} (cluster-specific), {gc_riders_loaded} (GC)', flush=True)
        
        # Add previous snapshots to the list
        race_ratings_snapshot_list = [previous_snapshot_df]
    else:
        race_ratings_snapshot_list = []
        team_ratings = ranking_by_race_class_team['teammate_rating']
        last_update_team = ranking_by_race_class_team['teammate_last_update']
    
    # Get groups as a list of tuples: ((race, date), group)
    grouped_races = list(data_set.groupby(['race', 'date']))

    # Sort groups by date (the second element of the key)
    grouped_races = sorted(grouped_races, key=lambda x: x[0][1])

    # Get the list of all clusters that appear in the training data.
    all_clusters = data_set['cluster'].unique()

    setup(backend='mpmath')

    # Increase precision for percentage-based schemes to handle numerical stability
    if scheme == 'time_lag_perc':
        import mpmath
        mpmath.mp.dps = 50  # Increase decimal precision

    print(f'Running {race_class} Race Class...', flush=True)
    
    # Time gap support: track last update by cluster for batch processing
    if time_gap is not None:
        if previous_snapshot_df is not None:
            # Initialize last_update_by_cluster from previous snapshot
            last_update_by_cluster = defaultdict(lambda: {'last_update': pd.to_datetime('2016-12-31')})
            for cluster in previous_snapshot_df['cluster'].unique():
                cluster_dates = previous_snapshot_df[previous_snapshot_df['cluster'] == cluster]['date']
                if len(cluster_dates) > 0:
                    last_update_by_cluster[cluster]['last_update'] = cluster_dates.max() - timedelta(days=time_gap) if time_gap != 'season_start' else pd.to_datetime(f'{last_snapshot_year}-01-01')
        else:
            last_update_by_cluster = defaultdict(lambda: {'last_update': pd.to_datetime('2016-12-31')})
        
    for (race, race_date), riders_group in tqdm(grouped_races, desc='Running Races: ', total=len(grouped_races)):
        
        if race_date.year <= last_snapshot_year:
            continue
        # Get cluster value from the riders_group (assumed common for the race)
        cluster = riders_group['cluster'].iloc[0]
        cluster_team_rating = team_ratings[cluster]
        race_class = riders_group['classification'].iloc[0]
        race_date = riders_group['date'].iloc[0]
        
        # Time gap processing: update ratings up to decision point
        if time_gap is not None:
            if time_gap != 'season_start':
                decision_date = race_date - pd.Timedelta(days=time_gap)
            else:
                season = race_date.year
                decision_date = pd.to_datetime(f'{season}-01-01')

            # Get races to update for this cluster up to decision point
            last_update_date = last_update_by_cluster[cluster]['last_update']
            races_to_update = data_set[(data_set['cluster'] == cluster) & 
                                    (data_set['date'] <= decision_date) &
                                    (data_set['date'] > last_update_date)].copy()
            
            if not races_to_update.empty:
                for update_race_name, update_race_df in races_to_update.groupby('race'):
                    current_race_date = pd.to_datetime(update_race_df['date'].iloc[0])
                    
                    # Team updates with weights
                    sponsor_teams_list = []
                    sponsor_team_ranks = []
                    sponsor_team_weights = []
                    sponsor_rider_map = {}

                    for sponsor, team_df in update_race_df.groupby('team'):
                        if helpers_only and len(team_df) > 1:
                            team_leader = team_df.loc[team_df['rank_number'].idxmin(), 'rider']
                            helpers_df = team_df[team_df['rider'] != team_leader].copy()
                            
                            if len(helpers_df) == 0:
                                continue
                                
                            rider_ids = helpers_df['rider'].unique().tolist()
                            ratings = [cluster_team_rating.get(r, Rating(mu=25.0, sigma=8.333333)) for r in rider_ids]
                            weights = build_weights(helpers_df, update_race_df, scheme=scheme)
                            team_rank = helpers_df['final_team_rank'].min()
                        else:
                            rider_ids = team_df['rider'].unique().tolist()
                            ratings = [cluster_team_rating.get(r, Rating(mu=25.0, sigma=8.333333)) for r in rider_ids]
                            weights = build_weights(team_df, update_race_df, scheme=scheme)
                            team_rank = team_df['final_team_rank'].min()
                        
                        sponsor_teams_list.append(ratings)
                        sponsor_team_weights.append(weights)
                        sponsor_team_ranks.append(team_rank)
                        sponsor_rider_map[sponsor] = rider_ids
                    
                    if sponsor_teams_list:  # Only update if there are teams
                        updated_teams = rate(
                                sponsor_teams_list,
                                ranks=sponsor_team_ranks,
                                weights=sponsor_team_weights)
                        
                        for team_idx, (sponsor, rider_ids) in enumerate(sponsor_rider_map.items()):
                            for i, rider in enumerate(rider_ids):
                                cluster_team_rating[rider] = updated_teams[team_idx][i]
                                last_update_team[cluster][rider] = current_race_date

                # Update last update date for this cluster
                last_update_by_cluster[cluster]['last_update'] = decision_date
                
        # Create a snapshot for each rider.
        # Instead of storing only the ratings for the current cluster,
        # we store the ratings for all clusters in a single record.
        snapshot_records = []
        for rider in riders_group['rider'].unique():
            team_obj = cluster_team_rating.get(rider, Rating(mu=25.0, sigma=8.333333))
            team_obj_gc = team_ratings['General Classification'].get(rider, Rating(mu=25.0, sigma=8.333333))
            record = {
                'race': race,
                'date': race_date,
                'classification': race_class,
                'rider': rider,
                'team': riders_group[riders_group['rider'] == rider]['team'].iloc[0],
                'rank_number': riders_group[riders_group['rider'] == rider]['rank_number'].iloc[0],
                'team_rank': riders_group[riders_group['rider'] == rider]['final_team_rank'].iloc[0],
                'cluster': cluster,  # the race's own cluster, if needed
                'race_cluster_teammate_mu': round(team_obj.mu, 2),
                'race_cluster_teammate_sigma': round(team_obj.sigma, 2),
                'gc_teammate_mu': round(team_obj_gc.mu, 2),
                'gc_teammate_sigma': round(team_obj_gc.sigma, 2)
            }
            last_update = last_update_team[cluster].get(rider, None)
            last_update_gc = last_update_team['General Classification'].get(rider, None)
            if last_update is None:
                record[f'race_cluster_last_update_teammate'] = np.nan
            else:
                record[f'race_cluster_last_update_teammate'] = (race_date - last_update).days
            if last_update_gc is None:
                record[f'gc_last_update_teammate'] = np.nan
            else:
                record[f'gc_last_update_teammate'] = (race_date - last_update_gc).days

            snapshot_records.append(record)
        
        # Append the DataFrame for the current race snapshot.
        race_ratings_snapshot_list.append(pd.DataFrame(snapshot_records))

        # Only update ratings if time_gap is None (real-time updates)
        if time_gap is None:
            #########################
            # TEAM UPDATES
            #########################
            # build sponsor-based teams
            sponsor_teams_list   = []      # list of lists of Rating objects
            sponsor_team_ranks   = []      # ordinal team rank in the race
            sponsor_team_weights = []      # parallel list of tuples of rider weights
            sponsor_rider_map    = {}      # team → list(riderId)   (to save order)
            total_participants = riders_group['rider'].nunique()

            for sponsor, team_df in riders_group.groupby('team'):
                # Filter out team leader if helpers_only is True
                if helpers_only and len(team_df) > 1:
                    # Find team leader (rider with lowest rank_number)
                    team_leader = team_df.loc[team_df['rank_number'].idxmin(), 'rider']
                    # Filter team to only include helpers
                    helpers_df = team_df[team_df['rider'] != team_leader].copy()
                    
                    # Skip teams with no helpers after filtering
                    if len(helpers_df) == 0:
                        continue
                        
                    # Use helpers data for weight calculation and rating updates
                    rider_ids = helpers_df['rider'].unique().tolist()
                    ratings = [cluster_team_rating.get(r, Rating(mu=25.0, sigma=8.333333)) for r in rider_ids]
                    weights = build_weights(helpers_df, riders_group, scheme=scheme)
                    team_rank = helpers_df['final_team_rank'].min()
                else:
                    # Original behavior: include all team members
                    rider_ids = team_df['rider'].unique().tolist()
                    ratings = [cluster_team_rating.get(r, Rating(mu=25.0, sigma=8.333333)) for r in rider_ids]
                    weights = build_weights(team_df, riders_group, scheme=scheme)
                    team_rank = team_df['final_team_rank'].min()
                
                # Add to lists for TrueSkill calculation
                sponsor_teams_list.append(ratings)
                sponsor_team_weights.append(weights)
                sponsor_team_ranks.append(team_rank)
                sponsor_rider_map[sponsor] = rider_ids
            
            # One multi‑entrant call, **with weights**
            if sponsor_teams_list:  # Only if there are teams to update
                updated_teams = rate(
                        sponsor_teams_list,
                        ranks   = sponsor_team_ranks,
                        weights = sponsor_team_weights)
                
                # Write new ratings back
                for team_idx, (sponsor, rider_ids) in enumerate(sponsor_rider_map.items()):
                    # Update all riders in the list (team leaders already filtered out if helpers_only=True)
                    for i, rider in enumerate(rider_ids):
                        cluster_team_rating[rider] = updated_teams[team_idx][i]
                        last_update_team[cluster][rider] = race_date

    # Combine all the race ratings snapshots into one DataFrame.
    ratings_snapshot_df = pd.concat(race_ratings_snapshot_list, ignore_index=True)
    return ratings_snapshot_df

def compute_team_perf_flattened_k_helpers(df, race_class=False, pref='teammate', skill_sources=['race_cluster', 'gc'], top_n=5):
    """
    Simplified K=6 approach for team performance features.
    
    Creates flattened roster features with 1 leader + 5 helpers (K=6 total):
    - 12 individual features per skill source: [μ_L, σ_L, μ_H1, σ_H1, μ_H2, σ_H2, μ_H3, σ_H3, μ_H4, σ_H4, μ_H5, σ_H5]
    - 3 global aggregates per skill source: mean μ, mean σ, mean μ/σ
    - NaN for missing slots (XGBoost handles this)
    
    Parameters:
      df (pd.DataFrame): DataFrame with columns including 'race', 'date', 'team', 'rider'
                         and skill columns: '{skill_source}_{pref}_mu', '{skill_source}_{pref}_sigma'
      race_class (bool): Whether to use race class specific features
      pref (str): The prefix for the performance columns ('teammate' or 'leader')
      skill_sources (list): List of skill sources to use (e.g., ['race_cluster', 'general_classification'])
      top_n (int): Number of top helpers to include (default: 5)
    
    Returns:
      pd.DataFrame: DataFrame with flattened K=6 roster features added for each rider
    """
    df = df.copy()
    
    def compute_flattened_features(group):
        result_rows = []
        
        for idx, row in group.iterrows():
            current_rider = row['rider']
            
            # Create new row with original data
            new_row = row.copy()
            
            # Process each skill source
            for skill_source in skill_sources:
                # Define column names based on skill source and preference
                if race_class:
                    skill_source += '_class'
                mu_col = f'{skill_source}_{pref}_mu'
                sigma_col = f'{skill_source}_{pref}_sigma'
                
                # Check if required columns exist
                if mu_col not in group.columns or sigma_col not in group.columns:
                    raise ValueError(f"Columns {mu_col} or {sigma_col} not found in DataFrame")
                
                # Get teammates (exclude current rider)
                teammates = group[group['rider'] != current_rider].copy()

                # Remove teammates with NaN values
                teammates = teammates.dropna(subset=[mu_col, sigma_col])
                
                # Sort teammates by mu (descending) to determine helper order
                if len(teammates) > 0:
                    teammates = teammates.sort_values(by=mu_col, ascending=False)
                
                role = 'helper' if pref == 'teammate' else 'leader'
                # Extract helpers (top 5 teammates)
                for i in range(1, top_n + 1):  # Helper indices 1-5
                    helper_idx = i - 1  # 0-based index for teammates iloc
                    if len(teammates) > helper_idx:
                        helper = teammates.iloc[helper_idx]
                        new_row[f'roster_{role}_{i}_mu_{skill_source}'] = round(helper[mu_col], 4)
                        new_row[f'roster_{role}_{i}_sigma_{skill_source}'] = round(helper[sigma_col], 4)
                    else:
                        # Missing slot - keep as NaN (XGBoost will handle this)
                        new_row[f'roster_{role}_{i}_mu_{skill_source}'] = np.nan
                        new_row[f'roster_{role}_{i}_sigma_{skill_source}'] = np.nan
                
                # Compute global aggregates across all team riders (including current rider)
                all_team_riders = group.dropna(subset=[mu_col, sigma_col])
                
                if len(all_team_riders) > 0:
                    mu_values = all_team_riders[mu_col].values
                    sigma_values = all_team_riders[sigma_col].values
                    
                    new_row[f'roster_mean_mu_{skill_source}'] = round(np.mean(mu_values), 4)
                    new_row[f'roster_mean_sigma_{skill_source}'] = round(np.mean(sigma_values), 4)
                    
                    # Mean μ/σ ratio
                    mu_sigma_ratios = mu_values / sigma_values
                    # Handle division by zero
                    mu_sigma_ratios = mu_sigma_ratios[np.isfinite(mu_sigma_ratios)]
                    if len(mu_sigma_ratios) > 0:
                        new_row[f'roster_mean_mu_sigma_ratio_{skill_source}'] = round(np.mean(mu_sigma_ratios), 4)
                    else:
                        new_row[f'roster_mean_mu_sigma_ratio_{skill_source}'] = np.nan
                else:
                    new_row[f'roster_mean_mu_{skill_source}'] = np.nan
                    new_row[f'roster_mean_sigma_{skill_source}'] = np.nan
                    new_row[f'roster_mean_mu_sigma_ratio_{skill_source}'] = np.nan
                
            # Add roster size (number of riders in team)
            if not race_class:
                new_row['roster_size'] = len(group)
            
            result_rows.append(new_row)
        
        return pd.DataFrame(result_rows)
    
    # Apply the function to each team group with progress bar
    skill_sources_str = ", ".join(skill_sources)
    tqdm.pandas(desc=f"Computing K={top_n} flattened roster features ({pref}) for: {skill_sources_str}")
    
    result_df = df.groupby(['race', 'date', 'team']).progress_apply(compute_flattened_features).reset_index(drop=True)
    
    return result_df

def run_trueskill_features_extraction_pipeline(race_class, scheme, time_gap=None, target_years=None):
    """
    Run TrueSkill features extraction pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT')
        scheme (str): Scheme name ('time_lag', 'equal_weight', 'rank_norm', 'leader', 'baseline')
        time_gap (int or str, optional): Time gap in days for decision points, or 'season_start' for season-based decisions
        target_years (list, optional): List of years to extract features for (e.g., [2024, 2025]).
                            If None, extracts for all years in the data.
                            Features will still be calculated using ALL historical data.
    Returns:
        str: Path to created TrueSkill features file
    """

    print(f"=" * 80)
    print(f"{'TRUESKILL FEATURES EXTRACTION PIPELINE':^80}")
    print(f"=" * 80)
    print(f"Race class: {race_class}")
    print(f"Scheme: {scheme}")
    print(f"Time gap: {time_gap}")
    if target_years is not None:
        print(f"Target years: {target_years}")
    # Determine output path based on scheme
    if scheme == 'leader':
        output_path = get_data_dir('leader_power')
    else:
        output_path = get_data_dir('team_power')
    
    if 'alone' in output_path:
        helpers_only = True
    else:
        helpers_only = False

    if time_gap == 365:
        time_gap = 'season_start'

    previous_trueSkill_rating_df = None
    if target_years is not None:
        previous_trueSkill_rating_path = get_previous_trueSkill_rating(race_class, scheme, time_gap, leader=scheme == 'leader')
        previous_trueSkill_rating_df = pd.read_csv(previous_trueSkill_rating_path)
        
    print(f"Output path: {output_path}")
    print(f"Helpers only: {helpers_only}")
    print(f"\nStep 1: Loading and Preprocessing Data")
    print("-" * 50)
    
    try:
        # Get data paths from config
        riders_results_path = get_data_dir('riders_race_results')
        print(f"Loading riders race results from {riders_results_path}")
        
        # Load riders race results
        riders_race_results = pd.read_csv(riders_results_path)
        # riders_race_results = riders_race_results.tail(2000)
        riders_race_results['date'] = pd.to_datetime(riders_race_results['date'])
        riders_race_results['year'] = riders_race_results['date'].dt.year
        print(f"✓ Loaded {len(riders_race_results)} records")

        riders_race_results['classification'] = riders_race_results['classification'].astype(str)
        riders_race_results['race_class'] = riders_race_results['race_class'].astype(str)
        # Apply race class filter
        if race_class == 'WT':
            riders_race_results = riders_race_results[
                riders_race_results['classification'].isin(['1.UWT', '2.UWT'])
            ].copy()
            print(f"✓ After WT filter: {len(riders_race_results)} records")
        
        print(f'Dont include Prologue races')
        riders_race_results = riders_race_results[~riders_race_results['race'].str.contains('Prologue')].copy()
        print(f'After Prologue filter: {len(riders_race_results)} records')
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

    try:        
        # Run appropriate TrueSkill function based on scheme
        if scheme == 'leader':
            print(f"Running TrueSkill extraction with leader scheme, with {len(riders_race_results)} records")
            ranking_by_race_class_induv = {'induv_rating': defaultdict(dict), 'induv_last_update': defaultdict(dict)}
            # For leader skille estimation don't include DNF and DF results (KEEP OTL)
            true_skill_df = create_leader_data_set(
                data_set=riders_race_results, 
                ranking_by_race_class_induv=ranking_by_race_class_induv, 
                race_class=race_class,
                time_gap=time_gap,
                previous_snapshot_df=previous_trueSkill_rating_df
            )
            # if race_class == 'all':
            #     true_skill_df_class = create_leader_data_set_by_cluster_class(riders_race_results, race_class, time_gap=time_gap)

        else:
            print(f"Running TrueSkill extraction with {scheme} scheme, with {len(riders_race_results)} records")
            ranking_by_race_class_team = {'teammate_rating': defaultdict(dict), 'teammate_last_update': defaultdict(dict)}
            true_skill_df = create_data_set_team_weights(
                data_set=riders_race_results, 
                ranking_by_race_class_team=ranking_by_race_class_team, 
                race_class=race_class, 
                scheme=scheme,
                helpers_only=helpers_only,
                time_gap=time_gap,
                previous_snapshot_df=previous_trueSkill_rating_df
            )
            # if race_class == 'all':
            #     true_skill_df_class = create_data_set_team_weights_by_cluster_class(
            #         data_set=riders_race_results, 
            #         race_class=race_class, 
            #         scheme=scheme,
            #         helpers_only=helpers_only,
            #         time_gap=time_gap
            #     )

        
        print(f"✓ TrueSkill extraction completed: {len(true_skill_df)} records")
        
    except Exception as e:
        print(f"❌ Error in TrueSkill extraction: {e}")
        raise
    
    print(f"\nStep 4: Computing Team Performance Features")
    print("-" * 50)
    merge_on = ['race', 'date', 'rider', 'cluster', 'team','classification','rank_number', 'team_rank']

    try:
        # Compute team performance features for non-leader schemes
        if scheme != 'leader':
            print("Computing team performance features...")
            final_true_skill = compute_team_perf_flattened_k_helpers(true_skill_df)
            # if race_class == 'all':
            #     final_true_skill_class = compute_team_perf_flattened_k_helpers(true_skill_df_class, race_class=True)

            #     final_true_skill_class['classification'] = final_true_skill_class['classification'].astype(str)
            #     final_true_skill['classification'] = final_true_skill['classification'].astype(str)
            #     print(f"✓ Team performance features computed with shapes: {final_true_skill_class.shape} and {final_true_skill.shape}")
            #     # Merge the two dataframes
            #     final_true_skill = final_true_skill_class.merge(final_true_skill, on=merge_on, how='inner')

        else:
            final_true_skill = true_skill_df.copy()
            # if race_class == 'all':
            #     final_true_skill_class = true_skill_df_class.copy()
            #     final_true_skill_class['classification'] = final_true_skill_class['classification'].astype(str)
            #     final_true_skill['classification'] = final_true_skill['classification'].astype(str)

            #     print(f"✓ Leader performance features computed with shapes: {final_true_skill_class.shape} and {final_true_skill.shape}")
            #     # Merge the two dataframes
            #     final_true_skill = final_true_skill_class.merge(final_true_skill[merge_on + ALL_RAW_LEADER_FEATURES], on=merge_on, how='inner')

    except Exception as e:
        print(f"❌ Error computing team performance features: {e}")
        raise
    
    print(f"✓ Final dataset shape: {final_true_skill.shape}")
    print(f"✓ Features extracted: {len(final_true_skill.columns)} columns")
    print(f"\nStep 5: Saving Results")
    print("-" * 50)
    
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        if time_gap is not None:
            if target_years is not None:
                year_str = f"{'_'.join(map(str, sorted(target_years)))}"
                file_path = os.path.join(output_path, f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme_{year_str}_years_{time_gap}_tg.csv')
            else:
                file_path = os.path.join(output_path, f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme_{time_gap}_tg.csv')
        else:
            if target_years is not None:
                year_str = f"{'_'.join(map(str, sorted(target_years)))}"
                file_path = os.path.join(output_path, f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme_{year_str}_years.csv')
            else:
                file_path = os.path.join(output_path, f'trueSkill_features_class_{race_class}_ratings_snapshot_{scheme}_scheme.csv')
            

        # Sort data consistently
        final_true_skill.sort_values(by=['race','date','team','rank_number'], inplace=True)
        
        # Save the features
        final_true_skill.to_csv(file_path, index=False)
        print(f"✓ TrueSkill features saved to: {file_path}")
        print(f"✓ Final dataset shape: {final_true_skill.shape}")
        
        return file_path
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise

