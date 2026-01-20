import pandas as pd
import numpy as np
from trueskill import Rating, rate, setup
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from datetime import timedelta
from scipy.stats import linregress
import os
from datetime import timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from roster_ranker.utils import *
from roster_ranker.data import extract_race_features

# --------------------------
# 0. Rider Unique ID Management
# --------------------------
def add_rider_unique_id(df):
    """
    Adds a unique identifier for riders with duplicate names.
    Uses age progression across years to distinguish different riders with the same name.
    Only processes riders with duplicate names to avoid unnecessary work.
    
    Args:
        df: DataFrame with 'rider', 'year', and 'age' columns
        
    Returns:
        DataFrame with added 'rider_unique_id' column
    """
    print("Adding unique rider identifiers for duplicate names...")
    
    df = df.copy()
    
    # Find riders with duplicate names by checking for same name + different ages in the same year
    # This indicates there are actually multiple distinct riders with the same name
    duplicate_riders = set()
    
    # Group by rider name and year, check if there are multiple different ages
    rider_year_age = df[['rider', 'year', 'age']].drop_duplicates()
    for rider_name, group in rider_year_age.groupby('rider'):
        # Check each year for this rider
        for year, year_group in group.groupby('year'):
            unique_ages = year_group['age'].dropna().unique()
            if len(unique_ages) > 1 and abs(unique_ages[0] - unique_ages[1]) > 1:
                # Found same name with different ages in the same year
                duplicate_riders.add(rider_name)
                break
    
    duplicate_riders = list(duplicate_riders)
    
    if len(duplicate_riders) == 0:
        print("✓ No duplicate rider names found")
        df['rider_unique_id'] = df['rider']
        return df
    
    print(f"Found {len(duplicate_riders)} rider names with actual duplicates")
    
    # Initialize rider_unique_id with original rider names
    df['rider_unique_id'] = df['rider']
    
    # Process each duplicate rider name
    for rider_name in tqdm(duplicate_riders, desc="Processing duplicate names"):
        rider_data = df[df['rider'] == rider_name].copy()
        
        # Get unique age-year combinations sorted by year
        age_year_data = rider_data[['age', 'year']].drop_duplicates().sort_values(['year', 'age'])
        
        # Build a mapping of age progressions to identify distinct riders
        rider_instances = {}  # Maps (year, age) -> instance_id
        age_progressions = defaultdict(list)  # instance_id -> list of (year, age) tuples
        instance_counter = 0
        
        for _, row in age_year_data.iterrows():
            year = row['year']
            age = row['age']
            
            # Try to match this (year, age) to an existing rider instance
            matched = False
            for instance_id, progression in age_progressions.items():
                # Check if this age fits the progression
                # Age should increase by 1 for each year difference
                valid_for_instance = True
                for prev_year, prev_age in progression:
                    expected_age_diff = year - prev_year
                    actual_age_diff = age - prev_age
                    
                    # Allow some tolerance (±1 year) for birthdays
                    if abs(actual_age_diff - expected_age_diff) > 1:
                        valid_for_instance = False
                        break
                
                if valid_for_instance:
                    # This (year, age) fits this rider instance
                    rider_instances[(year, age)] = instance_id
                    age_progressions[instance_id].append((year, age))
                    matched = True
                    break
            
            if not matched:
                # Create a new rider instance
                rider_instances[(year, age)] = instance_counter
                age_progressions[instance_counter].append((year, age))
                instance_counter += 1
        
        # Assign unique IDs to each row based on year and age
        # This assigns the same ID to the same rider across all years
        for idx, row in rider_data.iterrows():
            year = row['year']
            age = row['age']
            if (year, age) in rider_instances:
                instance_id = rider_instances[(year, age)]
                df.loc[idx, 'rider_unique_id'] = f"{rider_name}_ID{instance_id}"
            else:
                # Fallback: assign to instance 0
                df.loc[idx, 'rider_unique_id'] = f"{rider_name}_ID0"
        
        print(f"  - {rider_name}: Split into {len(age_progressions)} distinct riders")
    
    print(f"✓ Unique rider IDs created")
    return df

def remove_rider_unique_id(df):
    """
    Removes the temporary rider_unique_id column and restores original rider names.
    
    Args:
        df: DataFrame with 'rider_unique_id' column
        
    Returns:
        DataFrame with 'rider_unique_id' column removed
    """
    print("Removing temporary unique rider IDs...")
    if 'rider_unique_id' in df.columns:
        df = df.drop('rider_unique_id', axis=1)
    print("✓ Temporary IDs removed")
    return df

def race_class_to_category(race_class):
    # Returns 3 if 'WT' in race_class, 2 if 'PRO' in race_class, else 1.
    if 'WT' in race_class:
        return 'WT'
    elif 'Pro' in race_class:
        return 'Pro'
    elif race_class == '1.1' or race_class == '1.2':
        return '1'
    else:
        return '2'
    
# --------------------------
# 1. Preprocessing & Date Conversion
# --------------------------
def preprocess_dates(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# --------------------------
# 2. Career and Demographic Features
# --------------------------
def add_career_features(df):
    first_race = df.groupby('rider_unique_id')['date'].min().reset_index().rename(columns={'date': 'FirstRaceDate'})
    df = pd.merge(df, first_race, on='rider_unique_id', how='left')
    df['CareerLength'] = (df['decision_date'] - df['FirstRaceDate']).dt.days / 365.25
    return df

# --------------------------
# 3. Race Type Indicator (OneDayRace vs. StageRace)
# --------------------------
def add_race_type(df):
    df['RaceType'] = df['race'].apply(lambda x: 'StageRace' if 'Stage' in x else 'OneDayRace')
    df['race_class'] = df['race_class'].astype(str)
    return df

# --------------------------
# 6. Build Lookup for Historical Cluster-Based Features
# --------------------------
def build_lookup_df(df):
    df['rider'] = df['rider'].str.strip()
    lookup = df[['rider_unique_id', 'date', 'cluster', 'rank_number', 'race_class']].dropna(subset=['date', 'rank_number'])
    return lookup

# --------------------------
# 7. COMBINED EFFICIENT HISTORICAL FEATURES
# --------------------------
def add_combined_historical_features(df, race_class=False, decision_date_col=None):
    """
    Efficiently compute cluster-based, form, and best result features in a single pass.
    This replaces the three separate inefficient functions.
    
    Args:
        df: DataFrame with race data
        race_class: Whether to use race class filtering
        decision_date_col: Column name containing decision dates for historical filtering.
                          If None, uses race date (no time gap constraint).
    """
    print("Computing combined historical features...")
    
    # Prepare the dataframe
    df = df.copy()
    df = df.sort_values(['rider', 'date']).reset_index(drop=True)
    
    # Initialize result columns
    cluster_cols = [
        'Previous Result Race Cluster', 'Result 2 Races Ago Cluster', 'Result 3 Races Ago Cluster',
        'Result 4 Races Ago Cluster', 'Result 5 Races Ago Cluster', 'Result 6 Races Ago Cluster',
        'Result 7 Races Ago Cluster', 'Result 8 Races Ago Cluster', 'Result 9 Races Ago Cluster',
        'Result 10 Races Ago Cluster'
    ]
    
    for col in cluster_cols:
        df[col] = np.nan
    
    df['form'] = 0.0
    df['best_result_cluster'] = np.nan
    df['time_since_best_result_cluster'] = np.nan
    df['best_result_cluster_class'] = np.nan
    df['time_since_best_result_cluster_class'] = np.nan
    df['best_result_class'] = np.nan
    df['time_since_best_result_class'] = np.nan
    df['best_result_gc'] = np.nan
    df['time_since_best_result_gc'] = np.nan
    
    def process_rider_group(rider_group):
        """Process all historical features for a single rider's data"""
        rider_group = rider_group.sort_values('date').reset_index(drop=True)
        n_races = len(rider_group)
        
        # For each race, compute features based on all previous races
        for i in range(n_races):
            current_race = rider_group.iloc[i]
            
            # Skip if this is not a target record
            if not current_race.get('_is_target', True):
                continue
            
            current_date = current_race['date']
            current_cluster = current_race['cluster']
            current_race_class = current_race['race_class']
            
            # Determine the cutoff date for historical data
            if decision_date_col is not None:
                max_historical_date = current_race[decision_date_col]
                current_date = max_historical_date
            else:
                max_historical_date = current_date
            
            # Get all previous races for this rider up to decision date
            prev_races = rider_group[
                rider_group['date'] <= max_historical_date
            ]
            
            if len(prev_races) == 0:
                continue
                
            # 1. CLUSTER-BASED FEATURES
            # Get previous races in same cluster
            cluster_races = prev_races[prev_races['cluster'] == current_cluster]
            
            if race_class and len(cluster_races) > 0:
                # Try with race_class filter first
                cluster_class_races = cluster_races[cluster_races['race_class'] == current_race_class]
                if len(cluster_class_races) >= 5:
                    cluster_races = cluster_class_races
            
            # Get last 10 results in cluster
            if len(cluster_races) > 0:
                recent_cluster_results = cluster_races.tail(10)['rank_number'].values
                # Fill the cluster result columns
                for j, col in enumerate(cluster_cols):
                    if j < len(recent_cluster_results):
                        rider_group.loc[i, col] = recent_cluster_results[-(j+1)]
            
            # 2. FORM FEATURE (42-day window)
            form_window = timedelta(days=42)
            form_mask = (prev_races['date'] >= current_date - form_window)
            form_races = prev_races[form_mask]
            rider_group.loc[i, 'form'] = form_races['rider_race_points'].sum()
            
            # 3. BEST RESULT FEATURES
            # Best result in same cluster
            cluster_races = prev_races[prev_races['cluster'] == current_cluster]
            if len(cluster_races) > 0:
                best_rank = cluster_races['rank_number'].min()
                rider_group.loc[i, 'best_result_cluster'] = best_rank
                best_race_date = cluster_races[cluster_races['rank_number'] == best_rank]['date'].max()
                rider_group.loc[i, 'time_since_best_result_cluster'] = (current_date - best_race_date).days
            
            # Best result in same cluster + same class
            cluster_class_races = prev_races[
                (prev_races['cluster'] == current_cluster) & 
                (prev_races['race_class'] == current_race_class)
            ]
            if len(cluster_class_races) > 0:
                best_rank = cluster_class_races['rank_number'].min()
                rider_group.loc[i, 'best_result_cluster_class'] = best_rank
                best_race_date = cluster_class_races[cluster_class_races['rank_number'] == best_rank]['date'].max()
                rider_group.loc[i, 'time_since_best_result_cluster_class'] = (current_date - best_race_date).days
            
            # Best result in same class
            class_races = prev_races[prev_races['race_class'] == current_race_class]
            if len(class_races) > 0:
                best_rank = class_races['rank_number'].min()
                rider_group.loc[i, 'best_result_class'] = best_rank
                best_race_date = class_races[class_races['rank_number'] == best_rank]['date'].max()
                rider_group.loc[i, 'time_since_best_result_class'] = (current_date - best_race_date).days
            
            # Best result in GC races
            gc_races = prev_races[
                (prev_races['cluster'] == 'General Classification') & 
                (~prev_races['rank'].isin(['DNF', 'DNS', 'OTL']))  # Check actual rank status
            ]
            if len(gc_races) > 0:
                best_rank = gc_races['rank_number'].min()  # rank_number already handles non-finishers as max rank
                rider_group.loc[i, 'best_result_gc'] = best_rank
                best_race_date = gc_races[gc_races['rank_number'] == best_rank]['date'].max()
                rider_group.loc[i, 'time_since_best_result_gc'] = (current_date - best_race_date).days
        
        return rider_group
    
    # Process all riders with progress bar
    tqdm.pandas(desc="Processing riders")
    result_df = df.groupby('rider_unique_id').progress_apply(process_rider_group).reset_index(drop=True)
    
    return result_df

# --------------------------
# 7. Cluster-Based Historical Finishing Positions (DEPRECATED - use add_combined_historical_features)
# --------------------------
def get_10_previous_resultsCLUSTER(rider_unique_id, race_date, cluster, lookup_df, race_class=None):
    try:
        if race_class:
            # First try with race_class filter
            past_results = lookup_df[(lookup_df['rider_unique_id'] == rider_unique_id) &
                                   (lookup_df['date'] < race_date) &
                                   (lookup_df['cluster'] == cluster) &
                                   (lookup_df['race_class'] == race_class)]
            
            # If we have race_class but got less than 5 results, fallback to without race_class
            if past_results.shape[0] < 5:
                past_results = lookup_df[(lookup_df['rider_unique_id'] == rider_unique_id) &
                                       (lookup_df['date'] < race_date) &
                                       (lookup_df['cluster'] == cluster)]
        else:
            # No race_class provided, search without it
            past_results = lookup_df[(lookup_df['rider_unique_id'] == rider_unique_id) &
                                   (lookup_df['date'] < race_date) &
                                   (lookup_df['cluster'] == cluster)]
            
        if past_results.shape[0] > 10:
            luik = past_results.sort_values(by='date', ascending=False)['rank_number'].head(10)
            return luik.reset_index(drop=True)
        else:
            luik = past_results.sort_values(by='date', ascending=False)['rank_number']
            return luik.reset_index(drop=True)
    except Exception as e:
        print(rider_unique_id, race_date, cluster, e)
        return pd.Series([np.nan]*10)

def add_cluster_based_features(df, lookup_df, race_class=False):
    """DEPRECATED: Use add_combined_historical_features() instead for better performance"""
    cluster_features = []
    for idx, row in tqdm(df.iterrows(), desc="Adding cluster-based features", total=df.shape[0]):
        rider_unique_id = row['rider_unique_id']
        race_date = row['date']
        cluster = row['cluster']
        if race_class:
            race_class_value = row['race_class']
            past_series = get_10_previous_resultsCLUSTER(rider_unique_id, race_date, cluster, lookup_df, race_class_value)
        else:
            past_series = get_10_previous_resultsCLUSTER(rider_unique_id, race_date, cluster, lookup_df)
        if past_series.shape[0] < 10:
            past_series = past_series.reindex(range(10), fill_value=np.nan)
        # If you want to adjust the finishing positions (e.g., adding 1), do it here:
        # past_series = past_series + 1
        cluster_features.append(past_series.values)
    col_names = [
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
    cluster_df = pd.DataFrame(cluster_features, index=df.index, columns=col_names)
    df = pd.concat([df.reset_index(drop=True), cluster_df.reset_index(drop=True)], axis=1)
    return df

# --------------------------
# 8. Recent Form Feature (42-day window)
# --------------------------
def add_form_feature(df):
    """DEPRECATED: Use add_combined_historical_features() instead for better performance"""
    form_values = []
    for idx, row in tqdm(df.iterrows(), desc="Adding form feature", total=df.shape[0]):
        rider_unique_id = row['rider_unique_id']
        race_date = row['date']
        window = timedelta(days=42)
        mask = (df['rider_unique_id'] == rider_unique_id) & (df['date'] < race_date) & (df['date'] >= race_date - window)
        form_values.append(df.loc[mask, 'rider_race_points'].sum())
    df['form'] = form_values
    return df

# --------------------------
# 9. Best Result and Time Since Best Result
# --------------------------
def add_best_result_features(df, race_class=False):
    """DEPRECATED: Use add_combined_historical_features() instead for better performance"""
    best_results_cluster = []
    best_results_class_cluster = []
    time_since_best_cluster = []
    time_since_best_class_cluster = []
    best_results_class = []
    time_since_best_class = []

    for idx, row in tqdm(df.iterrows(), desc="Adding best result features", total=df.shape[0]):
        # Extract relevant information from the row
        rider_unique_id = row['rider_unique_id']
        race_date = row['date']
        race_cluster = row['cluster']
        race_class = row['race_class']
        past_cluster = df[(df['rider_unique_id'] == rider_unique_id) & (df['date'] < race_date) & (df['cluster'] == race_cluster)]
        past_class_cluster = df[(df['rider_unique_id'] == rider_unique_id) & (df['date'] < race_date) & (df['cluster'] == race_cluster) & (df['race_class'] == race_class)]
        past_class = df[(df['rider_unique_id'] == rider_unique_id) & (df['date'] < race_date) & (df['race_class'] == race_class)]
        
        if past_cluster.empty:
            best_results_cluster.append(np.nan)
            time_since_best_cluster.append(np.nan)
        else:
            best_type = past_cluster['rank_number'].min()
            best_results_cluster.append(best_type)
            best_race_date = past_cluster[past_cluster['rank_number'] == best_type]['date'].max()
            time_since_best_cluster.append((race_date - best_race_date).days / 365.25)
        if past_class_cluster.empty:
            best_results_class_cluster.append(np.nan)
            time_since_best_class_cluster.append(np.nan)
        else:
            best_class = past_class_cluster['rank_number'].min()
            best_results_class_cluster.append(best_class)
            best_race_date = past_class_cluster[past_class_cluster['rank_number'] == best_class]['date'].max()
            time_since_best_class_cluster.append((race_date - best_race_date).days / 365.25)
        if past_class.empty:
            best_results_class.append(np.nan)
            time_since_best_class.append(np.nan)
        else:
            best_type = past_class['rank_number'].min()
            best_results_class.append(best_type)
            best_race_date = past_class[past_class['rank_number'] == best_type]['date'].max()
            time_since_best_class.append((race_date - best_race_date).days / 365.25)
       
    df['best_result_cluster_class'] = best_results_class_cluster
    df['time_since_best_result_cluster_class'] = time_since_best_class_cluster
    df['best_result_cluster'] = best_results_cluster
    df['time_since_best_result_cluster'] = time_since_best_cluster
    df['best_result_class'] = best_results_class
    df['time_since_best_result_class'] = time_since_best_class

    return df

# --------------------------
# 10. Points History Features
# --------------------------
def add_points_features(df):
    df['year'] = df['year'].astype(int)
    # Compute yearly total points per rider
    yearly_points = df.groupby(['rider_unique_id', 'year'])['rider_race_points'].sum().reset_index().rename(columns={'rider_race_points': 'yearly_points'})
    df = pd.merge(df, yearly_points, on=['rider_unique_id', 'year'], how='left')
    return df, yearly_points

# --------------------------
# 11. Rider Evolution (Past 3 years points and CareerSlope)
# --------------------------
def points_past_3years(rider_unique_id, current_year, yearly_points):
    # Get all past years for this rider
    past = yearly_points[(yearly_points['rider_unique_id'] == rider_unique_id) & (yearly_points['year'] < current_year)]
    if past.empty:
        return (0, 0, 0, 0)
    past = past.sort_values(by='year', ascending=False)
    # Extract points for up to three years
    pts = past['yearly_points'].tolist()
    # Fill missing values if less than 3
    pts = pts[:3] if len(pts) >= 3 else pts + [0]*(3 - len(pts))
    # Mean over all available past years
    pts_mean = np.mean(pts)
    return (pts_mean, pts[0], pts[1], pts[2])

def apply_linregress(values):
    # Expect values to be a list/array of three numbers
    if np.all(np.array(values) == 0):
        return 0
    slope, intercept, r_value, p_value, std_err = linregress([1, 2, 3], values)
    return slope

def rider_evolution(df, yearly_points):
    try:
        evol = df.apply(lambda row: points_past_3years(row['rider_unique_id'], row['year'], yearly_points), axis=1)
        evol_df = pd.DataFrame(evol.tolist(), columns=['PointsPastSeasons', 'pointsMinus1', 'pointsMinus2', 'pointsMinus3'])
        # Calculate mean of past seasons points
        evol_df[['PointsPastSeasons', 'pointsMinus1', 'pointsMinus2', 'pointsMinus3']] = evol_df[['PointsPastSeasons', 'pointsMinus1', 'pointsMinus2', 'pointsMinus3']].fillna(0)
        evol_df['CareerSlope'] = evol_df[['pointsMinus3', 'pointsMinus2', 'pointsMinus1']].apply(apply_linregress, axis=1)
        df = pd.concat([df.reset_index(drop=True), evol_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print("Error in rider_evolution:", e)
    return df

def add_year_to_date_points_with_decision_date(df):
    """
    Adds year-to-date points features with decision date awareness:
      - PointsStartYear: Year-to-date points (all races)
      - PointsStartYearRaceType: Year-to-date points by race type (OneDayRace/StageRace)
      - PointsStartYearRaceClass: Year-to-date points by race class (WT/Pro/1/2)
      - PointsStartYearCluster: Year-to-date points by race cluster
      - PointsStartYearGC: Year-to-date points in GC races
    """
    print("Computing year-to-date points with decision date logic...")
    
    # Initialize all columns with zeros
    df['PointsStartYear'] = 0.0
    df['PointsStartYearRaceType'] = 0.0
    df['PointsStartYearRaceClass'] = 0.0
    df['PointsStartYearCluster'] = 0.0
    df['PointsStartYearGC'] = 0.0
    
    # Sort by rider, year, date for proper processing
    df_sorted = df.sort_values(['rider_unique_id', 'year', 'date']).copy()
    
    # Process each rider combination
    for rider_unique_id, group in tqdm(df_sorted.groupby('rider_unique_id'), desc="Computing YTD points"):
        for year, year_group in group.groupby('year'):
            # Sort races within the year by date
            year_races = year_group.sort_values('date').copy()
            
            for idx, current_race in year_races.iterrows():
                # Skip if this is not a target record
                if not current_race.get('_is_target', True):
                    continue
                
                current_date = current_race['date']
                decision_date = current_race.get('decision_date', current_date)
                
                # Get all races for this rider in the same year up to decision date
                ytd_races = year_races[
                    (year_races['date'] <= decision_date)
                ]
                
                if len(ytd_races) > 0:
                    # Total YTD points
                    df.loc[idx, 'PointsStartYear'] = ytd_races['rider_race_points'].sum()
                    
                    # YTD points by race type
                    race_type_points = ytd_races[ytd_races['RaceType'] == current_race['RaceType']]['rider_race_points'].sum()
                    df.loc[idx, 'PointsStartYearRaceType'] = race_type_points
                    
                    # YTD points by race class
                    race_class_points = ytd_races[ytd_races['race_class'] == current_race['race_class']]['rider_race_points'].sum()
                    df.loc[idx, 'PointsStartYearRaceClass'] = race_class_points
                    
                    # YTD points by cluster
                    cluster_points = ytd_races[ytd_races['cluster'] == current_race['cluster']]['rider_race_points'].sum()
                    df.loc[idx, 'PointsStartYearCluster'] = cluster_points
                    
                    # YTD points for GC races
                    if current_race['cluster'] == 'General Classification':
                        gc_points = ytd_races[ytd_races['cluster'] == 'General Classification']['rider_race_points'].sum()
                        df.loc[idx, 'PointsStartYearGC'] = gc_points
    
    return df

def add_points_race_type_and_class(df, time_gap=False):
    """
    Adds features based on race type, race class, cluster and GC:
      - PointsStartYear*: Year-to-date points (handled by separate function with decision dates)
      - PointsPastSeasonsRaceType: Historical average points by race type
      - PointsPastSeasonsRaceClass: Historical average points by race class
      - PointsPastSeasonsCluster: Historical average points by cluster
      - PointsPastSeasonsGC: Historical average points in GC races
    """
    print("Computing historical points averages (past seasons only)...")
    
    # Sort dataframe for efficient processing
    df_sorted = df.sort_values(['rider_unique_id', 'year', 'date']).copy()
    
    # Pre-compute yearly totals for each race type (for historical averages)
    yearly_points_one_day = df[df['RaceType'] == 'OneDayRace'] \
        .groupby(['rider_unique_id', 'year'])['rider_race_points'].sum() \
        .reset_index().rename(columns={'rider_race_points': 'yearly_points_oneday'})
    
    yearly_points_stage = df[df['RaceType'] == 'StageRace'] \
        .groupby(['rider_unique_id', 'year'])['rider_race_points'].sum() \
        .reset_index().rename(columns={'rider_race_points': 'yearly_points_stage'})
    
    # Pre-compute yearly totals for each race class (for historical averages)
    yearly_points_by_class = df.groupby(['rider_unique_id', 'year', 'race_class'])['rider_race_points'].sum() \
        .reset_index().rename(columns={'rider_race_points': 'yearly_points_class'})
    
    # Pre-compute yearly totals for each cluster (for historical averages)
    yearly_points_by_cluster = df.groupby(['rider_unique_id', 'year', 'cluster'])['rider_race_points'].sum() \
        .reset_index().rename(columns={'rider_race_points': 'yearly_points_cluster'})
    
    # Pre-compute yearly totals for GC races (for historical averages)
    yearly_points_gc = df[df['cluster'] == 'General Classification'] \
        .groupby(['rider_unique_id', 'year'])['rider_race_points'].sum() \
        .reset_index().rename(columns={'rider_race_points': 'yearly_points_gc'})
    
    if not time_gap:
        print("Computing year-to-date points...")
        # Compute cumulative points within each year for each rider
        df_sorted['PointsStartYear'] = df_sorted.groupby(['rider_unique_id', 'year'])['rider_race_points'].cumsum() - df_sorted['rider_race_points']
        
        # Compute cumulative points by race type within each year for each rider
        df_sorted['PointsStartYearRaceType'] = df_sorted.groupby(['rider_unique_id', 'year', 'RaceType'])['rider_race_points'].cumsum() - df_sorted['rider_race_points']
        
        # Compute cumulative points by race class within each year for each rider
        df_sorted['PointsStartYearRaceClass'] = df_sorted.groupby(['rider_unique_id', 'year', 'race_class'])['rider_race_points'].cumsum() - df_sorted['rider_race_points']
        
        # Compute cumulative points by cluster within each year for each rider
        df_sorted['PointsStartYearCluster'] = df_sorted.groupby(['rider_unique_id', 'year', 'cluster'])['rider_race_points'].cumsum() - df_sorted['rider_race_points']
        
        # Compute cumulative points for GC races within each year for each rider
        gc_mask = df_sorted['cluster'] == 'General Classification'
        df_sorted['PointsStartYearGC'] = 0.0  # Initialize with zeros
        df_sorted.loc[gc_mask, 'PointsStartYearGC'] = df_sorted[gc_mask].groupby(['rider_unique_id', 'year'])['rider_race_points'].cumsum() - df_sorted[gc_mask]['rider_race_points']
        
    print("Computing historical averages...")
    # For historical averages, we need to compute the mean of past years for each race type, class, and cluster
    def compute_historical_avg(group):
        group = group.sort_values('year')
        rider_unique_id = group['rider_unique_id'].iloc[0]
        
        # Get yearly points for this rider by race type
        rider_yearly_oneday = yearly_points_one_day[yearly_points_one_day['rider_unique_id'] == rider_unique_id]
        rider_yearly_stage = yearly_points_stage[yearly_points_stage['rider_unique_id'] == rider_unique_id]
        
        # Get yearly points for this rider by race class
        rider_yearly_by_class = yearly_points_by_class[yearly_points_by_class['rider_unique_id'] == rider_unique_id]
        
        # Get yearly points for this rider by cluster
        rider_yearly_by_cluster = yearly_points_by_cluster[yearly_points_by_cluster['rider_unique_id'] == rider_unique_id]
        
        # Get yearly points for this rider in GC races
        rider_yearly_gc = yearly_points_gc[yearly_points_gc['rider_unique_id'] == rider_unique_id]
        
        # Initialize result lists
        race_type_results = []
        race_class_results = []
        cluster_results = []
        gc_results = []
        indices = []
        
        for idx, row in group.iterrows():
            # Skip if this is not a target record
            if not row.get('_is_target', True):
                continue
            
            current_year = row['year']
            race_type = row['RaceType']
            race_class = row['race_class']
            cluster = row['cluster']
            
            # Historical average for race type
            if race_type == 'OneDayRace':
                past_data = rider_yearly_oneday[rider_yearly_oneday['year'] < current_year]
                avg_race_type = past_data['yearly_points_oneday'].mean() if len(past_data) > 0 else 0
            else:  # StageRace
                past_data = rider_yearly_stage[rider_yearly_stage['year'] < current_year]
                avg_race_type = past_data['yearly_points_stage'].mean() if len(past_data) > 0 else 0
            
            race_type_results.append(avg_race_type)
            
            # Historical average for race class
            past_class_data = rider_yearly_by_class[(rider_yearly_by_class['year'] < current_year) & (rider_yearly_by_class['race_class'] == race_class)]
            avg_race_class = past_class_data['yearly_points_class'].mean() if len(past_class_data) > 0 else 0
            race_class_results.append(avg_race_class)
            
            # Historical average for cluster
            past_cluster_data = rider_yearly_by_cluster[(rider_yearly_by_cluster['year'] < current_year) & (rider_yearly_by_cluster['cluster'] == cluster)]
            avg_cluster = past_cluster_data['yearly_points_cluster'].mean() if len(past_cluster_data) > 0 else 0
            cluster_results.append(avg_cluster)
            
            # Historical average for GC races
            past_gc_data = rider_yearly_gc[rider_yearly_gc['year'] < current_year]
            avg_gc = past_gc_data['yearly_points_gc'].mean() if len(past_gc_data) > 0 else 0
            gc_results.append(avg_gc)
            
            indices.append(idx)
        
        # Return all features as a DataFrame (only for target records)
        if len(indices) > 0:
            result_df = pd.DataFrame({
                'PointsPastSeasonsRaceType': race_type_results,
                'PointsPastSeasonsRaceClass': race_class_results,
                'PointsPastSeasonsCluster': cluster_results,
                'PointsPastSeasonsGC': gc_results
            }, index=indices)
        else:
            # No target records in this group
            result_df = pd.DataFrame({
                'PointsPastSeasonsRaceType': [],
                'PointsPastSeasonsRaceClass': [],
                'PointsPastSeasonsCluster': [],
                'PointsPastSeasonsGC': []
            })
        
        return result_df
    
    # Apply the historical average computation with progress bar
    tqdm.pandas(desc="Computing historical averages by race type, class, cluster and GC")
    historical_features = df_sorted.groupby('rider_unique_id').progress_apply(compute_historical_avg)
    
    # Initialize columns with NaN
    df_sorted['PointsPastSeasonsRaceType'] = np.nan
    df_sorted['PointsPastSeasonsRaceClass'] = np.nan
    df_sorted['PointsPastSeasonsCluster'] = np.nan
    df_sorted['PointsPastSeasonsGC'] = np.nan
    
    # Update only the target records with calculated values
    if len(historical_features) > 0:
        for col in ['PointsPastSeasonsRaceType', 'PointsPastSeasonsRaceClass', 
                    'PointsPastSeasonsCluster', 'PointsPastSeasonsGC']:
            if col in historical_features.columns:
                # Use loc to update specific indices
                valid_indices = historical_features.index.get_level_values(-1)
                df_sorted.loc[valid_indices, col] = historical_features[col].values
    
    # Restore original order
    df_result = df_sorted.sort_index()
    
    return df_result

def add_historical_performance_bins(df, decision_date_col=None):
    """
    Add historical performance bins features for each rider.
    
    Creates features based on binned performance (1-3, 4-10, 11-30, 31-50, DNF/DNS)
    for three categories: Cluster, Race Class, and Cluster & Race Class combined.
    
    For each bin in each category, computes:
    - Count of results in that bin
    - Days since most recent result in that bin
    - Total participation count
    
    Args:
        df (pd.DataFrame): Input dataframe with rider results
        decision_date_col: Column name containing decision dates for historical filtering.
                          If None, uses race date (no time gap constraint).
        
    Returns:
        pd.DataFrame: DataFrame with added bin features
    """
    print("Computing historical performance bins features...")
    
    # Prepare the dataframe
    df = df.copy()
    df = df.sort_values(['rider', 'date']).reset_index(drop=True)
    
    # Define rank bins
    rank_bins = {
        'rank_1_3': (1, 3),
        'rank_4_10': (4, 10),
        'rank_11_30': (11, 30),
        'rank_31_50': (31, 50),
        'dnf': None  # Special case for DNF/DNS
    }
    
    # Initialize columns for all combinations
    categories = ['cluster', 'RaceClass', 'ClusterClass']
    for category in categories:
        for bin_name in rank_bins.keys():
            # Count columns
            df[f'{category}_{bin_name}_count'] = 0
            # Time since columns
            df[f'{category}_{bin_name}_days_since'] = np.nan
        # Total participation
        df[f'{category}_total_races'] = 0
    
    def process_rider_group(rider_group):
        """Process historical performance bins for a single rider's data"""
        rider_group = rider_group.sort_values('date').reset_index(drop=True)
        n_races = len(rider_group)
        
        # For each race, compute features based on all previous races
        for i in range(n_races):
            current_race = rider_group.iloc[i]
            
            # Skip if this is not a target record
            if not current_race.get('_is_target', True):
                continue
            
            current_date = current_race['date']
            current_cluster = current_race['cluster']
            current_race_class = current_race['race_class']
            
            # Determine the cutoff date for historical data
            if decision_date_col is not None:
                max_historical_date = current_race[decision_date_col]
                current_date = max_historical_date
            else:
                max_historical_date = current_date
            
            # Get all previous races for this rider up to decision date
            prev_races = rider_group[
                rider_group['date'] <= max_historical_date
            ]
            
            if len(prev_races) == 0:
                continue
            
            # Process each category
            categories_data = {
                'Cluster': prev_races[prev_races['cluster'] == current_cluster],
                'RaceClass': prev_races[prev_races['race_class'] == current_race_class],
                'ClusterClass': prev_races[
                    (prev_races['cluster'] == current_cluster) & 
                    (prev_races['race_class'] == current_race_class)
                ]
            }
            
            # For each category
            for category, cat_races in categories_data.items():
                # Set total races
                rider_group.loc[i, f'{category}_total_races'] = len(cat_races)
                
                # Process each bin
                for bin_name, bin_range in rank_bins.items():
                    if bin_name == 'dnf':
                        # Handle DNF/DNS cases
                        bin_races = cat_races[cat_races['rank'].isin(['DNF', 'DNS','OTL'])]
                    else:
                        # Handle numeric ranks
                        bin_races = cat_races[
                            (~cat_races['rank'].isin(['DNF', 'DNS', 'OTL'])) & 
                            (cat_races['rank_number'].astype(float) >= bin_range[0]) & 
                            (cat_races['rank_number'].astype(float) <= bin_range[1])
                        ]
                    
                    # Set days since most recent
                    if len(bin_races) > 0:
                        most_recent_date = bin_races['date'].max()
                        days_since = (current_date - most_recent_date).days
                        rider_group.loc[i, f'{category}_{bin_name}_days_since'] = days_since
                        rider_group.loc[i, f'{category}_{bin_name}_count'] = len(bin_races)
        
        return rider_group
    
    # Process all riders with progress bar
    tqdm.pandas(desc="Processing riders")
    result_df = df.groupby('rider_unique_id').progress_apply(process_rider_group).reset_index(drop=True)
    
    return result_df

def feature_extraction_pipeline(df, race_class=False, time_gap=None, target_years=None):
    """
    Efficient single-pass feature extraction pipeline with decision date pre-calculation
    
    Args:
        df: DataFrame with race data
        race_class: Whether to use race class filtering
        time_gap: Time gap in days before race for feature extraction
        target_years: List of years to extract features for (e.g., [2024, 2025]).
                     If None, extracts for all years in the data.
                     Features will still be calculated using ALL historical data.
    """
    print("Running efficient feature extraction pipeline...")
    
    # Initial preprocessing
    df = preprocess_dates(df)
    
    # Mark target records (records we want to extract features for)
    if target_years is not None:
        print(f"Target years for output: {target_years}")
        print(f"Note: Features will only be calculated for target year records")
        print(f"      but will use ALL historical data for calculations")
        df['_is_target'] = df['year'].isin(target_years)
        target_count = df['_is_target'].sum()
        total_count = len(df)
        print(f"Target records: {target_count:,} / {total_count:,} ({target_count/total_count*100:.1f}%)")
    else:
        df['_is_target'] = True  # All records are targets
    
    # Add unique rider IDs (handles riders with same name but different ages)
    df = add_rider_unique_id(df)

    # Pre-calculate decision dates for all races
    if time_gap is not None:
        print(f"Applying {time_gap}-day time gap constraint...")
        if time_gap != 365:
            df['decision_date'] = df['date'] - pd.Timedelta(days=time_gap)
        else:
            df['decision_date'] = pd.to_datetime(df['date'].dt.year, format='%Y')
        decision_date_col = 'decision_date'
    else:
        df['decision_date'] = df['date']  # No constraint - use race date
        decision_date_col = 'decision_date'

    df = add_career_features(df)
    df = add_race_type(df)
    
    # Now run all feature functions with decision date awareness
    print("Computing historical features with decision date constraints...")
    df = add_combined_historical_features(df, race_class, decision_date_col)
    df = add_historical_performance_bins(df, decision_date_col)
    
    # Points features
    df, yearly_points = add_points_features(df)
    print('Computing rider evolution...')
    df = rider_evolution(df, yearly_points)
    
    # Year-to-date points - use decision date logic only if time gap is provided
    if time_gap is not None:
        df = add_year_to_date_points_with_decision_date(df)
        # Historical points averages (past seasons only)
        print('Computing historical points features...')
        df = add_points_race_type_and_class(df, time_gap=True)
    else:
        # Use efficient cumulative logic when no time gap
        print('Computing points race type and class features...')
        df = add_points_race_type_and_class(df)
    
    # Filter to target records only (features already calculated only for these)
    if target_years is not None:
        df = df[df['_is_target'] == True].copy()
        print(f"Returning target year records: {len(df):,} records")
    
    # Clean up temporary columns
    columns_to_drop = ['decision_date']
    if '_is_target' in df.columns:
        columns_to_drop.append('_is_target')
    df = df.drop(columns_to_drop, axis=1)
    df = remove_rider_unique_id(df)
    
    print("✓ Efficient feature extraction completed")
    return df

def run_rider_features_extraction_pipeline(race_class, time_gap=None, target_years=None):
    """
    Run rider features extraction pipeline
    
    Args:
        race_class (str): Race class ('all' or 'WT') 
        time_gap (int): Time gap in days before race for feature extraction (default: None)
        target_years (list): List of years to extract features for (e.g., [2024, 2025]).
                            If None, extracts for all years in the data.
                            Features will still be calculated using ALL historical data.
        
    Returns:
        str: Path to created rider features file
    """

    print(f"=" * 80)
    print(f"{'RIDER FEATURES EXTRACTION PIPELINE':^80}")
    print(f"=" * 80)
    print(f"Race class: {race_class}")
    print(f"Time gap: {time_gap} days")
    if target_years is not None:
        print(f"Target years: {target_years}")
    
    # Get data paths from config
    riders_results_path = get_data_dir('riders_race_results') 
    output_path = get_data_dir('rider_features')
    
    print(f"Input data: {riders_results_path}")
    print(f"Output path: {output_path}")
    
    
    print(f"\nStep 1: Loading and Preprocessing Data")
    print("-" * 50)
    
    try:
        # Load riders race results
        print(f"Loading riders race results from {riders_results_path}")
        riders_race_results = pd.read_csv(riders_results_path)
        print(f"✓ Loaded {len(riders_race_results)} records")
        riders_race_results['date'] = pd.to_datetime(riders_race_results['date'])

        riders_race_results['year'] = riders_race_results['date'].dt.year
        
        # Apply race class filter
        riders_race_results['classification'] = riders_race_results['classification'].astype(str)
        if race_class == 'WT':
            riders_race_results = riders_race_results[
                riders_race_results['classification'].isin(['1.UWT', '2.UWT'])
            ].copy()
            print(f"✓ After WT filter: {len(riders_race_results)} records")
        
    except Exception as e:
        print(f"❌ Error loading/preprocessing data: {e}")
        raise
    
    print(f"\nStep 2: Extracting Rider Features")
    print("-" * 50)
    
    try:
        # Run feature extraction pipeline
        print("Running feature extraction pipeline...")
        df_features = feature_extraction_pipeline(riders_race_results, race_class == 'all', time_gap, target_years)
        
        # Create binary flag for StageRace
        print(f"✓ Feature extraction completed: {len(df_features)} records")
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        raise
    
    print(f"\nStep 3: Saving Results")
    print("-" * 50)
    
    try:

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Build filename with optional parameters
        filename_parts = ['all_riders_features', f'{race_class}_race_class']
        
        if target_years is not None:
            # Add year range to filename
            year_str = f"{'_'.join(map(str, sorted(target_years)))}"
            filename_parts.append(f'years_{year_str}')
        
        if time_gap is not None:
            filename_parts.append(f'{time_gap}_tg')
        
        filename = '_'.join(filename_parts) + '.csv'
        file_path = os.path.join(output_path, filename)
        
        # Save the features
        df_features.to_csv(file_path, index=False)
        print(f"✓ Rider features saved to: {file_path}")
        print(f"✓ Final dataset shape: {df_features.shape}")
        
        return output_path
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise

