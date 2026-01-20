import pandas as pd
import numpy as np
import re
from datetime import datetime

def fix_negative_timelag(group):
    # Sort the group by the ranking (assuming 'rank_number' is numeric)
    group = group.sort_values('rank_number').copy()
    # Create shifted columns: the next row's time and timeLag_stage_sec
    group['next_time'] = group['time'].shift(-1)
    group['next_timeLag_stage_sec'] = group['timeLag_stage_sec'].shift(-1)
    # Create a mask for rows with negative timeLag_stage_sec
    mask = group['timeLag_stage_sec'] < 0
    # For these rows, replace with the values from the next row
    group.loc[mask, 'time'] = group.loc[mask, 'next_time']
    group.loc[mask, 'timeLag_stage_sec'] = group.loc[mask, 'next_timeLag_stage_sec']

    group['timeLag_stage'] = group['timeLag_stage_sec'].apply(seconds_to_hms)
    # Drop temporary columns
    group.drop(columns=['next_time', 'next_timeLag_stage_sec'], inplace=True)
    return group

def fix_OTL_in_group(group, penalty=60):
    """
    For each race group, if a rider's 'time' indicates 'OTL', replace its time gap
    with (max valid timeLag_stage_sec in the group) + penalty.
    """
    group = group.copy()
    # Identify OTL rows by checking if the 'time' column contains "OTL".
    # Adjust the string search as needed.
    mask_otl = group['rank'].astype(str).str.contains("OTL", na=False) & group['timeLag_stage_sec'].isnull()
    
    # For non-OTL rows, ensure we have valid timeLag_stage_sec values.
    valid_gap = group.loc[~mask_otl, 'timeLag_stage_sec']
    if not valid_gap.empty and mask_otl.any():
        max_gap = valid_gap.max()
    
        # For OTL rows, assign a gap equal to the maximum valid gap plus the penalty.
        group.loc[mask_otl, 'timeLag_stage_sec'] = max_gap + penalty
        # Optionally, update the formatted string version as well.
        group.loc[mask_otl, 'timeLag_stage'] = group.loc[mask_otl, 'timeLag_stage_sec'].apply(seconds_to_hms)
        
    return group

def parse_time_absolute(s):
    """Parse an absolute time string in H:M:S format to seconds."""
    if pd.isnull(s):
        return np.nan
    try:
        parts = s.strip().split(':')
        if len(parts) == 3:
            h, m, sec = parts
            return int(h)*3600 + int(m)*60 + int(sec)
        else:
            return np.nan
    except Exception as e:
        return np.nan

def clean_time_gap_string(s):
    """
    Clean a time gap string that might be corrupted.
    
    For example:
      - '1:00:071:00:07' should become '1:00:07'
      - ',,0:05166″' should become '0:05:16' or, if gap intended to be M:S, '0:05'
    
    This function first strips extraneous punctuation and then looks for duplicated patterns.
    """
    if pd.isnull(s):
        return np.nan
    # Remove leading/trailing punctuation, commas, quotes, and special quote characters.
    s = s.strip(" ,\"″")
    # Split by colon.
    parts = s.split(':')
    
    # If there are 4 or more parts, check if it's a duplication.
    if len(parts) >= 4:
        return parts[0]+ ":" + parts[1] + ":" + parts[2][:2]

    return s

def parse_gap_pre2020(s):
    """
    Parse a gap time string for races before 2020.
    Expected formats:
      - A normal "M:S" (e.g., "0:00") which is parsed as minutes*60 + seconds.
      - A corrupted string like "3:233:23" should be interpreted as "3:23".
    """
    s_clean = clean_time_gap_string(s)
    if pd.isnull(s_clean):
        return np.nan
    parts = s_clean.split(':')
    if len(parts) == 2:
        # Assume format is M:S.
        try:
            m = int(parts[0])
            sec = int(parts[1])
            return m * 60 + sec
        except:
            return np.nan
    elif len(parts) == 3:
        # Normal time (more than an hour)
        if len(parts[1]) == 2:
            return parse_time_absolute(s_clean)
        # Corrupted case: if the middle part has more than 2 digits, take only the first 2.
        try:
            # We ignore the hour part here because for gaps it should be M:S.
            if len(parts[1]) % 2 == 0:
                middle = len(parts[1]) // 2
            else:
                middle = (len(parts[1]) // 2) + 1

            m = int(parts[1][middle:])
            sec = int(parts[2])
            return m * 60 + sec
        except:
            return np.nan
    else:
        try:
            return float(s)
        except:
            return np.nan

def seconds_to_hms(seconds):
    """Convert seconds to H:M:S format."""
    if pd.isnull(seconds):
        return None
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def fill_timeLag_stage_group(group):
    """
    Compute a new column 'timeLag_stage_sec' that represents the time gap from the stage winner.
    
    For races before 2020:
      - For the first-ranked rider, set timeLag_stage to 0.
      - For other riders, parse the 'time' as a gap using parse_gap_pre2020.
    
    For races from 2020 onward:
      - Convert all 'time' values as absolute times (in seconds).
      - Set the baseline as the first-ranked rider's time and compute the gap = rider_time - baseline_time.
    """
    group = group.copy()
    # Get the race year (assume it's the same for all rows in the group)
    race_year = group['year'].iloc[0]
    
    if race_year < 2020:
        # Pre-2020: first-ranked rider's time is absolute; others are gap strings.
        # For the baseline row, we set gap to 0.
        def compute_gap_pre(row):
            if row['rank_number'] == 1:
                return 0
            else:
                return parse_gap_pre2020(row['time'])
        group['timeLag_stage_sec'] = group.apply(compute_gap_pre, axis=1)
        group['time_sec'] = group.loc[group['rank_number'] == 1]['time'].apply(parse_time_absolute)
    else:
        # From 2020 onward: all times are absolute.
        group['time_sec'] = group['time'].apply(parse_time_absolute)
        # Identify the baseline row: first-ranked rider (using rank_number).
        baseline_candidates = group[(group['rank_number'] == 1) & (group['time_sec'].notnull())]
        if not baseline_candidates.empty:
            baseline_time = group.loc[baseline_candidates.index[0], 'time_sec']
        else:
            valid = group[group['time_sec'].notnull()].sort_values('rank_number')
            if not valid.empty:
                baseline_time = valid['time_sec'].iloc[0]
            else:
                baseline_time = np.nan
        group['timeLag_stage_sec'] = group['time_sec'] - baseline_time
        group.loc[group['rank_number'] == 1, 'timeLag_stage_sec'] = 0
    
    # Optionally, convert seconds to H:M:S
    group['timeLag_stage'] = group['timeLag_stage_sec'].apply(seconds_to_hms)
    return group

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
    
def log_transform(riders_race_results):
    riders_race_results = riders_race_results.copy()
    mask = (riders_race_results['timeLag_stage_sec'] == 0) & (riders_race_results['rank_number'] != 1)

    riders_race_results['timeLag_stage_sec_eps'] = riders_race_results[ 'timeLag_stage_sec']
    riders_race_results.loc[mask, 'timeLag_stage_sec_eps'] += riders_race_results.loc[mask,'rank_number'] * 0.01
    riders_race_results['log_timeLag_stage_sec'] = np.log(riders_race_results['timeLag_stage_sec_eps'] + 1)

    return riders_race_results