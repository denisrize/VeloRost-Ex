"""
Race results filtering utilities.

Applies team-based cleaning rules and race-level filtering
before feature extraction and experiments.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.config import (
    MIN_TEAM_SIZE,
    MIN_TEAMS_PER_RACE,
    FILTERED_RACE_RESULTS_PATH,
    ROOT_DIR,
    get_data_dir,
)


def _ensure_race_class(df: pd.DataFrame) -> pd.DataFrame:
    if 'race_class' in df.columns:
        return df
    if 'classification' not in df.columns:
        return df

    def map_class(race_class: str) -> str:
        if 'WT' in race_class:
            return 'WT'
        if 'Pro' in race_class:
            return 'Pro'
        if '.1' in race_class:
            return '1'
        return '2'

    df = df.copy()
    df['race_class'] = df['classification'].astype(str).apply(map_class)
    return df


def filter_race_results(
    df: pd.DataFrame,
    min_team_size: int = MIN_TEAM_SIZE,
    min_teams_per_race: int = MIN_TEAMS_PER_RACE,
) -> pd.DataFrame:
    """
    Filter race results using team size rules and minimum teams per race.

    Rules:
    - If team name equals rider name, treat as no team.
    - If a team has fewer than min_team_size riders in a race, treat as no team.
    - Keep only races that have at least min_teams_per_race teams (non-null).
    - Drop rows missing race_class/cluster/rider.
    """
    df_clean = df.copy()

    if 'parcoursType' in df_clean.columns and 'cluster' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'parcoursType': 'cluster'})

    df_clean = _ensure_race_class(df_clean)

    team_eq_rider_mask = df_clean['team'] == df_clean['rider']

    team_size_per_race = (
        df_clean.groupby(['race', 'date', 'team'])
        .size()
        .reset_index(name='team_size_in_race')
    )
    df_clean = df_clean.merge(
        team_size_per_race, on=['race', 'date', 'team'], how='left'
    )

    insufficient_team_size_mask = df_clean['team_size_in_race'] < min_team_size
    total_no_team_mask = team_eq_rider_mask | insufficient_team_size_mask
    df_clean.loc[total_no_team_mask, 'team'] = np.nan
    df_clean = df_clean.drop(columns=['team_size_in_race'])

    key_columns = [col for col in ['race_class', 'cluster', 'rider'] if col in df_clean.columns]
    if key_columns:
        df_clean = df_clean.dropna(subset=key_columns)

    # Filter races by number of teams in the race
    df_clean['race_id'] = df_clean['race'].astype(str) + '_' + df_clean['date'].astype(str)
    teams_per_race = (
        df_clean[df_clean['team'].notna()]
        .groupby('race_id')['team']
        .nunique()
        .reset_index(name='num_teams')
    )
    races_with_min_teams = teams_per_race[
        teams_per_race['num_teams'] >= min_teams_per_race
    ]['race_id'].unique()
    df_filtered = df_clean[df_clean['race_id'].isin(races_with_min_teams)].copy()
    df_filtered = df_filtered.drop(columns=['race_id'])

    return df_filtered


def run_race_results_filtering_pipeline(
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    min_team_size: int = MIN_TEAM_SIZE,
    min_teams_per_race: int = MIN_TEAMS_PER_RACE,
) -> str:
    """
    Load race results, apply filters, and save filtered dataset.
    """
    if input_path is None:
        input_path = get_data_dir('raw_riders_race_results')

    df = pd.read_csv(input_path)
    filtered_df = filter_race_results(
        df,
        min_team_size=min_team_size,
        min_teams_per_race=min_teams_per_race,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    return output_path
