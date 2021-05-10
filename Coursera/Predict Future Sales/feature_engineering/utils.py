import pandas as pd
import numpy as np
from typing import Any


def create_target_lags(df: pd.DataFrame, lag_list: list, na_filler: Any = None) -> pd.DataFrame:
    """
    :param df: pd.DataFrame which you need to calculate lags for
    :param lag_list: list of lags, i.e. [1, 2, 3] means 1, 2 and 3 months lags
    :param na_filler: Any, the value to fill np.nan values in pd.DataFrame in result
    :return: pd.DataFrame with target lags
    """
    lagged_df: pd.DataFrame = df.copy()
    df_helper: pd.DataFrame = df.copy()
    df_helper = df_helper.rename(columns={'date_block_num': 'date_block_actual'})

    for lag in lag_list:
        df_helper['date_block_num'] = df_helper['date_block_actual'] + lag
        lagged_df = lagged_df.merge(df_helper[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']] \
                                    .rename(columns={'item_cnt_month': f'target_lag_{lag}'}),
                                    on=['date_block_num', 'shop_id', 'item_id'], how='left')

    if na_filler is not None:
        lagged_df.fillna(na_filler, inplace=True)

    return lagged_df


def create_price_lags(df: pd.DataFrame, lag_list: list, na_filler: Any = None) -> pd.DataFrame:
    """
    :param df: pd.DataFrame which you need to calculate lags for
    :param lag_list: list of lags, i.e. [1, 2, 3] means 1, 2 and 3 months lags
    :param na_filler: Any, the value to fill np.nan values in pd.DataFrame in result
    :return: pd.DataFrame with mean price lags
    """
    lagged_df: pd.DataFrame = df.copy()
    df_helper: pd.DataFrame = df.copy()
    df_helper = df_helper.rename(columns={'date_block_num': 'date_block_actual'})

    for lag in lag_list:
        df_helper['date_block_num'] = df_helper['date_block_actual'] + lag
        lagged_df = lagged_df.merge(df_helper[['date_block_num', 'shop_id', 'item_id', 'item_price']] \
                                    .rename(columns={'item_price': f'mean_item_price_lag_{lag}'}),
                                    on=['date_block_num', 'shop_id', 'item_id'], how='left')

    if na_filler is not None:
        lagged_df.fillna(na_filler, inplace=True)

    return lagged_df


def create_income_lags(df: pd.DataFrame, lag_list: list, na_filler: Any = None) -> pd.DataFrame:
    """
    :param df: pd.DataFrame which you need to calculate lags for
    :param lag_list: list of lags, i.e. [1, 2, 3] means 1, 2 and 3 months lags
    :param na_filler: Any, the value to fill np.nan values in pd.DataFrame in result
    :return: pd.DataFrame with income lags
    """
    lagged_df: pd.DataFrame = df.copy()
    df_helper: pd.DataFrame = df.copy()
    df_helper = df_helper.rename(columns={'date_block_num': 'date_block_actual'})

    for lag in lag_list:
        df_helper['date_block_num'] = df_helper['date_block_actual'] + lag
        lagged_df = lagged_df.merge(df_helper[['date_block_num', 'shop_id', 'item_id', 'income']] \
                                    .rename(columns={'income': f'income_lag_{lag}'}),
                                    on=['date_block_num', 'shop_id', 'item_id'], how='left')

    if na_filler is not None:
        lagged_df.fillna(na_filler, inplace=True)

    return lagged_df


def rolling_mean(df: pd.DataFrame, window: int, na_filler: Any, column: str) -> pd.Series:
    """
    :param df: pd.DataFrame with created lags
    :param window: int, the width of the rolling window
    :param na_filler: na_filler from the function create_target_lags
    :param column: str, name of the lag column, i.e. target_lag_1, name='target'
    :return: pd.Series, with result column
    """
    df: pd.DataFrame = df.copy()
    rolling_mean_column: str = f'rolling_mean_{column}_{window}'
    df[rolling_mean_column] = np.nan
    subset = df[[f'{column}_lag_{i}' for i in range(1, window + 1)]]
    mask = (subset == na_filler).sum(axis=1) == 0
    df.loc[mask, rolling_mean_column] = subset[mask].sum(axis=1) / window
    df.loc[~mask, rolling_mean_column] = na_filler
    return df[rolling_mean_column]


def create_rolling_mean(df: pd.DataFrame, windowlist: list, na_filler: Any, column: str):
    """
    :param df: pd.DataFrame with created columns of lags
    :param windowlist: list of ints, widths of the rolling window
    :param na_filler: na_filler from the create_target_lags
    :param column: str, name of the lag column, i.e. target_lag_1, name='target'
    :return: pd.DataFrame with rolling means added
    """
    data: list = [df.copy()]
    for window in windowlist:
        data.append(pd.Series(rolling_mean(data[0], window, na_filler=na_filler, column=column)))
    return pd.concat(data, axis=1, copy=False)


def rolling_std(df: pd.DataFrame, window: int, na_filler: Any, column: str) -> pd.Series:
    """
    :param df: pd.DataFrame with created lags
    :param window: int, the width of the rolling window
    :param na_filler: na_filler from the function create_target_lags
    :param column: str, name of the lag column, i.e. target_lag_1, name='target'
    :return: pd.Series, with result column
    """
    df: pd.DataFrame = df.copy()
    rolling_std_column: str = f'rolling_std_{column}_{window}'
    df[rolling_std_column] = np.nan
    subset = df[[f'{column}_lag_{i}' for i in range(1, window + 1)]]
    rolling_mean_column = df[[f'rolling_mean_{column}_{window}']]
    mask = (subset == na_filler).sum(axis=1) == 0
    df.loc[mask, rolling_std_column] = ((subset[mask].values - rolling_mean_column[mask].values) ** 2).sum(axis=1) / \
                                       (window - 1)
    df.loc[~mask, rolling_std_column] = na_filler
    return df[rolling_std_column]


def create_rolling_std(df: pd.DataFrame, windowlist: list, na_filler: Any, column: str):
    """
    :param df: pd.DataFrame with created columns of lags
    :param windowlist: list of ints, widths of the rolling window
    :param na_filler: na_filler from the create_target_lags
    :param column: str, name of the lag column, i.e. target_lag_1, name='target'
    :return: pd.DataFrame with rolling std added
    """
    data: list = [df.copy()]
    for window in windowlist:
        data.append(pd.Series(rolling_std(data[0], window, na_filler=na_filler, column=column)))
    return pd.concat(data, axis=1, copy=False)
