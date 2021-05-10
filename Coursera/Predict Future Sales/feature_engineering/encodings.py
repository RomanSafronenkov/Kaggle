import pandas as pd
from typing import Any


def encode_with_expanding_mean(data: pd.DataFrame, column_to_encode: str, target_column: str, na_filler: Any = None)\
        -> pd.DataFrame:
    """
    :param data: pd.DataFrame with all the data
    :param column_to_encode: column from the data which is needed to be encoded
    :param target_column: target column for encoding
    :param na_filler: value to fill NaNs
    :return: pd.DataFrame with new encoded column added
    """
    df: pd.DataFrame = data.copy()
    cumsum = df.groupby(column_to_encode)[target_column].cumsum() - df[target_column]
    cumcnt = df.groupby(column_to_encode)[target_column].cumcount()
    df[column_to_encode + '_target_enc'] = cumsum / (cumcnt + 1e-6)

    if na_filler is not None:
        df[column_to_encode + '_target_enc'].fillna(na_filler, inplace=True)

    return df
