import pandas as pd
from typing import Tuple

DATA_PATH = './data/'


def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param path: str, path where to find data sales_train.csv, item_categories.csv, items.csv, shops.csv, test.csv
    :return: sales_train, item_categories, items, shops, test
    """
    sales_train = pd.read_csv(path + 'sales_train.csv')
    sales_train.date = pd.to_datetime(sales_train.date, format='%d.%m.%Y')
    item_categories = pd.read_csv(path + 'item_categories.csv')
    items = pd.read_csv(path + 'items.csv')
    shops = pd.read_csv(path + 'shops.csv')
    test = pd.read_csv(path + 'test.csv')
    return sales_train, item_categories, items, shops, test


def downgrade_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: pd.DataFrame, which columns with types int64 and float64 need to be downgraded to int32 and float32
    relatively
    :return: same pd.DataFrame but with the downgraded types
    """
    df: pd.DataFrame = df.copy()

    for column in df.columns:
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('int32')
        elif df[column].dtype == 'float64':
            df[column] = df[column].astype('float32')
    return df
