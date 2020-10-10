import pandas as pd
from typing import List
from logging import Logger
import gc

def agg(df: pd.DataFrame,
        id_name: str,
        columns: List[str],
        agg_funcs: List[str],
        ids: List[int] = None,
        logger: Logger = None,
        is_test: bool = False):
    """

    :param df:
    :param id_name:
    :param columns: 集計対象のカラム
    :param ids:
    :param logger:
    :param is_test:
    :return:
    """

    if is_test:
        df = df[df[id_name].isin(ids)]

    for agg_col in columns:
        group = df.groupby(id_name)[agg_col]
        col_name = f"{col}_count_groupby_{id_name}"

        for agg_func in agg_funcs:
            if agg_func == "cumsum":
                df[col_name] = group.cumsum()
            else:
                raise NotImplementedError(f"agg_func: {agg_func} is not implemented")

    return df

def one_hot_encoding_count(df: pd.DataFrame,
                           id_name: str,
                           column: str,
                           logger: Logger = None):
    """
    id_nameごとに、指定されたcolumnsの値それぞれの出現回数をカウントする

    df[id_name] = [1, 1, 1, 2], df[column] = ["a", "b", "a", "c"]

    id_name_"a" id_name_"b" id_name_"c"
    -----------------------------------
    1           0           0
    1           1           0
    2           1           0
    0           0           1


    :param df:
    :param id_name:
    :param column:
    :param ids:
    :param logger:
    :param is_test:
    :return:
    """


    df_dummies = pd.get_dummies(df[column])
    df_dummies.columns = [f"{column}_{x}" for x in df_dummies.columns]
    df = pd.concat([df, df_dummies], axis=1)

    for count_col in df_dummies.columns:
        col_name = f"{count_col}_count"
        df[col_name] = df.groupby(id_name)[count_col].cumsum()

    return df