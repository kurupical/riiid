import pandas as pd
from typing import List
from logging import Logger
import gc
import numpy as np

def target_encoding(df: pd.DataFrame,
                    col: str,
                    df_testdata: pd.DataFrame=None):
    def f(series):
        return series.shift(1).cumsum() / (np.arange(len(series)) + 1)

    col_name = f"target_enc_{col}"
    if df_testdata is None:
        df[col_name] = df.groupby(col)["answered_correctly"].transform(f).astype("float32")
        return df
    else:
        df_testdata = df.groupby(col)["answered_correctly"].mean()
        return df