import pandas as pd
import numpy as np
from typing import List
from logging import Logger

def user_intelligency(df):
    def f(series):
        return series.shift(1).cumsum() / (np.arange(len(series)) + 1)

    df["intelligency_point"] = df["answered_correctly"] / (df["target_enc_content_id"]+0.1)
    df["intelligency_point_cummean"] = df.groupby("user_id")["intelligency_point"].transform(f).astype("float16Z")

    df = df.drop("intelligency_point", axis=1)
    return df