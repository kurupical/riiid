import pandas as pd
import numpy as np

class PartialAggregator:

    def __init__(self, key, logger):
        self.key = key
        self.data_dict = {"count_enc": {},
                          "target_enc": {}}
        self.logger = logger

    def _count_encoding(self,
                        df: pd.DataFrame):
        self.data_dict["count_enc"] = df.groupby(self.key).size().astype("int16").to_dict()

    def _target_encoding(self,
                         df: pd.DataFrame):
        self.data_dict["target_enc"] = df.groupby(self.key)["answered_correctly"].mean().astype("float32").to_dict()

    def _count_encoding_all(self,
                             df: pd.DataFrame):
        self.logger.info(f"count_encoding_all_{self.key}")
        col_name = f"count_enc_{self.key}"
        df[col_name] = df.groupby(self.key).cumcount().astype("int32")
        return df

    def _target_encoding_all(self,
                             df: pd.DataFrame):
        def f(series):
            return series.shift(1).cumsum() / np.arange(len(series))

        self.logger.info(f"target_encoding_all_{self.key}")

        col_name = f"target_enc_{self.key}"
        df[col_name] = df.groupby(self.key)["answered_correctly"].transform(f).astype("float32")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        """
        訓練時に使う。全データを一括処理する
        :return:
        """
        df = self._target_encoding_all(df)
        df = self._count_encoding_all(df)
        return df

    def fit(self,
            df: pd.DataFrame):
        for key, w_df in df.groupby(self.key):
            if key not in self.data_dict["count_enc"]:
                self.data_dict["count_enc"][key] = len(w_df)
                self.data_dict["target_enc"][key] = w_df["answered_correctly"].sum() / len(w_df)
            else:
                count = self.data_dict["count_enc"][key]
                target_enc = self.data_dict["target_enc"][key]
                length = len(w_df)

                self.data_dict["count_enc"][key] = count + length
                self.data_dict["target_enc"][key] = \
                    (count * target_enc + w_df["answered_correctly"].sum()) / (count + length)
        return self

    def partial_predict(self,
                        df: pd.DataFrame):
        df[f"target_enc_{self.key}"] = [self.data_dict["target_enc"][x] if x in self.data_dict["target_enc"] else np.nan
                                        for x in df[self.key].values]
        df[f"count_enc_{self.key}"] = [self.data_dict["count_enc"][x] if x in self.data_dict["count_enc"] else np.nan
                                       for x in df[self.key].values]
        return df


