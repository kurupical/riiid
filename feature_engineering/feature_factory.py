from typing import List, Dict
import pandas as pd
import numpy as np
from logging import Logger

class FeatureFactory:
    feature_name_base = ""
    def __init__(self,
                 column: str,
                 logger: Logger=None):
        self.column = column
        self.logger = logger
        self.data_dict = {}
        self.make_col_name = f"{self.feature_name_base}_{self.column}"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, object]]):
        raise NotImplementedError

    def all_predict(self,
                    df: pd.DataFrame):
        raise NotImplementedError

    def partial_predict(self,
                        df: pd.DataFrame):
        df[self.make_col_name] = [self.data_dict[x] if x in self.data_dict else np.nan
                                  for x in df[self.column].values]
        return df

class CountEncoder(FeatureFactory):
    feature_name_base = "count_enc"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        if key not in self.data_dict:
            self.data_dict[key] = len(df)
        else:
            self.data_dict[key] += len(df)

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"count_encoding_all_{self.column}")
        col_name = f"{self.feature_name_base}_{self.column}"
        df[col_name] = df.groupby(self.column).cumcount().astype("int32")
        return df

class TargetEncoder(FeatureFactory):
    feature_name_base = "target_enc"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        if key not in self.data_dict:
            self.data_dict[key] = df["answered_correctly"].sum() / len(df)
        else:
            # count_encoderの値は更新済のため、
            # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
            count = feature_factory_dict[self.column]["CountEncoder"].data_dict[key]
            target_enc = self.data_dict[key]
            self.data_dict[key] = \
                ((count - len(df)) * target_enc + df["answered_correctly"].sum()) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            return series.shift(1).cumsum() / np.arange(len(series))

        self.logger.info(f"target_encoding_all_{self.column}")

        df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].transform(f).astype("float32")
        return df


class MeanAggregator(FeatureFactory):
    feature_name_base = "mean"

    def __init__(self,
                 column: str,
                 agg_column: str,
                 remove_now: bool,
                 logger: Logger=None):
        super().__init__(column, logger)
        self.agg_column = agg_column
        self.remove_now = remove_now
        self.make_col_name = f"{self.feature_name_base}_{self.agg_column}_by_{self.column}"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        if key not in self.data_dict:
            self.data_dict[key] = df[self.agg_column].sum() / len(df)
        else:
            # count_encoderの値は更新済のため、
            # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
            count = feature_factory_dict[self.column]["CountEncoder"].data_dict[key]
            target_enc = self.data_dict[key]
            self.data_dict[key] = \
                ((count - len(df)) * target_enc + df[self.agg_column].sum()) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            if self.remove_now:
                return series.shift(1).cumsum() / np.arange(len(series))
            else:
                return series.cumsum() / (np.arange(len(series)) + 1)

        self.logger.info(f"{self.feature_name_base}_all_{self.column}_{self.agg_column}")
        df[self.make_col_name] = df.groupby(self.column)[self.agg_column].transform(f).astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df[self.make_col_name] = [self.data_dict[x] if x in self.data_dict else np.nan
                                  for x in df[self.column].values]
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df

class FeatureFactoryManager:
    def __init__(self,
                 feature_factory_dict: Dict[str,
                                            Dict[str, FeatureFactory]],
                 logger: Logger):
        """

        :param feature_factory_dict:
            for example:
            {"user_id":
                {"TargetEncoder": TargetEncoder(key="user_id"),
                 "CountEncoder": CountEncoder(key="user_id")}
            }
        """
        self.feature_factory_dict = feature_factory_dict
        self.logger = Logger
        """
        for column in feature_factory_dict.keys():
            self.feature_factory_dict[column] = {}
            self.feature_factory_dict[column]["CountEncoder"] = CountEncoder(column=column, logger=logger)
        """
        for column, dicts in self.feature_factory_dict.items():
            for factory_name, factory in dicts.items():
                factory.logger = logger


    def fit(self,
            df: pd.DataFrame):
        for column, dicts in self.feature_factory_dict.items():
            # カラム(ex: user_idなど)ごとに処理
            for key, w_df in df.groupby(column):
                # カラムのキー(ex. user_id=20000)ごとに処理
                for factory in dicts.values():
                    factory.fit(df=w_df,
                                key=key,
                                feature_factory_dict=self.feature_factory_dict)

    def fit_predict(self,
                    df: pd.DataFrame):
        self.fit(df)
        df = self.all_predict(df)
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        """
        モデル訓練時にまとめてpredictするときに使う
        :param df:
        :return:
        """
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                df = factory.all_predict(df=df)
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        """
        推論時
        :param df:
        :return:
        """
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                df = factory.partial_predict(df)
        return df
