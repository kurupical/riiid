from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
from logging import Logger

class FeatureFactory:
    feature_name_base = ""
    def __init__(self,
                 column: Union[list, str],
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = column
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.feature_name_base}_{self.column}"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):
        raise NotImplementedError

    def make_feature(self,
                     df: pd.DataFrame):
        """
        単純な特徴の追加
        :param df:
        :return:
        """
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        raise NotImplementedError

    def _partial_predict(self,
                         df: pd.DataFrame):
        if type(self.column) == list:
            df[self.make_col_name] = [self.data_dict[tuple(x)] if tuple(x) in self.data_dict else np.nan
                                      for x in df[self.column].values]
            return df
        if type(self.column) == str:
            df[self.make_col_name] = [self.data_dict[x] if x in self.data_dict else np.nan
                                      for x in df[self.column].values]
            return df
        raise ValueError

    def partial_predict(self,
                        df: pd.DataFrame):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(key=f{self.column}"

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

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict(df)
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int32")
        if "user_id" not in self.make_col_name:
            df[self.make_col_name] *= self.split_num
        return df


class TargetEncoder(FeatureFactory):
    feature_name_base = "target_enc"

    def __init__(self,
                 column: Union[list, str],
                 initial_weight: int = 0,
                 initial_score: float = 0,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        super().__init__(column=column,
                         split_num=split_num,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.initial_weight = initial_weight
        self.initial_score = initial_score

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]]):

        initial_bunshi = self.initial_score * self.initial_weight
        if key not in self.data_dict:
            self.data_dict[key] = (df["answered_correctly"].sum() + initial_bunshi)/ (len(df) + self.initial_weight)
        else:
            # count_encoderの値は更新済のため、
            # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
            if type(self.column) == list:
                count = feature_factory_dict[tuple(self.column)]["CountEncoder"].data_dict[key] + self.initial_weight
            elif type(self.column) == str:
                count = feature_factory_dict[self.column]["CountEncoder"].data_dict[key] + self.initial_weight
            else:
                raise ValueError
            target_enc = self.data_dict[key]

            # パフォーマンス対策:
            # df["answered_correctly"].sum()
            ans_sum = df["answered_correctly"].values
            if len(df) == 1:
                ans_sum = ans_sum[0]
            else:
                ans_sum = ans_sum.sum()

            self.data_dict[key] = \
                ((count - len(df)) * target_enc + ans_sum) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            return (series.shift(1).cumsum().fillna(0) + self.initial_weight * self.initial_score) / (np.arange(len(series)) + self.initial_weight)

        self.logger.info(f"target_encoding_all_{self.column}")

        df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].transform(f).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict(df)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        if self.initial_weight > 0:
            df[self.make_col_name] = df[self.make_col_name].fillna(self.initial_score)
        return df


class TagsSeparator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):
        tag = df["tags"].str.split(" ", n=10, expand=True)
        tag.columns = [f"tags{i}" for i in range(1, len(tag.columns) + 1)]

        for col in ["tags1", "tags2", "tags3", "tags4", "tags5", "tags6"]:
            if col in tag.columns:
                df[col] = pd.to_numeric(tag[col], errors='coerce').fillna(-1).astype("int16")
            else:
                df[col] = -1
                df[col].astype("int16")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

class MeanAggregator(FeatureFactory):
    feature_name_base = "mean"

    def __init__(self,
                 column: str,
                 agg_column: str,
                 remove_now: bool,
                 logger: Union[Logger, None] =None,
                 is_partial_fit: bool = False):
        super().__init__(column=column,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.agg_column = agg_column
        self.remove_now = remove_now
        self.make_col_name = f"{self.feature_name_base}_{self.agg_column}_by_{self.column}"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]]):
        if key not in self.data_dict:
            self.data_dict[key] = df[self.agg_column].sum() / len(df)
        else:
            # count_encoderの値は更新済のため、
            # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
            count = feature_factory_dict[self.column]["CountEncoder"].data_dict[key]
            target_enc = self.data_dict[key]
            self.data_dict[key] = \
                ((count - len(df)) * target_enc + df[self.agg_column].values.sum()) / count
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
        df = self._partial_predict(df)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df


class UserLevelEncoder(FeatureFactory):
    feature_name_base = "user_level"
    def __init__(self,
                 initial_score: float =.0,
                 initial_weight: float = 0,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = "user_id"
        self.initial_score = initial_score
        self.initial_weight = initial_weight
        self.logger = logger
        self.is_partial_fit = False
        self.data_dict = {}
        self.make_col_name = f"{self.feature_name_base}"

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        initial_bunshi = self.initial_score * self.initial_weight
        if key not in self.data_dict:
            self.data_dict[key] = (df["target_enc_content_id"].sum() + initial_bunshi) / (len(df) + self.initial_weight)
        else:
            # count_encoderの値は更新済のため、
            # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
            target_enc = self.data_dict[key]
            count = feature_factory_dict["user_id"]["CountEncoder"].data_dict[key] + self.initial_weight

            # パフォーマンス対策:
            # df["answered_correctly"].sum()
            ans_sum = df["target_enc_content_id"].values
            if len(df) == 1:
                ans_sum = ans_sum[0]
            else:
                ans_sum = ans_sum.sum()
            self.data_dict[key] = ((count - len(df)) * target_enc + ans_sum) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            return (series.cumsum() + self.initial_weight * self.initial_score) / (np.arange(len(series)) + 1 + self.initial_weight)

        df["user_level"] = df.groupby("user_id")["target_enc_content_id"].transform(f).astype("float32")
        df["diff_user_level"] = df[self.make_col_name] - df["target_enc_content_id"]
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict(df)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df["diff_user_level"] = df[self.make_col_name] - df["target_enc_content_id"]
        if self.initial_weight > 0:
            df[self.make_col_name] = df[self.make_col_name].fillna(self.initial_score)
        return df


class NUniqueEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 groupby: str,
                 column: str,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"nunique_{self.column}_by_{self.groupby}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):
        for column in df[self.column].values:
            if key not in self.data_dict:
                self.data_dict[key] = set(column)
            else:
                self.data_dict[key].add(column)
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        ret = []
        # 1件ずつ確認し
        for x in df[[self.groupby, self.column]].values:
            groupby = x[0]
            column = x[1]
            if groupby not in self.data_dict:
                self.data_dict[groupby] = set(column)
            else:
                self.data_dict[groupby].add(column)
            ret.append(len(self.data_dict[groupby]))
            print(self.data_dict, groupby, ret)

        df[self.make_col_name] = ret
        df[self.make_col_name] = df[self.make_col_name].astype("int32")
        df[f"new_ratio_{self.make_col_name}"] = (df[self.make_col_name] / (df[f"count_enc_{self.groupby}"]+1)).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df[self.make_col_name] = [len(self.data_dict[x]) if x in self.data_dict else np.nan
                                  for x in df[self.groupby].values]
        df[f"new_ratio_{self.make_col_name}"] = (df[self.make_col_name] / (df[f"count_enc_{self.groupby}"])).astype("float32")
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int32")
        return df


class ShiftDiffEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 groupby: str,
                 column: str,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"shiftdiff_{self.column}_by_{self.groupby}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            key: str,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):
        self.data_dict[key] = df[self.column].iloc[-1]
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df[self.column] - df.groupby(self.groupby)[self.column].shift(1)
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int64")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        w_diff = [self.data_dict[x] if x in self.data_dict else np.nan for x in df[self.groupby].values]
        df[self.make_col_name] = df[self.column] - w_diff
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int64")
        return df


class FeatureFactoryManager:
    def __init__(self,
                 feature_factory_dict: Dict[Union[str, tuple],
                                            Dict[str, FeatureFactory]],
                 logger: Logger,
                 split_num: int=1):
        """

        :param feature_factory_dict:
            for example:
            {"user_id":
                {"TargetEncoder": TargetEncoder(key="user_id"),
                 "CountEncoder": CountEncoder(key="user_id")}
            }
        """
        self.feature_factory_dict = feature_factory_dict
        self.logger = logger
        self.split_num = split_num
        """
        for column in feature_factory_dict.keys():
            self.feature_factory_dict[column] = {}
            self.feature_factory_dict[column]["CountEncoder"] = CountEncoder(column=column, logger=logger)
        """
        for column, dicts in self.feature_factory_dict.items():
            for factory_name, factory in dicts.items():
                factory.logger = logger
                factory.split_num = split_num


    def fit(self,
            df: pd.DataFrame,
            partial_predict_mode: bool=True):
        for column, dicts in self.feature_factory_dict.items():
            # カラム(ex: user_idなど)ごとに処理
            if type(column) == tuple:
                group = df.groupby(list(column))
            elif type(column) == str:
                group = df.groupby(column)
            else:
                raise ValueError
            import time
            t = time.time()
            for key, w_df in group:
                # カラムのキー(ex. user_id=20000)ごとに処理
                for factory in dicts.values():
                    factory.fit(df=w_df,
                                key=key,
                                feature_factory_dict=self.feature_factory_dict)
            for factory in dicts.values():
                df = factory.make_feature(df)
            for factory in dicts.values():
                if factory.is_partial_fit and partial_predict_mode:
                    df = factory.partial_predict(df)

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

