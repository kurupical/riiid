from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import tqdm
from logging import Logger
import time
import pickle
import os
import glob

tqdm.tqdm.pandas()

class FeatureFactory:
    feature_name_base = ""
    def __init__(self,
                 column: Union[list, str],
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        """

        :param column:
        :param split_num:
        :param logger:
        :param is_partial_fit:
        :param is_all_fit:
            fit時のflag. fitは処理時間削減のため通常150行に1回まとめて行うが、そうではなく逐次fitしたいときはTrueを入れる
        """
        self.column = column
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.feature_name_base}_{self.column}"# .replace(" ", "").replace("'", "")

    def fit(self,
            group,
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

    def _partial_predict2(self,
                          df: pd.DataFrame,
                          column: str):
        if type(self.column) == list:
            df[column] = [self.data_dict[tuple(x)][column] if tuple(x) in self.data_dict else np.nan
                          for x in df[self.column].values]
            df[column] = df[column].astype("float32")
            return df
        if type(self.column) == str:
            df[column] = [self.data_dict[x][column] if x in self.data_dict else np.nan
                          for x in df[self.column].values]
            df[column] = df[column].astype("float32")
            return df
        raise ValueError

    def partial_predict(self,
                        df: pd.DataFrame):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"

class CountEncoder(FeatureFactory):
    feature_name_base = "count_enc"

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        w_dict = group.size().to_dict()

        for key, value in w_dict.items():
            if key not in self.data_dict:
                self.data_dict[key] = value
            else:
                self.data_dict[key] += value

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"count_encoding_all_{self.column}")
        df[self.make_col_name] = df.groupby(self.column).cumcount().astype("int32")
        if "user_id" not in self.make_col_name:
            df[self.make_col_name] *= self.split_num

        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict(df)
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int32")
        if "user_id" not in self.make_col_name:
            df[self.make_col_name] *= self.split_num
        return df


class Counter(FeatureFactory):
    def __init__(self,
                 groupby_column: str,
                 agg_column: str,
                 categories: list,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby_column = groupby_column
        self.agg_column = agg_column
        self.categories = categories
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):

        for key, df in group:
            if key not in self.data_dict:
                self.data_dict[key] = {}
                for col in self.categories:
                    self.data_dict[key][col] = 0
            for col, w_df in df.groupby(self.agg_column):
                self.data_dict[key][col] += len(w_df)

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"categories_count_{self.groupby_column}")
        for col in df[self.agg_column].drop_duplicates():
            def f(series):
                w = (series == col).astype("int8").values
                return np.cumsum(w)

            def f_ratio(series):
                return series / (np.arange(len(series)) + 1)

            col_name = f"groupby_{self.groupby_column}_{self.agg_column}_{col}_count"
            df[col_name] = df.groupby(self.groupby_column)[self.agg_column].transform(f)
            df[f"{col_name}_ratio"] = df.groupby(self.groupby_column)[col_name].transform(f_ratio)
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        cols = []
        for col in self.categories:
            col_name = f"groupby_{self.groupby_column}_{self.agg_column}_{col}_count"
            cols.append(col_name)
            df[col_name] = [self.data_dict[x][col] if x in self.data_dict else 0
                            for x in df[self.groupby_column].values]
            df[col_name] = df[col_name].astype("int32")

        for col in cols:
            df[f"{col}_ratio"] = df[col] / df["count_enc_user_id"]
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
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]]):

        def f(series):
            return series.notnull().sum()
        initial_bunshi = self.initial_score * self.initial_weight
        sum_dict = group["answered_correctly"].sum().to_dict()
        size_dict = group["is_question"].sum().to_dict()

        for key in sum_dict.keys():
            w_sum = sum_dict[key]
            w_size = size_dict[key]

            if w_size == 0:

                continue
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"target_enc_{self.column}"] = (w_sum + initial_bunshi) / (w_size + self.initial_weight)
                self.data_dict[key]["size"] = w_size
            else:
                target_enc = self.data_dict[key][f"target_enc_{self.column}"]
                count = self.data_dict[key]["size"] + w_size + self.initial_weight

                # パフォーマンス対策:
                # df["answered_correctly"].sum()

                self.data_dict[key][f"target_enc_{self.column}"] = ((count - w_size) * target_enc + w_sum) / count
                self.data_dict[key]["size"] = self.data_dict[key]["size"] + w_size
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            bunshi = series.shift(1).notnull().cumsum() + self.initial_weight
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        self.logger.info(f"target_encoding_all_{self.column}")

        df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].transform(f).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict2(df, column=f"target_enc_{self.column}")
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
            group,
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

        for col in ["tags1", "tags2"]:
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

    def __repr__(self):
        return f"{self.__class__.__name__}"


class TagsSeparator2(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
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
        tags = np.arange(1, 188)
        for col in tag.columns:
            tag[col] = pd.to_numeric(tag[col], errors='coerce').fillna(-1).astype("uint8")

        for t in tags:
            df[f"tag_{t}"] = (tag == t).sum(axis=1).astype("uint8")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class PartSeparator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        for i in [1, 2, 3, 4, 5, 6, 7]:
            df[f"part{i}"] = (df["part"] == i).astype("int8")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"part_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class UserCountBinningEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):

        df["user_count_bin"] = pd.cut(df["count_enc_user_id"], [-1, 30, 10**2, 10**2.5, 10**3, 10**4, 10**5],
                                      labels=False).astype("uint8")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"usercount_bin_all")

        return self._predict(df=df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df=df)

    def __repr__(self):
        return f"{self.__class__.__name__}"



class PriorQuestionElapsedTimeDiv10Encoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        df["prior_question_elapsed_time_div10"] = df["prior_question_elapsed_time"] % 10
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"prior_question_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PriorQuestionElapsedTimeBinningEncoder(FeatureFactory):

    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):
        df["prior_question_elapsed_time_bin"] = pd.cut(df["prior_question_elapsed_time"],
                                                       [-1, 1000, 5000, 10000, 15000, 20000, 25000, 50000, 75000,
                                                        100000, 1000000], labels=False).fillna(255).astype("uint8")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"



class TargetEncodeVsUserId(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.logger = logger
        self.is_partial_fit = is_partial_fit

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        target_cols = [x for x in df.columns if "target_enc_" in x and "user_id" not in x]

        for col in target_cols:
            col_name = f"target_enc_user_id_vs_{col}_diff"
            df[col_name] = df["target_enc_user_id"] - df[col]
            df[col_name] = df[col_name].astype("float32")
        return df

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"targetenc_vsuserid")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class MeanAggregator(FeatureFactory):
    feature_name_base = "mean"

    def __init__(self,
                 column: Union[str, list],
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
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]]):
        sum_dict = group[self.agg_column].sum().to_dict()
        size_dict = group["is_question"].sum().to_dict()
        for key in sum_dict.keys():
            sum_ = sum_dict[key]
            size_ = size_dict[key]
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][self.make_col_name] = sum_ / size_
                self.data_dict[key]["size"] = size_
            else:
                # count_encoderの値は更新済のため、
                # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
                count = self.data_dict[key]["size"].data_dict[key]
                target_enc = self.data_dict[key]
                self.data_dict[key][self.make_col_name] = ((count - size_) * target_enc + sum_) / count
                self.data_dict[key]["size"] = size_
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(series):
            return (series.shift(1).fillna(0).cumsum()) / series.shift(1).notnull().cumsum()

        self.logger.info(f"{self.feature_name_base}_all_{self.column}_{self.agg_column}")
        df[self.make_col_name] = df.groupby(self.column)[self.agg_column].transform(f).astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df = self._partial_predict2(df, column=self.make_col_name)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df


class UserLevelEncoder2(FeatureFactory):
    def __init__(self,
                 vs_column: Union[str, list],
                 initial_score: float =.0,
                 initial_weight: float = 0,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = "user_id"
        self.vs_column = vs_column
        self.initial_score = initial_score
        self.initial_weight = initial_weight
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        initial_bunshi = self.initial_score * self.initial_weight

        for key, df in group:
            rate = (df["answered_correctly"] - df[f"target_enc_{self.vs_column}"])
            rate = rate[rate.notnull()]
            if len(rate) == 0:
                continue
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"user_level_{self.vs_column}"] = (df[f"target_enc_{self.vs_column}"].sum() + initial_bunshi) / (len(rate) + self.initial_weight)
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = rate.sum()
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = rate.mean()
                self.data_dict[key]["count"] = len(rate)
            else:
                user_level = self.data_dict[key][f"user_level_{self.vs_column}"]
                user_rate_sum = self.data_dict[key][f"user_rate_sum_{self.vs_column}"]

                count = self.data_dict[key]["count"] + len(rate) + self.initial_weight

                # パフォーマンス対策:
                # df["answered_correctly"].sum()
                ans_sum = df[f"target_enc_{self.vs_column}"].sum()
                rate_sum = rate.sum()
                self.data_dict[key][f"user_level_{self.vs_column}"] = ((count - len(rate)) * user_level + ans_sum) / count
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = user_rate_sum + rate_sum
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = (user_rate_sum + rate_sum) / count
                self.data_dict[key]["count"] = self.data_dict[key]["count"] + len(rate)
        return self

    def all_predict(self,
                    df: pd.DataFrame):

        def f_shift1_mean(series):
            bunshi = series.shift(1).notnull().cumsum() + self.initial_weight
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        def f_shift1_sum(series):
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score)

        def f(series):
            bunshi = series.notnull().cumsum() + self.initial_weight
            return (series.fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]
        df[f"user_rate_sum_{self.vs_column}"] = df.groupby("user_id")["rate"].transform(f_shift1_sum).astype("float32")
        df[f"user_rate_mean_{self.vs_column}"] = df.groupby("user_id")["rate"].transform(f_shift1_mean).astype("float32")
        df[f"user_level_{self.vs_column}"] = df.groupby("user_id")[f"target_enc_{self.vs_column}"].transform(f).astype("float32")
        df[f"diff_user_level_target_enc_{self.vs_column}"] = \
            (df[f"user_level_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        df[f"diff_rate_mean_target_emc_{self.vs_column}"] = \
            (df[f"user_rate_mean_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")

        df = df.drop("rate", axis=1)
        return df


    def partial_predict(self,
                        df: pd.DataFrame):
        for col in [f"user_rate_sum_{self.vs_column}",
                    f"user_rate_mean_{self.vs_column}",
                    f"user_level_{self.vs_column}"]:
            df = self._partial_predict2(df, column=col)
        df[f"diff_user_level_target_enc_{self.vs_column}"] = \
            (df[f"user_level_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        df[f"diff_rate_mean_target_emc_{self.vs_column}"] = \
            (df[f"user_rate_mean_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        return df


class CategoryLevelEncoder(FeatureFactory):
    def __init__(self,
                 groupby_column: str,
                 agg_column: str,
                 categories: list,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 vs_columns: Union[str, list] = "content_id"):
        self.groupby_column = groupby_column
        self.agg_column = agg_column
        self.categories = categories
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.vs_columns = vs_columns


    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        for key, df in group:
            for category, w_df in df.groupby(self.agg_column):
                rate = w_df["answered_correctly"] - w_df[f"target_enc_{self.vs_columns}"]
                size = rate.notnull().sum()
                if key not in self.data_dict:
                    self.data_dict[key] = {}
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] = size
                    self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = rate.sum()
                    self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = rate.mean()
                elif f"user_rate_sum_{self.agg_column}_{category}" not in self.data_dict[key]:
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] = size
                    self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = rate.sum()
                    self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = rate.mean()
                else:
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] += size
                    user_rate_sum = self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"]

                    count = self.data_dict[key][f"count_{self.agg_column}_{category}"]
                    rate_sum = rate.sum()

                    self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = user_rate_sum + rate_sum
                    self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = (user_rate_sum + rate_sum) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f_shift1_sum(series):
            return series.shift(1).cumsum()

        for category in self.categories:
            df["is_target"] = (df[self.agg_column] == category).astype("uint8")
            df["is_target"] = df["is_target"] * df[f"target_enc_{self.vs_columns}"].notnull().astype("uint8")
            df["rate"] = (df["answered_correctly"] - df[f"target_enc_{self.vs_columns}"]).fillna(0) * df["is_target"]
            df["size"] = df.groupby("user_id")["is_target"].cumsum().shift(1)
            df[f"user_rate_sum_{self.agg_column}_{category}"] = df.groupby("user_id")["rate"].transform(f_shift1_sum).astype("float32")
            df[f"user_rate_mean_{self.agg_column}_{category}"] = (df[f"user_rate_sum_{self.agg_column}_{category}"] / df["size"]).astype("float32")
            df[f"diff_rate_mean_target_enc_{self.agg_column}_{category}"] = \
                df[f"user_rate_mean_{self.agg_column}_{category}"] - df[f"target_enc_{self.vs_columns}"]

        df = df.drop(["rate", "is_target", "size"], axis=1)
        return df

    def _partial_predict2(self,
                          df: pd.DataFrame,
                          column: str):
        def f(x):
            if x not in self.data_dict:
                return np.nan
            if column not in self.data_dict[x]:
                return np.nan
            return self.data_dict[x][column]
        df[column] = [f(x) for x in df[self.groupby_column].values]
        df[column] = df[column].astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        for category in self.categories:
            for col in [f"user_rate_sum_{self.agg_column}_{category}",
                        f"user_rate_mean_{self.agg_column}_{category}"]:
                df = self._partial_predict2(df, column=col)
            df[f"diff_rate_mean_target_enc_{self.agg_column}_{category}"] = \
                df[f"user_rate_mean_{self.agg_column}_{category}"] - df[f"target_enc_{self.vs_columns}"]
        return df
    def __repr__(self):
        return f"{self.__class__.__name__}(groupby_columns={self.groupby_column}, agg_column={self.agg_column})"



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
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):

        if len(self.data_dict) == 0:
            self.data_dict = group.nunique().to_dict()
            return self
        for key, df in group:
            for column in df[self.column].drop_duplicates().values:
                if key not in self.data_dict:
                    self.data_dict[key] = [column]
                else:
                    if column not in self.data_dict[key]:
                        self.data_dict[key].append(column)
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        ret = []
        # 1件ずつ確認し
        for x in df[[self.groupby, self.column]].values:
            groupby = x[0]
            column = x[1]
            if groupby not in self.data_dict:
                self.data_dict[groupby] = [column]
            else:
                if column not in self.data_dict[groupby]:
                    self.data_dict[groupby].append(column)
            ret.append(len(self.data_dict[groupby]))

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
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):
        if len(self.data_dict) == 0:
            self.data_dict = group[self.column].last().to_dict()
        else:
            for key, value in group[self.column].last().to_dict().items():
                self.data_dict[key] = value
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df[self.column] - df.groupby(self.groupby)[self.column].shift(1)
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int64")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        """
        リアルタイムfit
        :param df:
        :return:
        """
        groupby_values = df[self.groupby].values

        def f(idx):
            """
            xがnullのときは、辞書に前の時間が登録されていればその時間との差分を返す。
            そうでなければ、0を返す
            :param idx:
            :return:
            """
            if groupby_values[idx] in self.data_dict:
                return self.data_dict[groupby_values[idx]]
            else:
                return 0

        w_diff = df.groupby(self.groupby)[self.column].shift(1)
        w_diff = [x if not np.isnan(x) else f(idx) for idx, x in enumerate(w_diff.values)]
        df[self.make_col_name] = (df[self.column] - w_diff).astype("int64")

        for key, value in df.groupby(self.groupby)[self.column].last().to_dict().items():
            self.data_dict[key] = value
        return df


class PreviousAnswer(FeatureFactory):
    feature_name_base = "previous_answer"

    def _make_key(self, x):
        return (x[0]+1) * (x[1]+1)
    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):

        last_dict = group["answered_correctly"].last().to_dict()
        for key, value in last_dict.items():
            self.data_dict[self._make_key(key)] = value
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"previous_encoding_all_{self.column}")
        df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].shift(1).fillna(-99).astype("int8")

        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        def f(x):
            return (x[0]+1)*(x[1]+1)
        df[self.make_col_name] = [self.data_dict[self._make_key(x)] if self._make_key(x) in self.data_dict else np.nan
                                                for x in df[self.column].values]
        df[self.make_col_name] = df[self.make_col_name].fillna(-99).astype("int8")
        return df

class PreviousAnswer2(FeatureFactory):
    feature_name_base = "previous_answer"
    pickle_path = "../input/feature_engineering/previous_answer2_{}.pickle"
    def __init__(self,
                 groupby: str,
                 column: str,
                 is_debug: bool = False,
                 repredict: bool = False,
                 model_id: int = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.is_debug = is_debug
        self.logger = logger
        self.repredict = repredict
        self.model_id = model_id
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):

        for user_id, w_df in group[["content_id", "answered_correctly"]]:
            content_id = w_df["content_id"].values[::-1].tolist()
            answer = w_df["answered_correctly"].values[::-1].tolist()
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                self.data_dict[user_id]["content_id"] = content_id
                self.data_dict[user_id]["answered_correctly"] = answer
            else:
                self.data_dict[user_id]["content_id"] = content_id + self.data_dict[user_id]["content_id"][len(content_id):][:1000]
                self.data_dict[user_id]["answered_correctly"] = answer + self.data_dict[user_id]["answered_correctly"][len(content_id):][:1000]
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"previous_encoding_all_{self.column}")
        def f(series):
            """
            何個前の特徴だったか
            :param series:
            :return:
            """
            ary = []
            for i in range(len(series)):
                ary.append(series.shift(i).values)
            ary = np.array(ary) # shape=(len(series), len(series)

            diff_ary = ary[0:1, :] - ary[1:, :]
            ret = []
            for i in range(diff_ary.shape[1]):
                w_ret = np.where(diff_ary[:, i] == 0)[0]
                if len(w_ret) == 0:
                    ret.append(None)
                else:
                    ret.append(w_ret[0])
            return ret
        self.logger.info(f"previous_encoding_all_{self.column}")
        pickle_path = self.pickle_path.format(self.model_id)
        if not self.repredict and os.path.isfile(pickle_path) and not self.is_debug:
            self.logger.info(f"load_previous_encoding_pickle")
            with open(pickle_path, "rb") as f:
                w_df = pickle.load(f)
            for col in w_df.columns:
                df[col] = w_df[col].values
        else:
            prev_answer = df.groupby([self.groupby, "content_id"])["answered_correctly"].shift(1).fillna(-99).astype("int8")
            prev_answer_index = df.groupby("user_id")["content_id"].progress_transform(f).fillna(-99).astype("int16")
            df[f"previous_answer_{self.column}"] = prev_answer
            df[f"previous_answer_index_{self.column}"] = prev_answer_index

            # save feature
            if not self.is_debug:
                w_df = pd.DataFrame()
                with open(pickle_path, "wb") as f:
                    w_df[f"previous_answer_{self.column}"] = prev_answer
                    w_df[f"previous_answer_index_{self.column}"] = prev_answer_index
                    pickle.dump(w_df, f)

        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        def get_index(l, x):
            try:
                ret = l.index(x)
                return ret
            except ValueError:
                return None

        def f(x):
            """
            index, answered_correctlyを辞書から検索する
            data_dictはリアルタイム更新する。ただし、answered_correctlyはわからないのでnp.nanとしておく
            :param x:
            :return:
            """
            user_id = x[0]
            content_id = x[1]
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                self.data_dict[user_id]["content_id"] = [content_id]
                self.data_dict[user_id]["answered_correctly"] = [None]
                return [None, None]
            last_idx = get_index(self.data_dict[user_id]["content_id"], content_id) # listは逆順になっているので

            if last_idx is None: # user_idに対して過去content_idの記録がない
                ret = [None, None]
            else:
                ret = [self.data_dict[user_id]["answered_correctly"][last_idx], last_idx]
            self.data_dict[user_id]["content_id"] = [content_id] + self.data_dict[user_id]["content_id"]
            self.data_dict[user_id]["answered_correctly"] = [None] + self.data_dict[user_id]["answered_correctly"]
            return ret

        ary = [f(x) for x in df[[self.groupby, self.column]].values]
        ans_ary = [x[0] for x in ary]
        index_ary = [x[1] for x in ary]
        df[f"previous_answer_{self.column}"] = ans_ary
        df[f"previous_answer_{self.column}"] = df[f"previous_answer_{self.column}"].fillna(-99).astype("int8")
        df[f"previous_answer_index_{self.column}"] = index_ary
        df[f"previous_answer_index_{self.column}"] = df[f"previous_answer_index_{self.column}"].fillna(-99).astype("int16")
        return df



class QuestionLectureTableEncoder(FeatureFactory):
    feature_name_base = "previous_answer"
    question_lecture_dict_path = "../feature_engineering/question_lecture_dict.pickle"
    pickle_path = "../input/feature_engineering/ql_table_{}.pickle"

    def __init__(self,
                 repredict: bool = False,
                 model_id: int = None,
                 question_lecture_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        if question_lecture_dict is None:
            if not os.path.isfile(self.question_lecture_dict_path):
                print("make_new_dict")
                files = glob.glob("../input/riiid-test-answer-prediction/split10/*.pickle")
                df = pd.concat([pd.read_pickle(f).sort_values(["user_id", "timestamp"])[
                                    ["user_id", "content_id", "content_type_id", "answered_correctly"]] for f in files])
                self.make_dict(df)
            with open(self.question_lecture_dict_path, "rb") as f:
                self.question_lecture_dict = pickle.load(f)
        else:
            self.question_lecture_dict = question_lecture_dict
        self.repredict = repredict
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}

    def make_dict(self,
                  df: pd.DataFrame,
                  threshold: int = 50,
                  test_mode: bool = False,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        def f(series, content_id):
            return ((series == content_id).cumsum() > 0)

        if test_mode:
            # test_code用
            lectures = df[df["content_type_id"] == 1]["content_id"].drop_duplicates()
            questions = df[df["content_type_id"] == 0]["content_id"].drop_duplicates()
        else:
            df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                      dtype={"bundle_id": "int32",
                                             "question_id": "int32",
                                             "correct_answer": "int8",
                                             "part": "int8"})
            df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                                     dtype={"lecture_id": "int32",
                                            "tag": "int16",
                                            "part": "int8"})

            lectures = df_lecture["lecture_id"].drop_duplicates().values
            questions = df_question["question_id"].drop_duplicates().values

        ret_dict = {}
        for lecture in tqdm.tqdm(lectures, desc="make_dict..."):
            df["lectured"] = df.groupby(["user_id"])["content_id"].transform(f, **{"content_id": lecture})
            for question, w_df in df[df["content_type_id"] == 0].groupby("content_id"):
                w_dict = w_df.groupby("lectured")["answered_correctly"].mean().to_dict()
                w_dict_size = w_df.groupby("lectured").size()

                if 0 not in w_dict or 1 not in w_dict or len(w_df) < threshold:
                    ret_dict[(question, lecture)] = 0
                elif w_dict_size[0] < threshold or w_dict_size[1] < threshold:
                    ret_dict[(question, lecture)] = 0
                else:
                    score = w_dict[1] - w_dict[0]
                    ret_dict[(question, lecture)] = score

        for lecture in lectures:
            for question in questions:
                if (question, lecture) not in ret_dict:
                    ret_dict[(question, lecture)] = 0

        if output_dir is None:
            output_dir = self.question_lecture_dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        for user_id, w_df in group[["content_type_id", "content_id"]]:
            w_df = w_df[w_df["content_type_id"] == 1]
            if user_id not in self.data_dict:
                self.data_dict[user_id] = w_df["content_id"].values.tolist()
            else:
                self.data_dict[user_id].extend(w_df["content_id"].values.tolist())
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info(f"question_lecture_table_encode")
        def f(series):
            def make_lecture_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    w_ret.append(x)
                    ret.append(w_ret[:])
                return ret

            def calc_score(x):
                list_lectures = x[0]
                content_id = x[1]
                content_type_id = x[2]
                score = 0
                for lec in list_lectures:
                    if content_type_id == 1:
                        return 0
                    if (content_id, lec) in self.question_lecture_dict:
                        score += self.question_lecture_dict[(content_id, lec)]
                return score
            w_df = df.loc[series.index]
            w_df["w_content_id"] = w_df["content_id"] * w_df["content_type_id"] # content_type_id=0: questionは強制的に全部ゼロ
            w_df["list_lectures"] = w_df.groupby("user_id")["w_content_id"].transform(make_lecture_list)
            score = [calc_score(x) for x in w_df[["list_lectures", "content_id", "content_type_id"]].values]
            return score
        pickle_path = self.pickle_path.format(self.model_id)
        self.logger.info(f"ql_score_encoding")
        if not self.repredict and os.path.isfile(pickle_path) and not self.is_debug:
            self.logger.info(f"load_ql_score_pickle")
            with open(pickle_path, "rb") as f:
                w_df = pickle.load(f)
            for col in w_df.columns:
                df[col] = w_df[col].values
        else:
            ql_score = df.groupby("user_id")["content_id"].progress_transform(f).astype("float32")
            df["question_lecture_score"] = ql_score

            # save feature
            w_df = pd.DataFrame()
            if not self.is_debug:
                with open(pickle_path, "wb") as f:
                    w_df["question_lecture_score"] = ql_score
                    pickle.dump(w_df, f)

        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        def calc_score(x):
            user_id = x[0]
            content_id = x[1]
            content_type_id = x[2]

            if content_type_id == 1:
                return 0
            if user_id not in self.data_dict:
                return 0
            lecture_list = self.data_dict[user_id]
            score = np.array([self.question_lecture_dict[(content_id, x)] for x in lecture_list]).sum()
            return score
        df["question_lecture_score"] = [calc_score(x) for x in df[["user_id", "content_id", "content_type_id"]].values]
        df["question_lecture_score"] = df["question_lecture_score"].astype("float32")
        return df


class PreviousLecture(FeatureFactory):
    """
    user_idごとに前回のcontent_idを出す。
    前回がquestion(content_type_id==0)の場合は、ゼロとする

    EDA: 022_previous_contentからinspire
    """

    def fit(self,
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):

        for user_id, w_df in group:
            series = w_df.iloc[-1]
            if series["content_type_id"] == 0:
                self.data_dict[user_id] = None
            else:
                self.data_dict[user_id] = series["content_id"]
        return self


    def all_predict(self,
                    df: pd.DataFrame):
        def f(content_id: int, content_type_id: int):
            if content_type_id == 0:
                return np.nan
            else:
                return content_id

        df_prev = df.groupby("user_id")[["content_id", "content_type_id"]].shift(1)
        df["previous_lecture"] = [f(x[0], x[1]) for x in df_prev.values]
        df["previous_lecture"] = df["previous_lecture"].fillna(-1).astype("int8")
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        """
        リアルタイムfit
        :param df:
        :return:
        """

        def f(user_id, content_id, content_type_id):
            if user_id not in self.data_dict:
                ret = None
            else:
                ret = self.data_dict[user_id]

            # update dict
            if content_type_id == 0:
                self.data_dict[user_id] = None
            else:
                self.data_dict[user_id] = content_id
            return ret

        ret = [f(x[0], x[1], x[2]) for x in df[["user_id", "content_id", "content_type_id"]].values]
        df["previous_lecture"] = ret
        df["previous_lecture"] = df["previous_lecture"].fillna(-1).astype("int8")

        return df

class ContentLevelEncoder(FeatureFactory):
    def __init__(self,
                 vs_column: Union[str, list],
                 initial_score: float =.0,
                 initial_weight: float = 0,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = "content_id"
        self.vs_column = vs_column
        self.initial_score = initial_score
        self.initial_weight = initial_weight
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):
        initial_bunshi = self.initial_score * self.initial_weight

        for key, df in group:
            rate = (df["answered_correctly"] - df[f"target_enc_{self.vs_column}"])
            rate = rate[rate.notnull()]
            if len(rate) == 0:
                continue
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"content_level_{self.vs_column}"] = (df[f"target_enc_{self.vs_column}"].sum() + initial_bunshi) / (len(rate) + self.initial_weight)
                self.data_dict[key][f"content_rate_sum_{self.vs_column}"] = rate.sum()
                self.data_dict[key][f"content_rate_mean_{self.vs_column}"] = rate.mean()
                self.data_dict[key]["count"] = len(rate)
            else:
                content_level = self.data_dict[key][f"content_level_{self.vs_column}"]
                content_rate_sum = self.data_dict[key][f"content_rate_sum_{self.vs_column}"]

                count = self.data_dict[key]["count"] + len(rate) + self.initial_weight

                # パフォーマンス対策:
                # df["answered_correctly"].sum()
                ans_sum = df[f"target_enc_{self.vs_column}"].sum()
                rate_sum = rate.sum()
                self.data_dict[key][f"content_level_{self.vs_column}"] = ((count - len(rate)) * content_level + ans_sum) / count
                self.data_dict[key][f"content_rate_sum_{self.vs_column}"] = content_rate_sum + rate_sum
                self.data_dict[key][f"content_rate_mean_{self.vs_column}"] = (content_rate_sum + rate_sum) / count
                self.data_dict[key]["count"] = self.data_dict[key]["count"] + len(rate)
        return self

    def all_predict(self,
                    df: pd.DataFrame):

        def f_shift1_mean(series):
            bunshi = series.shift(1).notnull().cumsum() + self.initial_weight
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        def f_shift1_sum(series):
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score)

        def f(series):
            bunshi = series.notnull().cumsum() + self.initial_weight
            return (series.fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]
        df[f"content_rate_sum_{self.vs_column}"] = df.groupby("content_id")["rate"].transform(f_shift1_sum).astype("float32")
        df[f"content_rate_mean_{self.vs_column}"] = df.groupby("content_id")["rate"].transform(f_shift1_mean).astype("float32")
        df[f"content_level_{self.vs_column}"] = df.groupby("content_id")[f"target_enc_{self.vs_column}"].transform(f).astype("float32")
        df[f"diff_content_level_target_enc_{self.vs_column}"] = \
            (df[f"content_level_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        df[f"diff_rate_mean_target_emc_{self.vs_column}"] = \
            (df[f"content_rate_mean_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")

        df = df.drop("rate", axis=1)
        return df


    def partial_predict(self,
                        df: pd.DataFrame):
        for col in [f"content_rate_sum_{self.vs_column}",
                    f"content_rate_mean_{self.vs_column}",
                    f"content_level_{self.vs_column}"]:
            df = self._partial_predict2(df, column=col)
        df[f"diff_content_level_target_enc_{self.vs_column}"] = \
            (df[f"content_level_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        df[f"diff_rate_mean_target_emc_{self.vs_column}"] = \
            (df[f"content_rate_mean_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]).astype("float32")
        return df


class FirstColumnEncoder(FeatureFactory):
    feature_name_base = "first_column"
    def __init__(self,
                 column: Union[list, str],
                 astype: str,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        """

        :param column:
        :param split_num:
        :param logger:
        :param is_partial_fit:
        :param is_all_fit:
            fit時のflag. fitは処理時間削減のため通常150行に1回まとめて行うが、そうではなく逐次fitしたいときはTrueを入れる
        """
        self.column = column
        self.astype = astype
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.feature_name_base}_{self.column}"# .replace(" ", "").replace("'", "")

    def fit(self,
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):
        w_dict = group[self.column].first().to_dict()

        for key, value in w_dict.items():
            if key not in self.data_dict:
                self.data_dict[key] = value
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df.groupby("user_id")[self.column].transform("first").astype(self.astype)
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        def f(user_id, value):
            if user_id in self.data_dict:
                return self.data_dict[user_id]
            else:
                self.data_dict[user_id] = value
                return value

        df[self.make_col_name] = [f(x[0], x[1]) for x in df[["user_id", self.column]].values]
        df[self.make_col_name] = df[self.make_col_name].astype(self.astype)
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"


class FirstNAnsweredCorrectly(FeatureFactory):
    feature_name_base = "first_n"
    def __init__(self,
                 n: int,
                 column: str = "user_id",
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        """

        :param column:
        :param split_num:
        :param logger:
        :param is_partial_fit:
        :param is_all_fit:
            fit時のflag. fitは処理時間削減のため通常150行に1回まとめて行うが、そうではなく逐次fitしたいときはTrueを入れる
        """
        self.n = n
        self.column = column
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"first_{n}_ans"

    def fit(self,
            group,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]]):

        for key, w_df in group:
            ans = "".join(w_df["answered_correctly"].fillna(9).astype(int).astype(str).values[:5].tolist())

            if key not in self.data_dict:
                self.data_dict[key] = ans
            elif len(self.data_dict[key]) < 5:
                self.data_dict[key] = self.data_dict[key] + ans
                self.data_dict[key] = self.data_dict[key][:5]

        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f(x):
            rets = [""]
            ret = ""
            for s in x.values[:5]:
                if np.isnan(s):
                    ret += "9"
                else:
                    ret += str(int(s))
                rets.append(ret)
            if len(x) > 5:
                rets.extend([ret] * (len(x) - 6))
            else:
                rets = rets[:len(x)]
            return rets
        df[self.make_col_name] = df.groupby("user_id")["answered_correctly"].progress_transform(f)
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        df[self.make_col_name] = [self.data_dict[x] if x in self.data_dict else ""
                                  for x in df[self.column].values]
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"


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
            is_first_fit: bool = False):
        """

        :param df:
        :param partial_predict_mode:
        :param first_fit: データ取り込み後最初のfitをするときはTrue.
        :return:
        """
        # partial_fit
        df["is_question"] = df["answered_correctly"].notnull().astype("int8")
        for column, dicts in self.feature_factory_dict.items():
            # カラム(ex: user_idなど)ごとに処理
            if column == "postprocess":
                continue
            if type(column) == tuple:
                group = df.groupby(list(column))
            elif type(column) == str:
                group = df.groupby(column)
            else:
                raise ValueError

            for factory in dicts.values():
                if factory.is_partial_fit:
                    if not is_first_fit:
                        df = factory.partial_predict(df)
                    else:
                        df = factory.all_predict(df)
                    self.logger.info(factory)
                    df = factory.make_feature(df)
                    factory.fit(group=group,
                                feature_factory_dict=self.feature_factory_dict)

        # not partial_fit
        for column, dicts in self.feature_factory_dict.items():
            # カラム(ex: user_idなど)ごとに処理
            if column == "postprocess":
                continue
            if type(column) == tuple:
                group = df.groupby(list(column))
            elif type(column) == str:
                group = df.groupby(column)
            else:
                raise ValueError

            for factory in dicts.values():
                if not factory.is_partial_fit:
                    self.logger.info(factory)
                    factory.fit(group=group,
                                feature_factory_dict=self.feature_factory_dict)
        df = df.drop("is_question", axis=1)

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
        # partial_predictあり
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if factory.is_partial_fit:
                    df = factory.all_predict(df=df)

        # partial_predictなし
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if not factory.is_partial_fit:
                    df = factory.all_predict(df=df)
        return df

    def partial_predict(self,
                        df: pd.DataFrame):
        """
        推論時
        :param df:
        :return:
        """
        # partial_predictあり
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if factory.is_partial_fit:
                    self.logger.info(factory)
                    df = factory.partial_predict(df)

        # partial_predictなし
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if not factory.is_partial_fit:
                    self.logger.info(factory)
                    df = factory.partial_predict(df)

        return df

