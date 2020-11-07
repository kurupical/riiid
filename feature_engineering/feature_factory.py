from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import tqdm
from logging import Logger
import time


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

        initial_bunshi = self.initial_score * self.initial_weight
        sum_dict = group["answered_correctly"].sum().to_dict()
        size_dict = group.size().to_dict()

        for key in sum_dict.keys():
            w_sum = sum_dict[key]
            w_size = size_dict[key]

            if key not in self.data_dict:
                self.data_dict[key] = (w_sum + initial_bunshi)/ (w_size + self.initial_weight)
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

                self.data_dict[key] = \
                    ((count - w_size) * target_enc + w_sum) / count
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
        self.logger.info(f"tags_all")

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
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        df["user_count_bin"] = pd.cut(df["count_enc_user_id"], [-1, 30, 10**2, 10**2.5, 10**3, 10**4, 10**5],
                                      labels=False).astype("uint8")
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
        self.logger.info(f"tags_all")

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
        size_dict = group[self.agg_column].size().to_dict()
        for key in sum_dict.keys():
            sum_ = sum_dict[key]
            size_ = size_dict[key]
            if key not in self.data_dict:
                self.data_dict[key] = sum_ / size_
            else:
                # count_encoderの値は更新済のため、
                # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
                count = feature_factory_dict[self.column]["CountEncoder"].data_dict[key]
                target_enc = self.data_dict[key]
                self.data_dict[key] = \
                    ((count - size_) * target_enc + sum_) / count
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
            rate = (df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]).values
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"user_level_{self.vs_column}"] = (df[f"target_enc_{self.vs_column}"].sum() + initial_bunshi) / (len(df) + self.initial_weight)
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = rate.sum()
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = rate.mean()
            else:
                user_level = self.data_dict[key][f"user_level_{self.vs_column}"]
                user_rate_sum = self.data_dict[key][f"user_rate_sum_{self.vs_column}"]

                count = feature_factory_dict["user_id"]["CountEncoder"].data_dict[key] + self.initial_weight

                # パフォーマンス対策:
                # df["answered_correctly"].sum()
                ans_sum = df[f"target_enc_{self.vs_column}"].sum()
                rate_sum = rate.sum()

                self.data_dict[key][f"user_level_{self.vs_column}"] = ((count - len(df)) * user_level + ans_sum) / count
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = user_rate_sum + rate_sum
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = (user_rate_sum + rate_sum) / count
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        def f_shift1_mean(series):
            return (series.shift(1).cumsum() + self.initial_weight * self.initial_score) / (np.arange(len(series)) + self.initial_weight)

        def f_shift1_sum(series):
            return (series.shift(1).cumsum() + self.initial_weight * self.initial_score)

        def f(series):
            return (series.cumsum() + self.initial_weight * self.initial_score) / (np.arange(len(series)) + 1 + self.initial_weight)

        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]
        df[f"user_rate_sum_{self.vs_column}"] = df.groupby("user_id")["rate"].transform(f_shift1_sum).astype("float32")
        df[f"user_rate_mean_{self.vs_column}"] = df.groupby("user_id")["rate"].transform(f_shift1_mean).astype("float32")
        df[f"user_level_{self.vs_column}"] = df.groupby("user_id")[f"target_enc_{self.vs_column}"].transform(f).astype("float32")
        df[f"diff_user_level_target_enc_{self.vs_column}"] = \
            df[f"user_level_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]
        df[f"diff_rate_mean_target_emc_{self.vs_column}"] = \
            df[f"user_rate_mean_{self.vs_column}"] - df[f"target_enc_{self.vs_column}"]

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
                if key not in self.data_dict:
                    self.data_dict[key] = {}
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] = len(w_df)
                    self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = rate.sum()
                    self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = rate.mean()
                elif f"user_rate_sum_{self.agg_column}_{category}" not in self.data_dict[key]:
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] = len(w_df)
                    self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = rate.sum()
                    self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = rate.mean()
                else:
                    self.data_dict[key][f"count_{self.agg_column}_{category}"] += len(w_df)
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
        col_values = df[self.column].values

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

    def fit(self,
            group,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]]):

        last_data_dict = group["answered_correctly"].last().to_dict()
        for key, answer in last_data_dict.items():
            user_id = key[0]
            content_id = key[1]
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                self.data_dict[user_id]["content_id"] = [content_id]
                self.data_dict[user_id]["answered_correctly"] = [answer]
            else:
                self.data_dict[user_id]["content_id"] = [content_id] + self.data_dict[user_id]["content_id"]
                self.data_dict[user_id]["answered_correctly"] = [answer] + self.data_dict[user_id]["answered_correctly"]
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
            ret = []
            for i, content_id in enumerate(series.values):
                ary = series.values[:i][::-1]
                if len(ary) == 0:
                    ret.append(None)
                else:
                    try:
                        argmin = ary.tolist().index(content_id)
                        ret.append(argmin)
                    except ValueError:
                        ret.append(None)
            return ret
        self.logger.info(f"previous_encoding_all_{self.column}")
        df[f"previous_answer_{self.column}"] = df.groupby(self.column)["answered_correctly"].shift(1).fillna(-99).astype("int8")
        df[f"previous_answer_index_{self.column}"] = df.groupby("user_id")["content_id"].transform(f).fillna(-99).astype("int16")

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
            user_id = x[0]
            content_id = x[1]
            if user_id not in self.data_dict:
                return [None, None]
            last_idx = get_index(self.data_dict[user_id]["content_id"], content_id) # listは逆順になっているので
            if last_idx is None: # user_idに対して過去content_idの記録がない
                return [None, None]
            else:
                return [self.data_dict[user_id]["answered_correctly"][last_idx], last_idx]
        ary = [f(x) for x in df[self.column].values]
        ans_ary = [x[0] for x in ary]
        index_ary = [x[1] for x in ary]
        df[f"previous_answer_{self.column}"] = ans_ary
        df[f"previous_answer_{self.column}"] = df[self.make_col_name].fillna(-99).astype("int8")
        df[f"previous_answer_index_{self.column}"] = index_ary
        df[f"previous_answer_index_{self.column}"] = df[f"previous_answer_index_{self.column}"].fillna(-99).astype("int16")
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
            is_first_fit: bool = False):
        """

        :param df:
        :param partial_predict_mode:
        :param first_fit: データ取り込み後最初のfitをするときはTrue.
        :return:
        """
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
                df = factory.make_feature(df)
                factory.fit(group=group,
                            feature_factory_dict=self.feature_factory_dict)
                if factory.is_partial_fit:
                    if not is_first_fit:
                        df = factory.partial_predict(df)
                    else:
                        df = factory.all_predict(df)
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

