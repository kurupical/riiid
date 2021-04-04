from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import tqdm
from logging import Logger
import time
import pickle
import os
import glob
import random
from gensim.models import word2vec
from multiprocessing import Pool, cpu_count

pd.set_option("max_column", 20)
tqdm.tqdm.pandas()
MERGE_FILE_PATH = "../input/riiid-test-answer-prediction/train_merged.pickle"

class FeatureFactory:
    feature_name_base = ""
    def __init__(self,
                 column: Union[list, str],
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
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
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.feature_name_base}_{self.column}"# .replace(" ", "").replace("'", "")

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
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

        pickle_path = f"../input/feature_engineering/{self.make_col_name}/model_id_{self.model_id}.pickle"

        if self.load_feature and os.path.isfile(pickle_path):
            self.logger.info(f"load_feature from {pickle_path}")
            df = self._load_feature(df=df,
                                    pickle_path=pickle_path)
        else:
            original_cols = df.columns
            df = self._all_predict_core(df)
            update_columns = [x for x in df.columns if x not in original_cols]
            if self.save_feature:
                self._save_feature(df,
                                   cols=update_columns,
                                   pickle_path=pickle_path)
        return df

    def _load_feature(self,
                      df: pd.DataFrame,
                      pickle_path: str):
        with open(pickle_path, "rb") as f:
            w_df = pickle.load(f)
        for col in w_df.columns:
            df[col] = w_df[col].values
        return df

    def _save_feature(self,
                      df: pd.DataFrame,
                      cols: List[str],
                      pickle_path: str):
        print("save")
        df_save = pd.DataFrame()
        for col in cols:
            df_save[col] = df[col]
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(df_save, f)

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"

class CountEncoder(FeatureFactory):
    feature_name_base = "count_enc"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        w_dict = df.groupby(self.column).size().to_dict()

        for key, value in w_dict.items():
            if key not in self.data_dict:
                self.data_dict[key] = value
            else:
                self.data_dict[key] += value

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"count_encoding_all_{self.column}")
        df[self.make_col_name] = df.groupby(self.column).cumcount().astype("int32")
        if "user_id" not in self.make_col_name:
            df[self.make_col_name] *= self.split_num

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby_column = groupby_column
        self.agg_column = agg_column
        self.categories = categories
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"groupby_{self.groupby_column}_{self.agg_column}_counter"

    def _fit(self,
             df: pd.DataFrame):
        for key, w_df in df.groupby(self.groupby_column):
            if key not in self.data_dict:
                self.data_dict[key] = {}
                for col in self.categories:
                    self.data_dict[key][col] = 0
            for col, ww_df in w_df.groupby(self.agg_column):
                if col in self.data_dict[key]:
                    self.data_dict[key][col] += len(ww_df)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        if is_first_fit:
            self._fit(df)

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"categories_count_{self.groupby_column}")
        for col in self.categories:
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
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(x, col):
            if x not in self.data_dict:
                return 0
            if col not in self.data_dict[x]:
                return 0
            return self.data_dict[x][col]
        cols = []
        if is_update:
            self._fit(df)
        for col in [-1, 0, 1]:
            col_name = f"groupby_{self.groupby_column}_{self.agg_column}_{col}_count"
            cols.append(col_name)
            df[col_name] = [f(x, col) for x in df[self.groupby_column].values]
            df[col_name] = df[col_name].astype("int32")

        for col in cols:
            df[f"{col}_ratio"] = df[col] / df["count_enc_user_id"]
        return df

    def __repr__(self):
        return self.make_col_name


class TargetEncoder(FeatureFactory):
    feature_name_base = "target_enc"

    def __init__(self,
                 column: Union[list, str],
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 initial_weight: int = 0,
                 initial_score: float = 0,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        super().__init__(column=column,
                         split_num=split_num,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.initial_weight = initial_weight
        self.initial_score = initial_score

    def fit(self,
            df,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        def f(series):
            return series.notnull().sum()
        group = df.groupby(self.column)
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

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(series):
            bunshi = series.shift(1).notnull().cumsum() + self.initial_weight
            return (series.shift(1).fillna(0).cumsum() + self.initial_weight * self.initial_score) / bunshi

        self.logger.info(f"target_encoding_all_{self.column}")

        self.logger.info("shift1...")
        df["shift1"] = df.groupby(self.column)["answered_correctly"].shift(1)
        self.logger.info("notnull...")
        df["notnull"] = df["shift1"].notnull()
        df["shift1"] = df["shift1"].fillna(0)

        df[self.make_col_name] = \
            (df.groupby(self.column)["shift1"].cumsum() + self.initial_weight * self.initial_score) / \
            (df.groupby(self.column)["notnull"].cumsum() + self.initial_weight)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        # df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].transform(f).astype("float32")
        df = df.drop(["shift1", "notnull"], axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        df = self._partial_predict2(df, column=f"target_enc_{self.column}")
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        if self.initial_weight > 0:
            df[self.make_col_name] = df[self.make_col_name].fillna(self.initial_score)
        return df


class TagsSeparator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "tags_separator"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):
        tag = df["tags"].str.split(" ", n=3, expand=True)
        tag.columns = [f"tags{i}" for i in range(1, len(tag.columns) + 1)]

        for col in ["tags1", "tags2"]:
            if col in tag.columns:
                df[col] = pd.to_numeric(tag[col], errors='coerce').fillna(-1).astype("int16")
            else:
                df[col] = -1
                df[col].astype("int16")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class TagsSeparator2(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "tags_separator2"
    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
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

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class TagsTargetEncoder(FeatureFactory):
    tags_dict_path = "../feature_engineering/tags_dict.pickle"

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 tags_dict: Union[Dict[str, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        if tags_dict is None:
            if not os.path.isfile(self.tags_dict_path):
                print("make_new_dict")
                files = glob.glob("../input/riiid-test-answer-prediction/split10/*.pickle")
                df = pd.concat([pd.read_pickle(f)[["tags", "answered_correctly"]] for f in files])
                self.make_dict(df)
            with open(self.tags_dict_path, "rb") as f:
                self.tags_dict = pickle.load(f)
        else:
            self.tags_dict = tags_dict
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = self.__class__.__name__
        self.data_dict = {}

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        ret_dict = {}

        df = df[df["tags"].notnull()]
        df["tags_list"] = [x.split(" ") if type(x) == str else [] for x in df["tags"].values]

        for tag in tqdm.tqdm(np.arange(256)):
            df["is_target"] = [str(tag) in x for x in df["tags_list"].values]
            w_df = df[df["is_target"]]
            if len(w_df) > 0:
                ret_dict[str(tag)] = w_df["answered_correctly"].mean()

        if output_dir is None:
            output_dir = self.tags_dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)
        self.tags_dict = ret_dict

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        def f(ary):
            return [self.tags_dict[x] for x in ary]
        df["tags_te_list"] = [f(x.split(" ")) if type(x) == str else [np.nan] for x in df["tags"].values]
        df["tags_te_mean"] = [np.array(x).mean() for x in df["tags_te_list"].values]
        df["tags_te_max"] = [np.array(x).max() for x in df["tags_te_list"].values]
        df["tags_te_min"] = [np.array(x).min() for x in df["tags_te_list"].values]

        df["tags_te_mean"] = df["tags_te_mean"].astype("float32")
        df["tags_te_max"] = df["tags_te_max"].astype("float32")
        df["tags_te_min"] = df["tags_te_min"].astype("float32")

        return df.drop("tags_te_list", axis=1)

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_encoder_all")
        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PartSeparator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "part_separator"
        
    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        for i in [1, 2, 3, 4, 5, 6, 7]:
            df[f"part{i}"] = (df["part"] == i).astype("int8")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"part_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ContentIdTargetEncoderAggregator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "content_id_target_encode_aggregator"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        cols = [x for x in df.columns if x[:11] == "target_enc_" and "content_id" in x]
        data = df[cols]
        df[f"content_id_te_mean"] = data.mean(axis=1).astype("float32")
        df[f"content_id_te_max"] = data.max(axis=1).astype("float32")
        df[f"content_id_te_min"] = data.min(axis=1).astype("float32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"te_content_id")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class DurationFeaturePostProcess(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "content_id_target_encode_aggregator"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        df["timediff_vs_studytime_userid_priorq"] = (
            df["duration_previous_content_cap100k"] - df["mean_study_time_by_user_id"]
        ).astype("int32")
        df["timediff-elapsedtime"] = (df["duration_previous_content_cap100k"] - df["elapsed_time_content_id_mean"]).astype("int32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"te_content_id")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class TargetEncoderAggregator(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "target_encode_aggregator"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):

        cols = [x for x in df.columns if x[:11] == "target_enc_"]
        data = df[cols]
        df[f"te_mean"] = data.mean(axis=1).astype("float32")
        df[f"te_max"] = data.max(axis=1).astype("float32")
        df[f"te_min"] = data.min(axis=1).astype("float32")
        df[f"te_std"] = data.std(axis=1).astype("float32")
        df[f"te_peak"] = df[f"te_max"] - df[f"te_min"]
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"te_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ListeningReadingEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "is_listening"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):

        df[f"is_listening"] = (df["part"] < 5).astype("uint8")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"te_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class UserCountBinningEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "user_count_binning_encoder"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):

        df["user_count_bin"] = pd.cut(df["count_enc_user_id"], [-1, 30, 10**2, 10**2.5, 10**3, 10**4, 10**5],
                                      labels=False).astype("uint8")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"usercount_bin_all")

        return self._predict(df=df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df=df)

    def __repr__(self):
        return f"{self.__class__.__name__}"



class PriorQuestionElapsedTimeDiv10Encoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = self.__class__.__name__

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        df["prior_question_elapsed_time_div10"] = df["prior_question_elapsed_time"] % 10
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"prior_question_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PriorQuestionElapsedTimeBinningEncoder(FeatureFactory):

    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = self.__class__.__name__

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
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

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"



class TargetEncodeVsUserContentId(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = self.__class__.__name__

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return self._predict(df)

    def _predict(self,
                 df: pd.DataFrame):

        target_cols = [x for x in df.columns if x[:11] == "target_enc_" and "user_id" not in x]

        for col in target_cols:
            col_name = f"diff_target_enc_user_id_vs_{col}"
            df[col_name] = df["target_enc_user_id"] - df[col]
            df[col_name] = df[col_name].astype("float32")

        target_cols = [x for x in df.columns if x[:11] == "target_enc_" and "content_id" not in x]

        for col in target_cols:
            col_name = f"diff_target_enc_content_id_vs_{col}"
            df[col_name] = df["target_enc_content_id"] - df[col]
            df[col_name] = df[col_name].astype("float32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"targetenc_vsuserid")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class MeanAggregator(FeatureFactory):
    feature_name_base = "mean"

    def __init__(self,
                 column: Union[str, list],
                 agg_column: str,
                 remove_now: bool,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] =None,
                 is_partial_fit: bool = False):
        super().__init__(column=column,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.agg_column = agg_column
        self.remove_now = remove_now
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.make_col_name = f"{self.feature_name_base}_{self.agg_column}_by_{self.column}"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        if is_first_fit:
            self._fit(df,
                      feature_factory_dict=feature_factory_dict)
        return self

    def _fit(self,
             df: pd.DataFrame,
             feature_factory_dict: Dict[Union[str, tuple],
                                        Dict[str, FeatureFactory]]):
        group = df[df[self.agg_column].notnull()].groupby(self.column)
        sum_dict = group[self.agg_column].sum().to_dict()
        size_dict = group.size().to_dict()

        for key in sum_dict.keys():
            sum_ = sum_dict[key]
            size_ = size_dict[key]
            if size_ == 0:
                continue

            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][self.make_col_name] = sum_ / size_
                self.data_dict[key]["size"] = size_
            else:
                # count_encoderの値は更新済のため、
                # count = 4, len(df) = 1の場合、もともと3件あって1件が足されたとかんがえる
                count = self.data_dict[key]["size"] + size_
                target_enc = self.data_dict[key][self.make_col_name]
                self.data_dict[key][self.make_col_name] = ((count - size_) * target_enc + sum_) / count
                self.data_dict[key]["size"] = count
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(series):
            if self.remove_now:
                return (series.shift(1).fillna(0).cumsum()) / series.shift(1).notnull().cumsum()
            else:
                return (series.fillna(0).cumsum()) / series.notnull().cumsum()

        self.logger.info(f"{self.feature_name_base}_all_{self.column}_{self.agg_column}")
        df[self.make_col_name] = df.groupby(self.column)[self.agg_column].transform(f).astype("float32")
        if self.remove_now:
            df[f"diff_{self.make_col_name}"] = \
                (df.groupby(self.column)[self.agg_column].shift(1) - df[self.make_col_name]).astype("float32")
        else:
            df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        if is_update:
            self._fit(df,
                      feature_factory_dict={})
        df = self._partial_predict2(df, column=self.make_col_name)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df[f"diff_{self.make_col_name}"] = (df[self.agg_column] - df[self.make_col_name]).astype("float32")
        return df


class UserLevelEncoder2(FeatureFactory):
    def __init__(self,
                 vs_column: Union[str, list],
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 initial_score: float =.0,
                 initial_weight: float = 0,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = "user_id"
        self.vs_column = vs_column
        self.initial_score = initial_score
        self.initial_weight = initial_weight
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}
        self.make_col_name = f"{self.__class__.__name__}_{vs_column}"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        initial_bunshi = self.initial_score * self.initial_weight
        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]
        for key, w_df in df.groupby(self.column):
            rate = w_df["rate"].values
            rate = rate[rate == rate]
            if len(rate) == 0:
                continue
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"user_level_{self.vs_column}"] = (w_df[f"target_enc_{self.vs_column}"].sum() + initial_bunshi) / (len(rate) + self.initial_weight)
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = rate.sum()
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = rate.mean()
                self.data_dict[key]["count"] = len(rate)
            else:
                user_level = self.data_dict[key][f"user_level_{self.vs_column}"]
                user_rate_sum = self.data_dict[key][f"user_rate_sum_{self.vs_column}"]

                count = self.data_dict[key]["count"] + len(rate) + self.initial_weight

                # パフォーマンス対策:
                # df["answered_correctly"].sum()
                ans_sum = w_df[f"target_enc_{self.vs_column}"].sum()
                rate_sum = rate.sum()
                self.data_dict[key][f"user_level_{self.vs_column}"] = ((count - len(rate)) * user_level + ans_sum) / count
                self.data_dict[key][f"user_rate_sum_{self.vs_column}"] = user_rate_sum + rate_sum
                self.data_dict[key][f"user_rate_mean_{self.vs_column}"] = (user_rate_sum + rate_sum) / count
                self.data_dict[key]["count"] = self.data_dict[key]["count"] + len(rate)
        df = df.drop("rate", axis=1)

        return self

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 vs_columns: Union[str, list] = "content_id"):
        self.groupby_column = groupby_column
        self.agg_column = agg_column
        self.categories = categories
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.vs_columns = vs_columns
        self.make_col_name = f"{self.__class__.__name__}_{groupby_column}_{agg_column}_{categories}"


    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_columns}"]
        group = df[df[self.agg_column].isin(self.categories)].groupby([self.groupby_column, self.agg_column])
        for keys, w_df in group[["answered_correctly", f"target_enc_{self.vs_columns}"]]:
            key = keys[0]
            category = keys[1]
            rate = w_df["rate"].values
            rate = rate[rate==rate]
            sum_ = rate.sum()
            size = len(rate)
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"count_{self.agg_column}_{category}"] = size
                self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = sum_
                self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = sum_ / size
            elif f"user_rate_sum_{self.agg_column}_{category}" not in self.data_dict[key]:
                self.data_dict[key][f"count_{self.agg_column}_{category}"] = size
                self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = sum_
                self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = sum_ / size
            else:
                self.data_dict[key][f"count_{self.agg_column}_{category}"] += size
                user_rate_sum = self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"]

                count = self.data_dict[key][f"count_{self.agg_column}_{category}"]

                self.data_dict[key][f"user_rate_sum_{self.agg_column}_{category}"] = user_rate_sum + sum_
                self.data_dict[key][f"user_rate_mean_{self.agg_column}_{category}"] = (user_rate_sum + sum_) / count
        df = df.drop("rate", axis=1)
        return self

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.logger = logger
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"nunique_{self.column}_by_{self.groupby}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):

        group = df.groupby(self.groupby)
        if len(self.data_dict) == 0:
            self.data_dict = group.nunique().to_dict()
            return self
        for key, w_df in group:
            for column in w_df[self.column].drop_duplicates().values:
                if key not in self.data_dict:
                    self.data_dict[key] = [column]
                else:
                    if column not in self.data_dict[key]:
                        self.data_dict[key].append(column)
        return self

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
        df[self.make_col_name] = [len(self.data_dict[x]) if x in self.data_dict else np.nan
                                  for x in df[self.groupby].values]
        df[f"new_ratio_{self.make_col_name}"] = (df[self.make_col_name] / (df[f"count_enc_{self.groupby}"])).astype("float32")
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("int32")
        return df


class SessionEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}
        self.make_col_name = self.__class__.__name__

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        # TODO: 精度upするなら書く
        df["timestamp_diff"] = df["timestamp"] - df.groupby("user_id")["timestamp"].shift(1)
        df["timestamp_diff"] = df["timestamp_diff"].fillna(0).astype("int64")
        df["is_change_session"] = df["change_session"] = (df["timestampdiff"] > 300000).astype("int8")
        df["session"] = df.groupby("user_id")["change_session"].cumsum()
        df["session_nth"] = df.groupby(["user_id", "session"]).cumcount()
        df["first_timestamp"] = df.groupby("user_id").first()
        return self
        for key, w_df in df.groupby("user_id")[["timestamp", "session", "session_nth"]].last():
            last_df = w_df.last()
            first_df = w_df.first()
            timestamp = values[0]
            session = values[1]
            session_nth = values[2]
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key]["timestamp"] = timestamp
                self.data_dict[key]["session"] = session
                self.data_dict[key]["session_nth"] = session_nth
            else:
                pass
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(session, session_nth):
            if session > 0:
                return None
            else:
                return session_nth
        df["timestamp_diff"] = df["timestamp"] - df.groupby("user_id")["timestamp"].shift(1)
        df["timestamp_diff"] = df["timestamp_diff"].fillna(0).astype("int64")

        df["is_change_session"] = df["change_session"] = (df["timestamp_diff"] > 300000).astype("int8")
        df["session"] = df.groupby("user_id")["change_session"].cumsum().fillna(-1).astype("int16")
        df["session_nth"] = df.groupby(["user_id", "session"]).cumcount().fillna(-1).astype("int16")
        df["first_session_nth"] = [f(x[0], x[1]) for x in df[["session", "session_nth"]].values]
        df["first_session_nth"] = df["first_session_nth"].fillna(-1).astype("int16")

        df = df.drop("is_change_session", axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        last_dict = group["answered_correctly"].last().to_dict()
        for key, value in last_dict.items():
            self.data_dict[self._make_key(key)] = value
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"previous_encoding_all_{self.column}")
        df[self.make_col_name] = df.groupby(self.column)["answered_correctly"].shift(1).fillna(-99).astype("int8")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                 n: int = 500,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 is_debug: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.n = n
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.is_debug = is_debug
        self.logger = logger
        self.model_id = model_id
        if is_partial_fit: raise ValueError("can't partialfit=True")
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}
        self.make_col_name = f"{self.__class__.__name__}_{groupby}_{column}_{n}"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        group = df.groupby(self.groupby)
        for user_id, w_df in group[[self.column, "answered_correctly"]]:
            content_id = w_df[self.column].values[::-1].astype("int16")
            answer = w_df["answered_correctly"].values[::-1].astype("int8")
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                self.data_dict[user_id][self.column] = content_id[:self.n]
                self.data_dict[user_id]["answered_correctly"] = answer[:self.n]
            else:
                self.data_dict[user_id][self.column] = np.concatenate([content_id,
                                                                         self.data_dict[user_id][self.column][len(content_id):][:self.n]])
                self.data_dict[user_id]["answered_correctly"] = np.concatenate([answer,
                                                                                self.data_dict[user_id]["answered_correctly"][len(content_id):][:self.n]])
        return self

    def _all_predict_core(self,
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
        prev_answer_index = df.groupby("user_id")[self.column].progress_transform(f).fillna(-99).values
        prev_answer = df.groupby([self.groupby, self.column])["answered_correctly"].shift(1).fillna(-99).values
        df[f"previous_answer_index_{self.column}"] = [x if x < self.n else None for x in prev_answer_index]
        df[f"previous_answer_{self.column}"] = [prev_answer[i] if x < self.n else None for i, x in enumerate(prev_answer_index)]
        df[f"previous_answer_index_{self.column}"] = df[f"previous_answer_index_{self.column}"].fillna(-99).astype("int16")
        df[f"previous_answer_{self.column}"] = df[f"previous_answer_{self.column}"].fillna(-99).astype("int8")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def get_index(l, x):
            try:
                ret = np.where(np.array(l[:self.n]) == x)[0][0]
                return ret
            except IndexError:
                return None

        def f(x):
            """
            index, answered_correctlyを辞書から検索する
            data_dictはリアルタイム更新する。ただし、answered_correctlyはわからないのでnp.nanとしておく
            :param x:
            :param x:
            :return:
            """
            user_id = x[0]
            content_id = x[1]
            if user_id not in self.data_dict:
                if is_update:
                    self.data_dict[user_id] = {}
                    self.data_dict[user_id][self.column] = [content_id]
                    self.data_dict[user_id]["answered_correctly"] = [None]
                    return [None, None]
            last_idx = get_index(self.data_dict[user_id][self.column], content_id) # listは逆順になっているので

            if last_idx is None: # user_idに対して過去content_idの記録がない
                ret = [None, None]
            else:
                ret = [self.data_dict[user_id]["answered_correctly"][last_idx], last_idx]
            if is_update:
                self.data_dict[user_id][self.column] = np.concatenate([[content_id], self.data_dict[user_id][self.column]])
                self.data_dict[user_id]["answered_correctly"] = np.concatenate([[None], self.data_dict[user_id]["answered_correctly"]])
            return ret

        ary = [f(x) for x in df[[self.groupby, self.column]].values]
        ans_ary = [x[0] for x in ary]
        index_ary = [x[1] for x in ary]
        df[f"previous_answer_{self.column}"] = ans_ary
        df[f"previous_answer_{self.column}"] = df[f"previous_answer_{self.column}"].fillna(-99).astype("int8")
        df[f"previous_answer_index_{self.column}"] = index_ary
        df[f"previous_answer_index_{self.column}"] = df[f"previous_answer_index_{self.column}"].fillna(-99).astype("int16")
        return df



class PreviousAnswer3(FeatureFactory):
    feature_name_base = "previous_answer"
    pickle_path = "../input/feature_engineering/previous_answer3_{}.pickle"
    def __init__(self,
                 groupby: str,
                 column: str,
                 n: int = 500,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 is_debug: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.n = n
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.is_debug = is_debug
        self.logger = logger
        self.model_id = model_id
        if is_partial_fit: raise ValueError("can't partialfit=True")
        self.is_partial_fit = is_partial_fit
        self.data_dict = {}
        self.make_col_name = f"{self.__class__.__name__}_{groupby}_{column}_{n}"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        group = df.groupby(self.groupby)
        for user_id, w_df in group[[self.column, "answered_correctly"]]:
            content_id = w_df[self.column].values[::-1].astype("int16")
            answer = w_df["answered_correctly"].values[::-1].astype("int8")
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                self.data_dict[user_id][self.column] = content_id[:self.n]
                self.data_dict[user_id]["answered_correctly"] = answer[:self.n]
            else:
                self.data_dict[user_id][self.column] = np.concatenate([content_id,
                                                                         self.data_dict[user_id][self.column][len(content_id):][:self.n]])
                self.data_dict[user_id]["answered_correctly"] = np.concatenate([answer,
                                                                                self.data_dict[user_id]["answered_correctly"][len(content_id):][:self.n]])
        return self

    def _all_predict_core(self,
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
                    ret.append([-1]*5)
                else:
                    if len(w_ret) < 5:
                        ret.append(w_ret.tolist() + [-1]*(5-len(w_ret)))
                    else:
                        ret.append(w_ret[:5].tolist())
            return ret
        self.logger.info(f"previous_encoding_all_{self.column}")
        prev_answer_index = df.groupby("user_id")[self.column].progress_transform(f).fillna(-1).values

        for idx in range(5):
            prev_answer = df.groupby([self.groupby, self.column])["answered_correctly"].shift(idx+1).fillna(-1).values

            df[f"previous_answer_index_{self.column}_{idx}"] = [x[idx] if x[idx] < self.n else None for x in prev_answer_index]
            df[f"previous_answer_{self.column}_{idx}"] = [prev_answer[i] if x[idx] < self.n else None for i, x in enumerate(prev_answer_index)]
            df[f"previous_answer_index_{self.column}_{idx}"] = df[f"previous_answer_index_{self.column}_{idx}"].fillna(-1).astype("int16")
            df[f"previous_answer_{self.column}_{idx}"] = df[f"previous_answer_{self.column}_{idx}"].fillna(-1).astype("int8")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def get_index(l, x):
            try:
                ret = np.where(np.array(l[:self.n]) == x)[0]
                if len(ret) == 0:
                    return None
                else:
                    return ret[:5]
            except IndexError:
                return None

        def f(x):
            """
            index, answered_correctlyを辞書から検索する
            data_dictはリアルタイム更新する。ただし、answered_correctlyはわからないのでnp.nanとしておく
            :param x:
            :param x:
            :return:
            """
            none = [-1, -1, -1, -1, -1]
            user_id = x[0]
            content_id = x[1]
            if user_id not in self.data_dict:
                if is_update:
                    self.data_dict[user_id] = {}
                    self.data_dict[user_id][self.column] = [content_id]
                    self.data_dict[user_id]["answered_correctly"] = [None]
                    return [none, none]
            ans_idxs = get_index(self.data_dict[user_id][self.column], content_id) # listは逆順になっているので

            if ans_idxs is None: # user_idに対して過去content_idの記録がない
                ret = [none, none]
            else:
                ans_idxs = ans_idxs.tolist()
                ans_corrs = [self.data_dict[user_id]["answered_correctly"][idx] for idx in ans_idxs]
                if len(ans_idxs) < 5:
                    ans_idxs.extend([-1] * (5 - len(ans_idxs)))
                    ans_corrs.extend([-1] * (5 - len(ans_corrs)))
                ret = [ans_idxs, ans_corrs]
            if is_update:
                self.data_dict[user_id][self.column] = np.concatenate([[content_id], self.data_dict[user_id][self.column]])
                self.data_dict[user_id]["answered_correctly"] = np.concatenate([[None], self.data_dict[user_id]["answered_correctly"]])
            return ret

        ary = [f(x) for x in df[[self.groupby, self.column]].values]
        index_ary = np.array([x[0] for x in ary])
        ans_ary = np.array([x[1] for x in ary])

        ans_cols = [f"previous_answer_{self.column}_{i}" for i in range(5)]
        idx_cols = [f"previous_answer_index_{self.column}_{i}" for i in range(5)]

        df[ans_cols] = ans_ary
        df[ans_cols] = df[ans_cols].fillna(-1).astype("int8")

        df[idx_cols] = index_ary
        df[idx_cols] = df[idx_cols].fillna(-1).astype("int16")

        return df




class ShiftDiffEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 groupby: str,
                 column: str,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = column
        self.logger = logger
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"shiftdiff_{self.column}_by_{self.groupby}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        group = df.groupby(self.groupby)
        if len(self.data_dict) == 0:
            self.data_dict = group[self.column].last().to_dict()
        else:
            for key, value in group[self.column].last().to_dict().items():
                self.data_dict[key] = value
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df[self.column] - df.groupby(self.groupby)[self.column].shift(1)
        df[self.make_col_name] = df[self.make_col_name].replace(0, np.nan)
        df[self.make_col_name] = df.groupby(self.groupby)[self.make_col_name].fillna(method="ffill").fillna(0).astype("int64")
        df[f"{self.make_col_name}_cap200k"] = [x if x < 200000 else 200000 for x in df[self.make_col_name].values]
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
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
        df[self.make_col_name] = (df[self.column] - w_diff).replace(0, np.nan)
        df[self.make_col_name] = df.groupby(self.groupby)[self.make_col_name].fillna(method="ffill").fillna(0).astype("int64")
        df[f"{self.make_col_name}_cap200k"] = [x if x < 200000 else 200000 for x in df[self.make_col_name].values]

        if is_update:
            for key, value in df.groupby(self.groupby)[self.column].last().to_dict().items():
                self.data_dict[key] = value
        return df

class QuestionLectureTableEncoder(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_lecture_dict.pickle"

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
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
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = self.__class__.__name__
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
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 1].groupby("user_id")
        for user_id, w_df in group[["content_type_id", "content_id"]]:
            if user_id not in self.data_dict:
                self.data_dict[user_id] = w_df["content_id"].values.tolist()
            else:
                self.data_dict[user_id].extend(w_df["content_id"].values.tolist())
        return self

    def _all_predict_core(self,
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
        self.logger.info(f"ql_score_encoding")
        ql_score = df.groupby("user_id")["content_id"].progress_transform(f).astype("float32")
        df["question_lecture_score"] = ql_score

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
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

class QuestionLectureTableEncoder2(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_lecture2_dict_{}.pickle"

    def __init__(self,
                 past_n: int,
                 min_size: int=1000,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 question_lecture_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_th{self.min_size}_past{self.past_n}"
        self.data_dict = {}
        if question_lecture_dict is None:
            if not os.path.isfile(self.question_lecture_dict_path.format(self.min_size)):
                print("make_new_dict")
                cols = ["user_id", "content_id", "content_type_id", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]

                print("loaded")
                self.make_dict(df)
            with open(self.question_lecture_dict_path.format(self.min_size), "rb") as f:
                self.question_lecture_dict = pickle.load(f)
        else:
            self.question_lecture_dict = question_lecture_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  test_mode: bool = False,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        def f(series, content_id):
            return (series == content_id).cumsum()

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
        df["past_answered"] = (df.groupby(["user_id", "content_id"]).cumcount() > 0).astype("uint8")
        for lecture in tqdm.tqdm(lectures, desc="make_dict..."):
            df["lectured_flg"] = (df["content_type_id"] == 1).astype("uint8") * (df["content_id"] == lecture).astype("uint8")
            df["lectured"] = (df.groupby(["user_id"])["lectured_flg"].cumsum() > 0).astype("uint8")
            # for question, w_df in df[(df["content_type_id"] == 0) & (df["lectured"] == 1)].groupby("content_id"):
            group = df[(df["content_type_id"] == 0) & (df["lectured"] == 1)].groupby(
                ["content_id", "lectured", "past_answered"]
            )["answered_correctly"]
            w_dict_sum = group.sum().to_dict()
            w_dict_size = group.size().to_dict()
            for keys in w_dict_sum.keys():
                if w_dict_size[keys] > self.min_size:
                    question = keys[0]
                    lectured = keys[1]
                    past_answered = keys[2]
                    score = (w_dict_sum[keys] + 0.65 * 30) / (w_dict_size[keys] + 30)
                    ret_dict[(lecture, question, lectured, past_answered)] = score

        """
        for lecture in lectures:
            for question in questions:
                for lectured in [0, 1]:
                    for past_answered in [0, 1]:
                        if (lecture, question, lectured, past_answered) not in ret_dict:
                            ret_dict[(lecture, question, lectured, past_answered)] = 0.65
        """

        if output_dir is None:
            output_dir = self.question_lecture_dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 1].groupby("user_id")
        for user_id, w_df in group[["content_type_id", "content_id"]]:
            if user_id not in self.data_dict:
                self.data_dict[user_id] = w_df["content_id"].values.tolist()[-self.past_n:]
            else:
                update_list = w_df["content_id"].values.tolist()
                self.data_dict[user_id] = self.data_dict[user_id][:-len(update_list)] + w_df["content_id"].values.tolist()
                self.data_dict[user_id] = self.data_dict[user_id][-self.past_n:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"question_lecture_table_encode")
        def f(w_df):
            def make_lecture_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    if not np.isnan(x):
                        w_ret.append(x)
                    ret.append(w_ret[-self.past_n:])
                return ret

            def calc_score(x):
                list_lectures = x[0]
                content_id = x[1]
                content_type_id = x[2]
                if x[3] < 0:
                    past_answer = 0
                else:
                    past_answer = 1

                score = []
                if content_type_id == 1:
                    return [np.nan]
                for lec in list_lectures:
                    if (lec, content_id, 1, past_answer) in self.question_lecture_dict:
                        score.append(self.question_lecture_dict[(lec, content_id, 1, past_answer)])
                return score[-self.past_n:]
            w_df["w_content_id"] = w_df["content_id"] * w_df["content_type_id"].replace(0, np.nan) # content_type_id=0: questionは強制的に全部ゼロ
            w_df["list_lectures"] = w_df.groupby("user_id")["w_content_id"].transform(make_lecture_list)

            score = [calc_score(x) for x in w_df[["list_lectures", "content_id", "content_type_id", "previous_answer_content_id"]].values]
            return score
        self.logger.info(f"ql_score_encoding")

        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            df_ret = pd.DataFrame(index=w_df.index)
            expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
            expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
            expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
            expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
            expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
            df_ret["ql_table2_mean"] = expect_mean
            df_ret["ql_table2_sum"] = expect_sum
            df_ret["ql_table2_max"] = expect_max
            df_ret["ql_table2_min"] = expect_min
            df_ret["ql_table2_last"] = expect_last

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            content_id = x[1]
            content_type_id = x[2]
            if x[3] < 0:
                past_answer = 0
            else:
                past_answer = 1

            if content_type_id == 1:
                if is_update:
                    if user_id in self.data_dict:
                        self.data_dict[user_id].append(content_id)
                        self.data_dict[user_id] = self.data_dict[user_id][-self.past_n:]
                    else:
                        self.data_dict[user_id] = [content_id]
                return [np.nan]
            if user_id not in self.data_dict:
                return [np.nan]
            list_lectures = self.data_dict[user_id]
            score = []
            for lec in list_lectures:
                if (lec, content_id, 1, past_answer) in self.question_lecture_dict:
                    score.append(self.question_lecture_dict[(lec, content_id, 1, past_answer)])
            return score

        score = [calc_score(x) for x in df[["user_id", "content_id", "content_type_id", "previous_answer_content_id"]].values]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
        df["ql_table2_mean"] = expect_mean
        df["ql_table2_sum"] = expect_sum
        df["ql_table2_max"] = expect_max
        df["ql_table2_min"] = expect_min
        df["ql_table2_last"] = expect_last

        df["ql_table2_mean"] = df["ql_table2_mean"].astype("float32")
        df["ql_table2_sum"] = df["ql_table2_sum"].astype("float32")
        df["ql_table2_max"] = df["ql_table2_max"].astype("float32")
        df["ql_table2_min"] = df["ql_table2_min"].astype("float32")
        df["ql_table2_last"] = df["ql_table2_last"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__


class QuestionQuestionTableEncoder(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_question_dict_{}.pickle"

    def __init__(self,
                 past_n: int,
                 min_size: int=1000,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 question_lecture_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_th{self.min_size}"
        self.data_dict = {}
        if question_lecture_dict is None:
            if not os.path.isfile(self.question_lecture_dict_path.format(self.min_size)):
                print("make_new_dict")
                cols = ["user_id", "content_id", "content_type_id", "user_answer", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]

                print("loaded")
                self.make_dict(df)
            with open(self.question_lecture_dict_path.format(self.min_size), "rb") as f:
                self.question_lecture_dict = pickle.load(f)
        else:
            self.question_lecture_dict = question_lecture_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  test_mode: bool = False,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        if test_mode:
            # test_code用
            questions = df[df["content_type_id"] == 0]["content_id"].drop_duplicates()
        else:
            df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                      dtype={"bundle_id": "int32",
                                             "question_id": "int32",
                                             "correct_answer": "int8",
                                             "part": "int8"})

            questions = df_question["question_id"].drop_duplicates().values

        ret_dict = {}
        df["past_answered"] = (df.groupby(["user_id", "content_id"]).cumcount() > 0).astype("uint8")
        df = df[df["content_type_id"] == 0]
        for lecture in tqdm.tqdm(questions, desc="make_dict..."):
            df["lectured_flg"] = (df["content_id"] == lecture).astype("uint8")
            df["lectured"] = (df.groupby(["user_id"])["lectured_flg"].cumsum() > 0).astype("uint8")
            # for question, w_df in df[(df["content_type_id"] == 0) & (df["lectured"] == 1)].groupby("content_id"):
            group = df[(df["content_type_id"] == 0) & (df["lectured"] == 1)].groupby(
                ["content_id", "lectured", "past_answered"]
            )["answered_correctly"]
            w_dict_sum = group.sum().to_dict()
            w_dict_size = group.size().to_dict()
            for keys in w_dict_sum.keys():
                if w_dict_size[keys] > self.min_size:
                    question = keys[0]
                    lectured = keys[1]
                    past_answered = keys[2]
                    if question == lecture:
                        continue
                    score = w_dict_sum[keys] / w_dict_size[keys]
                    ret_dict[(lecture, question, lectured, past_answered)] = score

        if output_dir is None:
            output_dir = self.question_lecture_dict_path.format(self.min_size)
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 0].groupby("user_id")
        for user_id, w_df in group[["content_type_id", "content_id"]]:
            if user_id not in self.data_dict:
                self.data_dict[user_id] = w_df["content_id"].values.tolist()[-self.past_n:]
            else:
                update_list = w_df["content_id"].values.tolist()
                self.data_dict[user_id] = self.data_dict[user_id][:-len(update_list)] + w_df["content_id"].values.tolist()
                self.data_dict[user_id] = self.data_dict[user_id][-self.past_n:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"question_question_table_encode")
        def f(w_df):
            def make_lecture_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    if not np.isnan(x):
                        w_ret.append(x)
                    ret.append(w_ret[-self.past_n:])
                return ret

            def calc_score(x):
                list_lectures = x[0]
                content_id = x[1]
                content_type_id = x[2]
                if x[3] < 0:
                    past_answer = 0
                else:
                    past_answer = 1

                score = []
                if content_type_id == 1:
                    return [np.nan]
                for lec in list_lectures:
                    if (lec, content_id, 1, past_answer) in self.question_lecture_dict:
                        score.append(self.question_lecture_dict[(lec, content_id, 1, past_answer)])
                return score[-self.past_n:]
            w_df["w_content_id"] = w_df["content_id"] * w_df["content_type_id"].replace(1, np.nan).replace(0, 1) # content_type_id=0: questionは強制的に全部ゼロ
            w_df["list_lectures"] = w_df.groupby("user_id")["w_content_id"].transform(make_lecture_list)

            score = [calc_score(x) for x in w_df[["list_lectures", "content_id", "content_type_id", "previous_answer_content_id"]].values]
            return score
        self.logger.info(f"qq_score_encoding")

        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            df_ret = pd.DataFrame(index=w_df.index)
            expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
            expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
            expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
            expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
            expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
            df_ret["qq_table_mean"] = expect_mean
            df_ret["qq_table_sum"] = expect_sum
            df_ret["qq_table_max"] = expect_max
            df_ret["qq_table_min"] = expect_min
            df_ret["qq_table_last"] = expect_last

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            content_id = x[1]
            content_type_id = x[2]
            if x[3] < 0:
                past_answer = 0
            else:
                past_answer = 1

            if content_type_id == 1:
                return [np.nan]
            if is_update:
                if user_id in self.data_dict:
                    self.data_dict[user_id].append(content_id)
                    self.data_dict[user_id] = self.data_dict[user_id][-self.past_n:]
                else:
                    self.data_dict[user_id] = [content_id]
            if user_id not in self.data_dict:
                return [np.nan]
            list_lectures = self.data_dict[user_id]
            score = []
            for lec in list_lectures:
                if (lec, content_id, 1, past_answer) in self.question_lecture_dict:
                    score.append(self.question_lecture_dict[(lec, content_id, 1, past_answer)])
            return score

        score = [calc_score(x) for x in df[["user_id", "content_id", "content_type_id", "previous_answer_content_id"]].values]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
        df["qq_table_mean"] = expect_mean
        df["qq_table_sum"] = expect_sum
        df["qq_table_max"] = expect_max
        df["qq_table_min"] = expect_min
        df["qq_table_last"] = expect_last

        df["qq_table_mean"] = df["qq_table_mean"].astype("float32")
        df["qq_table_sum"] = df["qq_table_sum"].astype("float32")
        df["qq_table_max"] = df["qq_table_max"].astype("float32")
        df["qq_table_min"] = df["qq_table_min"].astype("float32")
        df["qq_table_last"] = df["qq_table_last"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__


class QuestionQuestionTableEncoder2(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_question2_dict_{}.pickle"

    def __init__(self,
                 past_n: int,
                 min_size: int=1000,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 question_lecture_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_th{self.min_size}_past{self.past_n}"
        self.data_dict = {}
        if question_lecture_dict is None:
            if not os.path.isfile(self.question_lecture_dict_path.format(self.min_size)):
                print("make_new_dict")

                cols = ["user_id", "content_id", "content_type_id", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols].head(30_000_000)
                print("loaded")
                self.make_dict(df)
            with open(self.question_lecture_dict_path.format(self.min_size), "rb") as f:
                self.question_lecture_dict = pickle.load(f)
        else:
            self.question_lecture_dict = question_lecture_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  test_mode: bool = False,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        if test_mode:
            # test_code用
            questions = df[df["content_type_id"] == 0]["content_id"].drop_duplicates()
        else:
            df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                      dtype={"bundle_id": "int32",
                                             "question_id": "int32",
                                             "correct_answer": "int8",
                                             "part": "int8"})

            questions = df_question["question_id"].drop_duplicates().values

        ret_dict = {}
        df["past_answered"] = (df.groupby(["user_id", "content_id"]).cumcount() > 0).astype("uint8")
        df = df[df["content_type_id"] == 0]
        for lecture in tqdm.tqdm(questions, desc="make_dict..."):
            for ans in [0, 1]:
                df["lectured_flg"] = \
                    (df["content_id"] == lecture).astype("uint8") * (df["answered_correctly"] == ans).astype("uint8")
                df["lectured"] = df.groupby(["user_id"])["lectured_flg"].shift(1)
                df["lectured"] = (df.groupby("user_id")["lectured"].cumsum() > 0).astype("uint8")
                group = df[df["lectured"] == 1].groupby(["content_id", "past_answered"])["answered_correctly"]
                w_dict_sum = group.sum().to_dict()
                w_dict_size = group.size().to_dict()
                for keys in w_dict_sum.keys():
                    if w_dict_size[keys] > self.min_size:
                        question = keys[0]
                        past_answered = keys[1]
                        answered_correctly = ans
                        score = w_dict_sum[keys] / w_dict_size[keys]
                        ret_dict[(lecture, question, past_answered, answered_correctly)] = score

        if output_dir is None:
            output_dir = self.question_lecture_dict_path.format(self.min_size)
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 0].groupby("user_id")
        for user_id, w_df in group[["content_id", "answered_correctly"]]:
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                for col in ["content_id", "answered_correctly"]:
                    self.data_dict[user_id][col] = w_df[col].values.tolist()[-self.past_n:]
            else:
                for col in ["content_id", "answered_correctly"]:
                    self.data_dict[user_id][col] = (self.data_dict[user_id][col] + w_df[col].values.tolist())[-self.past_n:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"question_question_table_encode")
        def f(w_df):
            def make_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    if not np.isnan(x):
                        w_ret.append(x)
                    ret.append(w_ret[-self.past_n:])
                return ret

            def calc_score(content_id, past_answer, lectures, answered_correctlies):
                score = []
                for lec, answered_correctly in zip(lectures, answered_correctlies):
                    if (lec, content_id, past_answer, answered_correctly) in self.question_lecture_dict:
                        score.append(self.question_lecture_dict[(lec, content_id, past_answer, answered_correctly)])
                return score[-self.past_n:]

            lectures = []
            answered_correctlies = []
            scores = []

            for x in w_df[["content_id", "content_type_id", "previous_answer_content_id", "answered_correctly"]].values:
                content_id = x[0]
                content_type_id = x[1]
                if x[2] < 0:
                    past_answer = 0
                else:
                    past_answer = 1
                answered_correctly = x[3]

                if content_type_id == 1:
                    scores.append([np.nan])
                else:
                    scores.append(calc_score(content_id=content_id,
                                             past_answer=past_answer,
                                             lectures=lectures,
                                             answered_correctlies=answered_correctlies))
                    lectures.append(content_id)
                    lectures = lectures[-self.past_n:]
                    answered_correctlies.append(answered_correctly)
                    answered_correctlies = answered_correctlies[-self.past_n:]

            return scores
        self.logger.info(f"qq_score2_encoding")

        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            df_ret = pd.DataFrame(index=w_df.index)
            expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
            expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
            expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
            expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
            expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
            df_ret["qq_table2_mean"] = expect_mean
            df_ret["qq_table2_sum"] = expect_sum
            df_ret["qq_table2_max"] = expect_max
            df_ret["qq_table2_min"] = expect_min
            df_ret["qq_table2_last"] = expect_last

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            content_id = x[1]
            content_type_id = x[2]
            if x[3] < 0:
                past_answer = 0
            else:
                past_answer = 1

            if content_type_id == 1:
                return [np.nan]
            if user_id not in self.data_dict:
                return [np.nan]
            list_lectures = self.data_dict[user_id]["content_id"]
            list_answered_correctly = self.data_dict[user_id]["answered_correctly"]
            score = []
            for lec, answered_correctly in zip(list_lectures, list_answered_correctly):
                if (lec, content_id, past_answer, answered_correctly) in self.question_lecture_dict:
                    score.append(self.question_lecture_dict[(lec, content_id, past_answer, answered_correctly)])
            return score

        score = [calc_score(x) for x in df[["user_id", "content_id", "content_type_id", "previous_answer_content_id"]].values]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
        df["qq_table2_mean"] = expect_mean
        df["qq_table2_sum"] = expect_sum
        df["qq_table2_max"] = expect_max
        df["qq_table2_min"] = expect_min
        df["qq_table2_last"] = expect_last

        df["qq_table2_mean"] = df["qq_table2_mean"].astype("float32")
        df["qq_table2_sum"] = df["qq_table2_sum"].astype("float32")
        df["qq_table2_max"] = df["qq_table2_max"].astype("float32")
        df["qq_table2_min"] = df["qq_table2_min"].astype("float32")
        df["qq_table2_last"] = df["qq_table2_last"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__


class QuestionQuestionTableEncoder3(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_question3_dict_{}.pickle"

    def __init__(self,
                 past_n: int,
                 min_size: int=1000,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 question_lecture_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_th{self.min_size}_past{self.past_n}"
        self.data_dict = {}
        if question_lecture_dict is None:
            if not os.path.isfile(self.question_lecture_dict_path.format(self.min_size)):
                print("make_new_dict")

                cols = ["user_id", "content_id", "content_type_id", "user_answer", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols].head(20_000_000)
                print("loaded")
                self.make_dict(df)
            with open(self.question_lecture_dict_path.format(self.min_size), "rb") as f:
                self.question_lecture_dict = pickle.load(f)
        else:
            self.question_lecture_dict = question_lecture_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  test_mode: bool = False,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        if test_mode:
            # test_code用
            questions = df[df["content_type_id"] == 0]["content_id"].drop_duplicates()
        else:
            df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                      dtype={"bundle_id": "int32",
                                             "question_id": "int32",
                                             "correct_answer": "int8",
                                             "part": "int8"})

            questions = df_question["question_id"].drop_duplicates().values

        ret_dict = {}
        df["past_answered"] = (df.groupby(["user_id", "content_id"]).cumcount() > 0).astype("uint8")
        df = df[df["content_type_id"] == 0]
        for lecture in tqdm.tqdm(questions, desc="make_dict..."):
            for ans in [1, 2, 3, 4]:
                df["lectured_flg"] = \
                    (df["content_id"] == lecture).astype("uint8") * (df["user_answer"] == ans).astype("uint8")
                df["lectured"] = df.groupby(["user_id"])["lectured_flg"].shift(1)
                df["lectured"] = (df.groupby("user_id")["lectured"].cumsum() > 0).astype("uint8")
                group = df[df["lectured"] == 1].groupby(["content_id", "past_answered"])["answered_correctly"]
                w_dict_sum = group.sum().to_dict()
                w_dict_size = group.size().to_dict()
                for keys in w_dict_sum.keys():
                    if w_dict_size[keys] > self.min_size:
                        question = keys[0]
                        past_answered = keys[1]
                        answered_correctly = ans
                        score = w_dict_sum[keys] / w_dict_size[keys]
                        ret_dict[(lecture, question, past_answered, answered_correctly)] = score

        if output_dir is None:
            output_dir = self.question_lecture_dict_path.format(self.min_size)
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 0].groupby("user_id")
        for user_id, w_df in group[["content_id", "user_answer"]]:
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                for col in ["content_id", "user_answer"]:
                    self.data_dict[user_id][col] = w_df[col].values.tolist()[-self.past_n:]
            else:
                for col in ["content_id", "user_answer"]:
                    self.data_dict[user_id][col] = (self.data_dict[user_id][col] + w_df[col].values.tolist())[-self.past_n:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"question_question_table_encode")
        def f(w_df):
            def make_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    if not np.isnan(x):
                        w_ret.append(x)
                    ret.append(w_ret[-self.past_n:])
                return ret

            def calc_score(content_id, past_answer, lectures, answered_correctlies):
                score = []
                for lec, answered_correctly in zip(lectures, answered_correctlies):
                    if (lec, content_id, past_answer, answered_correctly) in self.question_lecture_dict:
                        score.append(self.question_lecture_dict[(lec, content_id, past_answer, answered_correctly)])
                return score[-self.past_n:]

            lectures = []
            answered_correctlies = []
            scores = []

            for x in w_df[["content_id", "content_type_id", "previous_answer_content_id", "user_answer"]].values:
                content_id = x[0]
                content_type_id = x[1]
                if x[2] < 0:
                    past_answer = 0
                else:
                    past_answer = 1
                answered_correctly = x[3]

                if content_type_id == 1:
                    scores.append([np.nan])
                else:
                    scores.append(calc_score(content_id=content_id,
                                             past_answer=past_answer,
                                             lectures=lectures,
                                             answered_correctlies=answered_correctlies))
                    lectures.append(content_id)
                    lectures = lectures[-self.past_n:]
                    answered_correctlies.append(answered_correctly)
                    answered_correctlies = answered_correctlies[-self.past_n:]

            return scores
        self.logger.info(f"qq_score3_encoding")

        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            df_ret = pd.DataFrame(index=w_df.index)
            expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
            expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
            expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
            expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
            expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
            df_ret["qq_table3_mean"] = expect_mean
            df_ret["qq_table3_sum"] = expect_sum
            df_ret["qq_table3_max"] = expect_max
            df_ret["qq_table3_min"] = expect_min
            df_ret["qq_table3_last"] = expect_last

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            content_id = x[1]
            content_type_id = x[2]
            if x[3] < 0:
                past_answer = 0
            else:
                past_answer = 1

            if content_type_id == 1:
                return [np.nan]
            if user_id not in self.data_dict:
                return [np.nan]
            list_lectures = self.data_dict[user_id]["content_id"]
            list_answered_correctly = self.data_dict[user_id]["user_answer"]
            score = []
            for lec, answered_correctly in zip(list_lectures, list_answered_correctly):
                if (lec, content_id, past_answer, answered_correctly) in self.question_lecture_dict:
                    score.append(self.question_lecture_dict[(lec, content_id, past_answer, answered_correctly)])
            return score

        score = [calc_score(x) for x in df[["user_id", "content_id", "content_type_id", "previous_answer_content_id"]].values]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
        df["qq_table3_mean"] = expect_mean
        df["qq_table3_sum"] = expect_sum
        df["qq_table3_max"] = expect_max
        df["qq_table3_min"] = expect_min
        df["qq_table3_last"] = expect_last

        df["qq_table3_mean"] = df["qq_table3_mean"].astype("float32")
        df["qq_table3_sum"] = df["qq_table3_sum"].astype("float32")
        df["qq_table3_max"] = df["qq_table3_max"].astype("float32")
        df["qq_table3_min"] = df["qq_table3_min"].astype("float32")
        df["qq_table3_last"] = df["qq_table3_last"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__


class PreviousLecture(FeatureFactory):
    """
    user_idごとに前回のcontent_idを出す。
    前回がquestion(content_type_id==0)の場合は、ゼロとする

    EDA: 022_previous_contentからinspire
    """

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):

        group = df.groupby("user_id")
        for user_id, w_df in group:
            series = w_df.iloc[-1]
            if series["content_type_id"] == 0:
                self.data_dict[user_id] = None
            else:
                self.data_dict[user_id] = series["content_id"]
        return self


    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
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
            if is_update:
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
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.column = "content_id"
        self.vs_column = vs_column
        self.initial_score = initial_score
        self.initial_weight = initial_weight
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.__class__.__name__}_{vs_column}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        initial_bunshi = self.initial_score * self.initial_weight
        df["rate"] = df["answered_correctly"] - df[f"target_enc_{self.vs_column}"]
        group = df.groupby(self.column)
        for key, w_df in group:
            rate = w_df["rate"].values
            rate = rate[rate==rate]
            if len(rate) == 0:
                continue
            if key not in self.data_dict:
                self.data_dict[key] = {}
                self.data_dict[key][f"content_level_{self.vs_column}"] = (w_df[f"target_enc_{self.vs_column}"].sum() + initial_bunshi) / (len(rate) + self.initial_weight)
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
        df = df.drop("rate", axis=1)
        return self

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
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
                 agg_column: str,
                 astype: str,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
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
        self.column = "user_id"
        self.agg_column = agg_column
        self.astype = astype
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"{self.feature_name_base}_{self.agg_column}"# .replace(" ", "").replace("'", "")

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        group = df.groupby(self.column)
        w_dict = group[self.agg_column].first().to_dict()

        for key, value in w_dict.items():
            if key not in self.data_dict:
                self.data_dict[key] = value
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df.groupby("user_id")[self.agg_column].transform("first").astype(self.astype)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, value):
            if user_id in self.data_dict:
                return self.data_dict[user_id]
            else:
                if is_update:
                    self.data_dict[user_id] = value
                return value

        df[self.make_col_name] = [f(x[0], x[1]) for x in df[[self.column, self.agg_column]].values]
        df[self.make_col_name] = df[self.make_col_name].astype(self.astype)
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"


class FirstNAnsweredCorrectly(FeatureFactory):
    feature_name_base = "first_n"
    def __init__(self,
                 n: int,
                 column: str = "user_id",
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
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
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"first_{n}_ans"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):

        group = df.groupby(self.column)
        for key, w_df in group:
            ans = "".join(w_df["answered_correctly"].fillna(9).astype(int).astype(str).values[:5].tolist())

            if key not in self.data_dict:
                self.data_dict[key] = ans
            elif len(self.data_dict[key]) < 5:
                self.data_dict[key] = self.data_dict[key] + ans
                self.data_dict[key] = self.data_dict[key][:5]

        return self

    def _all_predict_core(self,
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
                        df: pd.DataFrame,
                        is_update: bool=True):
        df[self.make_col_name] = [self.data_dict[x] if x in self.data_dict else ""
                                  for x in df[self.column].values]
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"


def simple_diff(rate1, rate2):
    """ rate1: lose player's rate, rate2: win player's rate """
    diff = 16 + int((rate1 - rate2) * 0.04)
    return -diff, diff

def elo_rating(rate1, rate2, k=15):
    """ rate1: lose player's rate, rate2: win player's rate """
    expect_win_1 = 1 / (1 + 10**(-(rate1 - rate2)/400))
    diff_1 = k * (0 - expect_win_1)

    expect_win_2 = 1 / (1 + 10**(-(rate2 - rate1)/400))
    diff_2 = k * (1 - expect_win_2)

    return diff_1, diff_2

class UserContentRateEncoder(FeatureFactory):
    feature_name_base = "user_content_rate_encoder"

    def __init__(self,
                 rate_func: str,
                 column: Union[list, str],
                 initial_rate: int = 1500,
                 content_rate_dict: dict = None,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        """

        :param rate_func:
            func(rate1, rate2) return update_diff
        :param column:
        :param content_rate_dict:
        :param model_id:
        :param load_feature:
        :param save_feature:
        :param split_num:
        :param logger:
        :param is_partial_fit:
        """
        super().__init__(column=column,
                         split_num=split_num,
                         logger=logger,
                         is_partial_fit=is_partial_fit)

        if rate_func == "simple":
            self.rate_func = simple_diff
        if rate_func == "elo":
            self.rate_func = elo_rating
        self.initial_rate = initial_rate
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.content_rate_dict = content_rate_dict
        self.dict_path = f"../feature_engineering/content_{self.column}_{rate_func}_rate_dict.pickle"
        self.make_col_name = f"{self.feature_name_base}_{self.column}_{rate_func}_rate_dict.pickle"
        if self.content_rate_dict is None:
            if os.path.isfile(self.dict_path):
                with open(self.dict_path, "rb") as f:
                    self.content_rate_dict = pickle.load(f)
        else:
            self.content_rate_dict = self.content_rate_dict
    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str=None):

        w_df = df[df["content_type_id"] == 0]
        if type(self.column) == str:
            w_df = w_df[["content_id", "answered_correctly", self.column]]
        else:
            w_df = w_df[["content_id", "answered_correctly"] + self.column]
        content_dict = {}
        user_dict = {}

        for x in tqdm.tqdm(w_df.values):
            content_id = x[0]
            answered_correctly = x[1]
            if type(self.column) == str:
                user_keys = x[2]
            else:
                user_keys = tuple(x[2:])

            if content_id not in content_dict:
                content_dict[content_id] = 1500

            if user_keys not in user_dict:
                user_dict[user_keys] = 1500

            user_rate = user_dict[user_keys]
            content_rate = content_dict[content_id]
            if answered_correctly == 0:
                diff_1, diff_2 = self.rate_func(user_rate, content_rate)
                user_dict[user_keys] += diff_1
                content_dict[content_id] += diff_2

            if answered_correctly == 1:
                diff_1, diff_2 = self.rate_func(content_rate, user_rate)
                user_dict[user_keys] += diff_2
                content_dict[content_id] += diff_1

        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(content_dict, f)

    def update_dict(self,
                    user_dict: Dict[int, int],
                    user_key: Union[str, tuple],
                    content_id: str,
                    answered_correctly: int):

        if user_key not in user_dict:
            if self.initial_rate is None:
                user_dict[user_key] = self.content_rate_dict[content_id]
            else:
                user_dict[user_key] = self.initial_rate
        user_rate = user_dict[user_key]
        content_rate = self.content_rate_dict[content_id]

        if answered_correctly == 0:
            diff_1, diff_2 = self.rate_func(user_rate, content_rate)
            user_dict[user_key] += diff_1

        if answered_correctly == 1:
            diff_1, diff_2 = self.rate_func(content_rate, user_rate)
            user_dict[user_key] += diff_2

    def fit(self,
            df,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        columns = []
        if type(self.column) == str:
            columns = ["content_id", "answered_correctly", self.column]
        if type(self.column) == list:
            columns = ["content_id", "answered_correctly"] + self.column

        for x in df[df["content_type_id"] == 0][columns].values:
            content_id = x[0]
            answered_correctly = x[1]
            if type(self.column) == str:
                user_key = x[2]
            else:
                user_key = tuple(x[2:])
            self.update_dict(user_dict=self.data_dict,
                             user_key=user_key,
                             content_id=content_id,
                             answered_correctly=answered_correctly)

        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"user_rating {self.column}")

        if self.content_rate_dict is None:
            if not os.path.isfile(self.dict_path):
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.content_rate_dict = pickle.load(f)
        else:
            self.content_rate_dict = self.content_rate_dict

        user_dict = {}
        def f(user_key, content_id, content_type_id, answered_correctly):
            if content_type_id == 1:
                return -1

            if user_key not in user_dict:
                if self.initial_rate is None:
                    rate = self.content_rate_dict[content_id]
                else:
                    rate = self.initial_rate
                user_dict[user_key] = rate
                ret = rate
            else:
                ret = user_dict[user_key]

            self.update_dict(user_dict=user_dict,
                             user_key=user_key,
                             content_id=content_id,
                             answered_correctly=answered_correctly)
            return ret

        df[f"content_rating"] = [self.content_rate_dict[x[0]] if x[1] == 0 else -1
                                               for x in df[["content_id", "content_type_id"]].values]
        if type(self.column) == str:
            keys = df[self.column].values
        else:
            keys = [tuple(x) for x in df[self.column].values]
        df[f"{self.column}_rating"] = [f(key, x[0], x[1], x[2]) for key, x in zip(keys, df[["content_id", "content_type_id", "answered_correctly"]].values)]

        df[f"content_rating"] = df[f"content_rating"].astype("int16")
        df[f"{self.column}_rating"] = df[f"{self.column}_rating"].astype("int16")
        df[f"rating_diff_content_{self.column}"] = df[f"content_rating"] - df[f"{self.column}_rating"]
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, content_id, content_type_id):
            if content_type_id == 1:
                return -1
            if user_id in self.data_dict:
                return self.data_dict[user_id]
            else:
                if self.initial_rate is None:
                    return self.content_rate_dict[content_id]
                else:
                    return 1500

        if type(self.column) == str:
            keys = df[self.column].values
        else:
            keys = [tuple(x) for x in df[self.column].values]

        df[f"{self.column}_rating"] = [f(key, x[0], x[1]) for key, x in zip(keys, df[["content_id", "content_type_id"]].values)]
        df[f"content_rating"] = [self.content_rate_dict[x[0]] if x[1] == 0 else -1
                                for x in df[["content_id", "content_type_id"]].values]
        df[f"content_rating"] = df[f"content_rating"].astype("int16")
        df[f"{self.column}_rating"] = df[f"{self.column}_rating"].astype("int16")
        df[f"rating_diff_content_{self.column}"] = df[f"content_rating"] - df[f"{self.column}_rating"]

        return df


class UserContentNowRateEncoder(FeatureFactory):
    feature_name_base = "user_content_now_rate_encoder"

    def __init__(self,
                 rate_func: str,
                 column: Union[list, str],
                 target: list,
                 content_rate_dict: dict = None,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        """

        :param rate_func:
            func(rate1, rate2) return update_diff
        :param column:
        :param content_rate_dict:
        :param model_id:
        :param load_feature:
        :param save_feature:
        :param split_num:
        :param logger:
        :param is_partial_fit:
        """
        super().__init__(column=column,
                         split_num=split_num,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.target = target
        if rate_func == "simple":
            self.rate_func = simple_diff
        if rate_func == "elo":
            self.rate_func = elo_rating
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.content_rate_dict = content_rate_dict
        self.dict_path = f"../feature_engineering/content_['user_id', 'part']_{rate_func}_rate_dict.pickle"
        self.make_col_name = f"{self.feature_name_base}_{self.column}_{rate_func}_rate_dict_{self.target}"
        if self.content_rate_dict is None:
            if os.path.isfile(self.dict_path):
                with open(self.dict_path, "rb") as f:
                    self.content_rate_dict = pickle.load(f)
        else:
            self.content_rate_dict = content_rate_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str=None):

        w_df = df[df["content_type_id"] == 0]
        if type(self.column) == str:
            w_df = w_df[["content_id", "answered_correctly", self.column]]
        else:
            w_df = w_df[["content_id", "answered_correctly"] + self.column]
        content_dict = {}
        user_dict = {}

        for x in tqdm.tqdm(w_df.values):
            content_id = x[0]
            answered_correctly = x[1]
            if type(self.column) == str:
                user_keys = x[2]
            else:
                user_keys = tuple(x[2:])

            if content_id not in content_dict:
                content_dict[content_id] = 1500

            if user_keys not in user_dict:
                user_dict[user_keys] = 1500

            user_rate = user_dict[user_keys]
            content_rate = content_dict[content_id]
            if answered_correctly == 0:
                diff_1, diff_2 = self.rate_func(user_rate, content_rate)
                user_dict[user_keys] += diff_1
                content_dict[content_id] += diff_2

            if answered_correctly == 1:
                diff_1, diff_2 = self.rate_func(content_rate, user_rate)
                user_dict[user_keys] += diff_2
                content_dict[content_id] += diff_1

        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(content_dict, f)

    def update_dict(self,
                    user_dict: Dict[int, int],
                    user_key: Union[str, tuple],
                    target: Union[str, int],
                    content_id: str,
                    answered_correctly: int):

        if user_key not in user_dict:
            user_dict[user_key] = {}
        if target not in user_dict[user_key]:
            user_dict[user_key][target] = 1500

        user_rate = user_dict[user_key][target]
        content_rate = self.content_rate_dict[content_id]

        if answered_correctly == 0:
            diff_1, diff_2 = self.rate_func(user_rate, content_rate)
            user_dict[user_key][target] += diff_1

        if answered_correctly == 1:
            diff_1, diff_2 = self.rate_func(content_rate, user_rate)
            user_dict[user_key][target] += diff_2

    def fit(self,
            df,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        columns = ["content_id", "answered_correctly", "user_id", self.column]

        for target in self.target:
            for x in df[(df["content_type_id"] == 0) & (df[self.column] == target)][columns].values:
                content_id = x[0]
                answered_correctly = x[1]
                user_key = x[2]
                target_value = x[3]

                self.update_dict(user_dict=self.data_dict,
                                 user_key=user_key,
                                 target=target_value,
                                 content_id=content_id,
                                 answered_correctly=answered_correctly)

        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"user_rating {self.column}")

        if self.content_rate_dict is None:
            if not os.path.isfile(self.dict_path):
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.content_rate_dict = pickle.load(f)
        else:
            self.content_rate_dict = self.content_rate_dict

        user_dict = {}
        def f(user_key, target, content_id, content_type_id, answered_correctly, target_value):
            if target != target_value:
                if user_key in user_dict:
                    if target in user_dict[user_key]:
                        return user_dict[user_key][target]
                    else:
                        return 1500
                else:
                    return 1500

            if content_type_id == 1:
                return -1

            if user_key not in user_dict:
                user_dict[user_key] = {}
                user_dict[user_key][target] = 1500
                ret = 1500
            elif target not in user_dict[user_key]:
                user_dict[user_key][target] = 1500
                ret = 1500
            else:
                ret = user_dict[user_key][target]

            self.update_dict(user_dict=user_dict,
                             user_key=user_key,
                             target=target,
                             content_id=content_id,
                             answered_correctly=answered_correctly)
            return ret

        df[f"content_rating"] = [self.content_rate_dict[x[0]] if x[1] == 0 else -1
                                               for x in df[["content_id", "content_type_id"]].values]
        keys = df["user_id"].values

        df[f"content_rating"] = df[f"content_rating"].astype("int16")
        for t in tqdm.tqdm(self.target):
            df[f"{self.column}{t}_rating"] = [f(key, t, x[0], x[1], x[2], x[3]) for key, x in zip(keys, df[["content_id", "content_type_id", "answered_correctly", self.column]].values)]
            df[f"{self.column}{t}_rating"] = df[f"{self.column}{t}_rating"].astype("int16")
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, content_id, content_type_id, target_value, target):
            if content_type_id == 1:
                return -1

            if target != target_value:
                if user_id in self.data_dict:
                    if target in self.data_dict[user_id]:
                        return self.data_dict[user_id][target]
                    else:
                        return 1500
                else:
                    return 1500

            if content_type_id == 1:
                return -1
            if user_id in self.data_dict:
                if target in self.data_dict[user_id]:
                    return self.data_dict[user_id][target]
                else:
                    return 1500
            else:
                return 1500

        keys = df[self.column].values

        df[f"content_rating"] = [self.content_rate_dict[x[0]] if x[1] == 0 else -1
                                 for x in df[["content_id", "content_type_id"]].values]
        df[f"content_rating"] = df[f"content_rating"].astype("int16")

        for t in self.target:
            df[f"{self.column}{t}_rating"] = [f(x[0], x[1], x[2], x[3], t) for x in df[["user_id", "content_id", "content_type_id", self.column]].values]
            df[f"{self.column}{t}_rating"] = df[f"{self.column}{t}_rating"].astype("int16")

        return df


class UserRateBinningEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "user_rate_binning"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):
        df["uc_rate"] = (df["user_id_rating"] // 100).astype(str) + "_" + (df["content_rating"] // 100).astype(str) + df["part"].astype(str)

        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"



class UserAnswerLevelEncoder(FeatureFactory):
    user_answer_dict_path = "../feature_engineering/user_answer_dict.pickle"

    def __init__(self,
                 past_n: int,
                 min_size: int=100,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 user_answer_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_past{self.past_n}"
        self.data_dict = {}
        if user_answer_dict is None:
            if not os.path.isfile(self.user_answer_dict_path):
                print("make_new_dict")

                cols = ["user_id", "content_id", "content_type_id", "user_answer", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]
                print("loaded")
                self.make_dict(df)
            with open(self.user_answer_dict_path, "rb") as f:
                self.user_answer_dict = pickle.load(f)
        else:
            self.user_answer_dict = user_answer_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """
        class EmptyLogger:
            def __init__(self):
                pass
            def info(self, x):
                pass

        df = df[["user_id", "content_id", "content_type_id", "user_answer", "answered_correctly"]]
        df = TargetEncoder(column="user_id", logger=EmptyLogger()).all_predict(df[df["content_type_id"] == 0])
        df = df[df["target_enc_user_id"].notnull()]
        sum_dict = df.groupby(["content_id", "user_answer"])["target_enc_user_id"].sum().to_dict()
        size_dict = df.groupby(["content_id", "user_answer"])["target_enc_user_id"].size().to_dict()
        ret_dict = {}
        for k in sum_dict.keys():
            ret_dict[k] = (sum_dict[k] + 0.65*30) / (size_dict[k] + 30)

        if output_dir is None:
            output_dir = self.user_answer_dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df[df["content_type_id"] == 0].groupby("user_id")
        for user_id, w_df in group:
            w_df = w_df[["content_id", "user_answer"]]
            if user_id not in self.data_dict:
                self.data_dict[user_id] = [tuple(x) for x in w_df.values][-self.past_n:]
            else:
                update_list = [tuple(x) for x in w_df.values]
                self.data_dict[user_id] = (self.data_dict[user_id] + update_list)[-self.past_n:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"useranswer")
        def f(w_df):
            def make_lecture_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    if type(x) == tuple: # content_type_id=1はnp.nan, content_type_id=0はTuple(content_id, user_answer)
                        w_ret.append(x)
                    ret.append(w_ret[-self.past_n:])
                return ret

            def calc_score(x):
                list_keys = x[0]

                score = []
                for key in list_keys:
                    if key in self.user_answer_dict:
                        score.append(self.user_answer_dict[key])
                return score[-self.past_n:]
            w_df["w_content_id"] = w_df["content_id"] * w_df["content_type_id"].replace(1, np.nan).replace(0, 1) # content_type_id=0: questionは強制的に全部ゼロ
            w_df["key"] = [tuple(x) if not np.isnan(x[0]) else np.nan for x in w_df[["w_content_id", "user_answer"]].values]
            w_df["list_keys"] = w_df.groupby("user_id")["key"].transform(make_lecture_list)

            score = [calc_score(x) for x in w_df[["list_keys"]].values]
            return score

        self.logger.info(f"content_ua_score_encoding")

        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            score = [[np.nan]] + score[:-1] # shift(1)
            df_ret = pd.DataFrame(index=w_df.index)
            expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
            expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
            expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
            expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
            expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
            df_ret["content_ua_table2_mean"] = expect_mean
            df_ret["content_ua_table2_sum"] = expect_sum
            df_ret["content_ua_table2_max"] = expect_max
            df_ret["content_ua_table2_min"] = expect_min
            df_ret["content_ua_table2_last"] = expect_last

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            content_type_id = x[1]

            score = []
            if content_type_id == 1:
                return [np.nan]
            if not user_id in self.data_dict:
                return [np.nan]

            for key in self.data_dict[user_id]:
                if key in self.user_answer_dict:
                    score.append(self.user_answer_dict[tuple(key)])
            return score[-self.past_n:]

        score = [calc_score(x) for x in df[["user_id", "content_type_id"]].values]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] if len(x) > 0 else np.nan for x in score]
        df["content_ua_table2_mean"] = expect_mean
        df["content_ua_table2_sum"] = expect_sum
        df["content_ua_table2_max"] = expect_max
        df["content_ua_table2_min"] = expect_min
        df["content_ua_table2_last"] = expect_last

        df["content_ua_table2_mean"] = df["content_ua_table2_mean"].astype("float32")
        df["content_ua_table2_sum"] = df["content_ua_table2_sum"].astype("float32")
        df["content_ua_table2_max"] = df["content_ua_table2_max"].astype("float32")
        df["content_ua_table2_min"] = df["content_ua_table2_min"].astype("float32")
        df["content_ua_table2_last"] = df["content_ua_table2_last"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__

class PreviousNAnsweredCorrectly(FeatureFactory):
    """
    過去N回のanswered_correctlyを文字列として結合する
    ただし:
    bundle_idが同じものは,最後のbundle_idが終わるまで答えがわからないので、8
    partial_predictで答え未知の場合は、8
    lecture(content_type_id=1)の場合は、9
    をそれぞれ設定する。
    """
    def __init__(self,
                 n: int,
                 column: str = "user_id",
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.n = n
        self.column = column
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"previous_{n}_ans"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        group = df.groupby(self.column)
        for key, w_df in group:
            ans = "".join(w_df["answered_correctly"].fillna(9).astype(int).astype(str).values[::-1].tolist())
            ans = ans[:self.n]
            if key not in self.data_dict:
                self.data_dict[key] = ans
            else:
                self.data_dict[key] = ans + self.data_dict[key][len(ans):]
                self.data_dict[key] = self.data_dict[key][:self.n]

        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(is_bundled, answered_correctly):
            if is_bundled:
                return answered_correctly - 5 # bundleされているめじるし
            else:
                return answered_correctly

        def g(x):
            rets = [""]
            ret = [] # データに記録される数値
            w_ret = [] # 正しい数値
            for s in x.values:
                if np.isnan(s):
                    ret = [9] + ret
                    w_ret = [9] + w_ret
                elif s <= -4:
                    ret = [8] + ret
                    w_ret = [s+5] + w_ret
                else:
                    ret = [s] + ret
                    w_ret = [s] + w_ret
                if s >= 0:
                    ret = w_ret
                ret = ret[:self.n]
                rets.append("".join([str(int(x)) for x in ret]))
            return rets[:-1]
        df["is_bundled"] = \
            (df.groupby("user_id")["bundle_id"].shift(-1) == df["bundle_id"]).astype("int8")
        df["is_bundled"] = (df["is_bundled"] > 0)
        df["answered_correctly_bundle"] = [f(x[0], x[1]) for x in df[["is_bundled", "answered_correctly"]].values]
        df[self.make_col_name] = df.groupby("user_id")["answered_correctly_bundle"].progress_transform(g)

        df = df.drop(["is_bundled", "answered_correctly_bundle"], axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, content_type_id):
            if user_id not in self.data_dict:
                if is_update:
                    self.data_dict[user_id] = ""
                ret = ""
            else:
                ret = self.data_dict[user_id]

            if is_update:
                if content_type_id == 1:
                    self.data_dict[user_id] = ("9" + self.data_dict[user_id])[:self.n]
                else:
                    self.data_dict[user_id] = ("8" + self.data_dict[user_id])[:self.n]
            return ret

        df[self.make_col_name] = [f(x[0], x[1]) for x in df[["user_id", "content_type_id"]].values]
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"


class PreviousNSameContentIdAnsweredCorrectly(FeatureFactory):
    """
    過去N回のanswered_correctlyを文字列として結合する
    ただし:
    bundle_idが同じものは,最後のbundle_idが終わるまで答えがわからないので、8
    partial_predictで答え未知の場合は、8
    lecture(content_type_id=1)の場合は、9
    をそれぞれ設定する。
    """
    def __init__(self,
                 n: int,
                 column: str = ("user_id", "content_id"),
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.n = n
        self.column = column
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.split_num = split_num
        self.data_dict = {}
        self.is_partial_fit = is_partial_fit
        self.make_col_name = f"previous_{n}_same_content_id_ans"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        group = df.groupby(self.column)
        for key, w_df in group:
            ans = "".join(w_df["answered_correctly"].fillna(9).astype(int).astype(str).values[::-1].tolist())
            ans = ans[:self.n]
            if key not in self.data_dict:
                self.data_dict[key] = ans
            else:
                self.data_dict[key] = ans + self.data_dict[key][len(ans):]
                self.data_dict[key] = self.data_dict[key][:self.n]

        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def g(x):
            rets = [""]
            ret = [] # データに記録される数値
            w_ret = [] # 正しい数値
            for s in x.values:
                if np.isnan(s):
                    ret = [9] + ret
                    w_ret = [9] + w_ret
                else:
                    ret = [s] + ret
                    w_ret = [s] + w_ret
                if s >= 0:
                    ret = w_ret
                ret = ret[:self.n]
                rets.append("".join([str(int(x)) for x in ret]))
            return rets[:-1]
        df[self.make_col_name] = df.groupby(["user_id", "content_id"])["answered_correctly"].progress_transform(g)

        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, content_id, content_type_id):
            key = (user_id, content_id)
            if key not in self.data_dict:
                if is_update:
                    self.data_dict[key] = ""
                ret = ""
            else:
                ret = self.data_dict[key]

            if is_update:
                if content_type_id == 1:
                    self.data_dict[key] = ("9" + self.data_dict[key])[:self.n]
                else:
                    self.data_dict[key] = ("8" + self.data_dict[key])[:self.n]
            return ret

        df[self.make_col_name] = [f(x[0], x[1], x[2]) for x in df[["user_id", "content_id", "content_type_id"]].values]
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.column})"



class WeightDecayTargetEncoder(FeatureFactory):
    feature_name_base = "weighted_te_user_id"

    def __init__(self,
                 column: Union[list, str],
                 past_n: int = 100,
                 decay: float = 0.006,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 split_num: int = 1,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        super().__init__(column=column,
                         split_num=split_num,
                         logger=logger,
                         is_partial_fit=is_partial_fit)
        self.past_n = past_n
        self.decay = decay
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.make_col_name = f"{self.feature_name_base}_past{past_n}_decay{decay}"

    def fit(self,
            df,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        ww_df = df.reset_index(drop=True).groupby(self.column).tail(self.past_n).reset_index(drop=True)
        group = ww_df.groupby(self.column)
        ans_correct = ww_df["answered_correctly"].values # pandas -> np.array変換の時間節約

        for key, w_df in group:
            w_df = w_df[w_df["answered_correctly"].notnull()]
            if len(w_df) == 0:
                continue
            if key not in self.data_dict:
                start_weight = 1 - self.decay * len(w_df)
                start_weight += self.decay  # np.arange調整用
                weight = np.arange(start_weight, 1 + self.decay * 0.01, self.decay)
                weight_sum = weight.sum()
                value = ans_correct[w_df.index] * weight / weight_sum
                self.data_dict[key] = {}
                self.data_dict[key][self.make_col_name] = value
                self.data_dict[key]["ans_ary"] = ans_correct[w_df.index]

            else:
                ans_ary = np.concatenate([self.data_dict[key]["ans_ary"], ans_correct[w_df.index]])[-self.past_n:]
                start_weight = 1 - self.decay * len(ans_ary)
                start_weight += self.decay  # np.arange調整用
                weight = np.arange(start_weight, 1 + self.decay * 0.01, self.decay)

                value = (ans_ary * weight).sum() / weight.sum()
                self.data_dict[key][self.make_col_name] = value
                self.data_dict[key]["ans_ary"] = ans_ary

        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(series):
            series = series[series.notnull()]
            if len(series) == 0:
                return np.nan
            start_weight = 1 - self.decay * len(series)
            start_weight += self.decay
            weight = np.arange(start_weight, 1 + self.decay*0.01, self.decay)
            return (series * weight).sum() / weight.sum()

        self.logger.info(f"weightdecay_target_encoding_all_{self.column}")
        df["ans_shift1"] = df.groupby(self.column)[["user_id", "answered_correctly"]].shift(1)
        w_df = df.groupby(self.column)["ans_shift1"].rolling(
                window=self.past_n, min_periods=1
            ).apply(f).reset_index().astype("float32")
        w_df.columns = ["user_id", "index", self.make_col_name]
        w_df = w_df.set_index("index").drop("user_id", axis=1)
        df = pd.concat([df, w_df], axis=1).reset_index(drop=True)
        df = df.drop("ans_shift1", axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        df = self._partial_predict2(df, column=self.make_col_name)
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        return df


class StudyTermEncoder(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "study_time"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):
        df["study_time"] = df["shiftdiff_timestamp_by_user_id_cap200k"] - df["prior_question_elapsed_time"]
        df["study_time"] = df["study_time"].fillna(-1).astype("int32")
        df["study_time2"] = df["shiftdiff_timestamp_by_user_id_cap200k"] + df["prior_question_elapsed_time"]
        df["study_time2"] = df["study_time2"].fillna(-1).astype("int32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class StudyTermEncoder2(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.make_col_name = "study_time2"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        pass

    def make_feature(self,
                     df: pd.DataFrame):
        return df

    def _predict(self,
                 df: pd.DataFrame):
        df["study_time"] = df["duration_previous_content_cap100k"] - df["prior_question_elapsed_time"]
        df["study_time"] = df["study_time"].fillna(-1).astype("int32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"tags_all")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ElapsedTimeVsShiftDiffEncoder(FeatureFactory):

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 elapsed_time_dict: Union[Dict[str, list], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.dict_path = f"../feature_engineering/elapsedtime.pickle"
        self.make_col_name = f"shiftdiff_elapsedtime"
        if elapsed_time_dict is None:
            if not os.path.isfile(self.dict_path):
                print("make_new_dict")
                cols = ["user_id", "content_id", "prior_question_elapsed_time"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]
                print("loaded")
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.elapsed_time_dict = pickle.load(f)
        else:
            self.elapsed_time_dict = elapsed_time_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        df["elapsed_time"] = df.groupby("user_id")["prior_question_elapsed_time"].shift(-1)

        ret_dict = df.groupby("content_id")["elapsed_time"].mean().to_dict()
        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        df["elapsed_time_content_id"] = [self.elapsed_time_dict[x] for x in df["content_id"].values]
        df["diff_shiftdiff_elapsed_time"] = df["shiftdiff_timestamp_by_user_id_cap200k"] - df["elapsed_time_content_id"]
        df["elapsed_time_content_id"] = df["elapsed_time_content_id"].astype("int32")
        df["diff_shiftdiff_elapsed_time"] = df["diff_shiftdiff_elapsed_time"].astype("int32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):

        self.logger.info(f"elapsed_time")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Word2VecEncoder(FeatureFactory):

    def __init__(self,
                 columns: List[str],
                 window: int,
                 size: int,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 w2v_dict: Union[Dict[str, list], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.columns = columns
        self.window = window
        self.size = size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.dict_path = f"../feature_engineering/w2v_{columns}_window{window}_size{size}.pickle"
        self.make_col_name = f"w2v_{columns}_window{window}_size{size}"
        if w2v_dict is None:
            if not os.path.isfile(self.dict_path):
                print("make_new_dict")
                df = pd.read_pickle(MERGE_FILE_PATH)[["user_id"] + self.columns]
                print("loaded")
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.w2v_dict = pickle.load(f)
        else:
            self.w2v_dict = w2v_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        df["key"] = ["_".join(x.tolist()) for x in df[self.columns].astype(str).values]
        histories = df.groupby("user_id").agg({"key": list})
        w2v = word2vec.Word2Vec(histories.values.flatten().tolist(), size=self.size, window=self.window)
        ret_dict = {k: w2v[k].tolist() for k in w2v.wv.index2word}
        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        df["key"] = ["_".join(x.tolist()) for x in df[self.columns].astype(str).values]
        list_dict = df.groupby("user_id").agg({"key": list}).to_dict()["key"]
        for user_id, list_key in list_dict.items():
            if user_id not in self.data_dict:
                self.data_dict[user_id] = list_key[-self.window:]
            else:
                self.data_dict[user_id] = (self.data_dict[user_id] + list_key)[-self.window:]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):

        self.logger.info(f"useranswer")
        def f(w_df):
            def make_lecture_list(series):
                ret = []
                w_ret = []
                for x in series.values:
                    w_ret.append(x)
                    ret.append(w_ret[-self.window:])
                return ret

            def calc_score(x):
                score = []
                for key in x:
                    if key in self.w2v_dict:
                        score.append(self.w2v_dict[key])
                    else:
                        score.append([np.nan]*self.size)
                return score[-self.window:]
            w_df["list_keys"] = w_df.groupby("user_id")["key"].transform(make_lecture_list)

            score = np.array([calc_score(x) for x in w_df["list_keys"].values])
            return np.array(score)

        self.logger.info(f"Word2Vec encoding")

        df["key"] = ["_".join(x.tolist()) for x in df[self.columns].astype(str).values]
        df_rets = []
        for key, w_df in tqdm.tqdm(df.groupby("user_id")):
            score = f(w_df)
            df_ret = pd.DataFrame(index=w_df.index)

            expect_mean = np.array([np.array(x).mean(axis=0) if len(x) > 0 else np.nan for x in score])
            expect_max = np.array([np.array(x).max(axis=0) if len(x) > 0 else np.nan for x in score])
            expect_min = np.array([np.array(x).min(axis=0) if len(x) > 0 else np.nan for x in score])
            expect_last = np.array([x[-1] if len(x) > 0 else np.nan for x in score])
            for i in range(self.size):
                df_ret[f"swem_max_{self.make_col_name}_dim{i}"] = expect_max[:, i]
                df_ret[f"swem_min_{self.make_col_name}_dim{i}"] = expect_min[:, i]
                df_ret[f"swem_mean_{self.make_col_name}_dim{i}"] = expect_mean[:, i]
                df_ret[f"{self.make_col_name}_dim{i}"] = expect_last[:, i]

            df_rets.append(df_ret)

        df_rets = pd.concat(df_rets).sort_index()

        for col in df_rets.columns:
            df[col] = df_rets[col].astype("float32")
        df = df.drop("key", axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def calc_score(x):
            user_id = x[0]
            wv_key = x[1]

            score = []
            keys = []
            if user_id in self.data_dict:
                keys.extend(self.data_dict[user_id])
            keys.append(wv_key)
            for key in keys:
                if key in self.w2v_dict:
                    score.append(self.w2v_dict[key])
                else:
                    score.append([np.nan]*self.size)
            return np.array(score[-self.window:])

        df["key"] = ["_".join(x.tolist()) for x in df[self.columns].astype(str).values]
        score = [calc_score(x) for x in df[["user_id", "key"]].values]
        expect_mean = np.array([np.array(x).mean(axis=0) if len(x) > 0 else np.nan for x in score])
        expect_max = np.array([np.array(x).max(axis=0) if len(x) > 0 else np.nan for x in score])
        expect_min = np.array([np.array(x).min(axis=0) if len(x) > 0 else np.nan for x in score])
        expect_last = np.array([x[-1] if len(x) > 0 else np.nan for x in score])
        for i in range(self.size):
            df[f"swem_max_{self.make_col_name}_dim{i}"] = expect_max[:, i]
            df[f"swem_min_{self.make_col_name}_dim{i}"] = expect_min[:, i]
            df[f"swem_mean_{self.make_col_name}_dim{i}"] = expect_mean[:, i]
            df[f"{self.make_col_name}_dim{i}"] = expect_last[:, i]

            df[f"swem_max_{self.make_col_name}_dim{i}"] = df[f"swem_max_{self.make_col_name}_dim{i}"].astype("float32")
            df[f"swem_min_{self.make_col_name}_dim{i}"] = df[f"swem_min_{self.make_col_name}_dim{i}"].astype("float32")
            df[f"swem_mean_{self.make_col_name}_dim{i}"] = df[f"swem_mean_{self.make_col_name}_dim{i}"].astype("float32")
            df[f"{self.make_col_name}_dim{i}"] = df[f"{self.make_col_name}_dim{i}"].astype("float32")

        return df

    def __repr__(self):
        return self.__class__.__name__

class PastNFeatureEncoder(FeatureFactory):
    question_lecture_dict_path = "../feature_engineering/question_question_dict_{}.pickle"

    def __init__(self,
                 column: str,
                 past_ns: List[int],
                 remove_now: bool,
                 agg_funcs: List[str],
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.column = column
        self.past_ns = past_ns
        self.remove_now = remove_now
        self.agg_funcs = agg_funcs
        self.past_n = np.array(past_ns).max()
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_{column}_{past_ns}"
        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        if is_first_fit:
            group = df[df[self.column].notnull()].groupby("user_id")
            for user_id, w_df in group:
                if user_id not in self.data_dict:
                    self.data_dict[user_id] = w_df[self.column].values.tolist()[-self.past_n:]
                else:
                    self.data_dict[user_id] = (self.data_dict[user_id] + w_df[self.column].values.tolist())[-self.past_n:]
        return self

    def make_lecture_list(self, series):
        ret = []
        w_ret = []
        for x in series.values:
            if not np.isnan(x):
                w_ret.append(x)
            if self.remove_now:
                ret.append(w_ret[-self.past_n-1:-1])
            else:
                ret.append(w_ret[-self.past_n:])
        return ret

    def _make_feature(self, df, values):
        for past_n in self.past_ns:
            for agg_func in self.agg_funcs:
                if agg_func == "min":
                    df[f"past{past_n}_{self.column}_min"] = [np.array(x[-past_n:]).min() if len(x) > 0 else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_min"] = df[f"past{past_n}_{self.column}_min"].astype("float32")
                if agg_func == "max":
                    df[f"past{past_n}_{self.column}_max"] = [np.array(x[-past_n:]).max() if len(x) > 0 else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_max"] = df[f"past{past_n}_{self.column}_max"].astype("float32")
                if agg_func == "mean":
                    df[f"past{past_n}_{self.column}_mean"] = [np.array(x[-past_n:]).mean() if len(x) > 0 else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_mean"] = df[f"past{past_n}_{self.column}_mean"].astype("float32")
                if agg_func == "last":
                    df[f"past{past_n}_{self.column}_last"] = [x[-past_n] if len(x) >= past_n else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_last"] = df[f"past{past_n}_{self.column}_last"].astype("float32")
                if agg_func == "vslast":
                    df[f"past{past_n}_{self.column}_vslast"] = [x[-1] - x[-past_n] if len(x) >= past_n else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_vslast"] = df[f"past{past_n}_{self.column}_vslast"].astype("float32")
                if agg_func == "std":
                    df[f"past{past_n}_{self.column}_mean"] = [np.array(x[-past_n:]).std() if len(x) > 0 else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_mean"] = df[f"past{past_n}_{self.column}_std"].astype("float32")
                if agg_func == "nunique":
                    df[f"past{past_n}_{self.column}_mean"] = [np.array(x[-past_n:]).nunique() if len(x) > 0 else np.nan for x in values]
                    df[f"past{past_n}_{self.column}_mean"] = df[f"past{past_n}_{self.column}_nunique"].astype("float32")

        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"past_n_feature")

        values = df.groupby("user_id")[self.column].progress_transform(self.make_lecture_list).values
        return self._make_feature(df, values)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        """
        リアルタイムfit
        :param df:
        :return:
        """

        def f(user_id, value):
            """
            user_idをしらべdataを返す、ついでに辞書もupdateする!
            :param user_id:
            :param value:
            :return:
            """
            if user_id in self.data_dict:
                if self.remove_now:
                    ret = self.data_dict[user_id]
                else:
                    ret = (self.data_dict[user_id] + [value])[-self.past_n:]
            else:
                if self.remove_now:
                    ret = []
                else:
                    ret = [value]
            if is_update and not self.remove_now:
                self.data_dict[user_id] = ret
            return ret

        values = [f(x[0], x[1]) for x in df[["user_id", self.column]].values]

        return self._make_feature(df, values)

    def __repr__(self):
        return self.__class__.__name__


class PreviousContentAnswerTargetEncoder(FeatureFactory):
    prev_dict_path = "../feature_engineering/previous_content_answer_te_{}.pickle"

    def __init__(self,
                 min_size: int=1000,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 prev_dict: Union[Dict[tuple, float], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.make_col_name = f"{self.__class__.__name__}_th{self.min_size}"
        self.data_dict = {}
        if prev_dict is None:
            if not os.path.isfile(self.prev_dict_path.format(self.min_size)):
                print("make_new_dict")
                files = glob.glob("../input/riiid-test-answer-prediction/split10/*.pickle")
                cols = ["user_id", "content_id", "content_type_id", "user_answer", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]

                print("loaded")
                self.make_dict(df)
            with open(self.prev_dict_path.format(self.min_size), "rb") as f:
                self.prev_dict = pickle.load(f)
        else:
            self.prev_dict = prev_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        ret_dict = {}
        df["previous_content_id"] = df.groupby("user_id")["content_id"].shift(1).fillna(-1)
        df["previous_user_answer"] = df.groupby("user_id")["user_answer"].shift(1).fillna(-1)

        group = df[df["content_type_id"] == 0].groupby(["content_id", "previous_content_id", "previous_user_answer"])["answered_correctly"]

        sum_dict = group.sum()
        size_dict = group.size()

        for key in tqdm.tqdm(sum_dict.keys()):
            if size_dict[key] > self.min_size:
                ret_dict[key] = sum_dict[key] / size_dict[key]

        if output_dir is None:
            output_dir = self.prev_dict_path.format(self.min_size)
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):

        w_ary = df.groupby("user_id").last()[["content_id", "user_answer"]].reset_index().values
        for data in w_ary:
            self.data_dict[data[0]] = (data[1], data[2])
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        self.logger.info(f"previous_content_answer_te__encode")

        df["previous_content_id"] = df.groupby("user_id")["content_id"].shift(1).fillna(-1)
        df["previous_user_answer"] = df.groupby("user_id")["user_answer"].shift(1).fillna(-1)

        cols = ["content_id", "previous_content_id", "previous_user_answer"]
        df["prev_ans_te"] = [self.prev_dict[tuple(x)] if tuple(x) in self.prev_dict else np.nan for x in df[cols].values]
        df["prev_ans_te"] = df["prev_ans_te"].astype("float32")

        df = df.drop(["previous_content_id", "previous_user_answer"], axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(x):
            user_id = x[0]
            content_id = x[1]
            if user_id in self.data_dict:
                prev_content_id = self.data_dict[user_id][0]
                prev_user_answer = self.data_dict[user_id][1]
                key = (content_id, prev_content_id, prev_user_answer)
                if key in self.prev_dict:
                    return self.prev_dict[key]
                else:
                    return np.nan
            else:
                key = (content_id, -1, -1)
                if key in self.prev_dict:
                    return self.prev_dict[key]
                else:
                    return np.nan
        df["prev_ans_te"] = [f(x) for x in df[["user_id", "content_id"]].values]
        df["prev_ans_te"] = df["prev_ans_te"].astype("float32")
        return df

    def __repr__(self):
        return self.__class__.__name__


class DurationPreviousContent(FeatureFactory):
    feature_name_base = ""

    def __init__(self,
                 groupby: Union[str, list] = "user_id",
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False):
        self.groupby = groupby
        self.column = "timestamp"
        self.logger = logger
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.is_partial_fit = is_partial_fit
        if self.groupby == "user_id":
            self.make_col_name = f"duration_previous_content"
        else:
            self.make_col_name = f"duration_previous_content_{self.groupby}"

        self.data_dict = {}

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[Union[str, tuple],
                                       Dict[str, object]],
            is_first_fit: bool):
        group = df.groupby(self.groupby)
        if len(self.data_dict) == 0:
            self.data_dict = group[self.column].last().to_dict()
        else:
            for key, value in group[self.column].last().to_dict().items():
                self.data_dict[key] = value
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        df[self.make_col_name] = df[self.column] - df.groupby(self.groupby)[self.column].shift(1)
        df[self.make_col_name] = df[self.make_col_name].replace(0, np.nan)
        df[self.make_col_name] = df.groupby(self.groupby)[self.make_col_name].fillna(method="ffill")
        df[self.make_col_name] = df[self.make_col_name] / df.groupby(["user_id", "task_container_id"])["user_id"].transform("count")
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("uint32")
        df[f"{self.make_col_name}_cap100k"] = [x if x < 100000 else 100000 for x in df[self.make_col_name].values]
        df[f"{self.make_col_name}_cap100k"] = df[f"{self.make_col_name}_cap100k"].astype("uint32")
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        """
        リアルタイムfit
        :param df:
        :return:
        """
        groupby_values = df[self.groupby].values
        if type(self.groupby) == list:
            groupby_values = [tuple(x) for x in groupby_values]

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
        df[self.make_col_name] = (df[self.column] - w_diff).replace(0, np.nan)
        df[self.make_col_name] = df.groupby(self.groupby)[self.make_col_name].fillna(method="ffill")
        df[self.make_col_name] = df[self.make_col_name] / df.groupby(["user_id", "task_container_id"])["user_id"].transform("count")
        df[self.make_col_name] = df[self.make_col_name].fillna(0).astype("uint32")
        df[f"{self.make_col_name}_cap100k"] = [x if x < 100000 else 100000 for x in df[self.make_col_name].values]
        df[f"{self.make_col_name}_cap100k"] = df[f"{self.make_col_name}_cap100k"].astype("uint32")

        if is_update:
            for key, value in df.groupby(self.groupby)[self.column].last().to_dict().items():
                self.data_dict[key] = value
        return df

class ElapsedTimeMeanByContentIdEncoder(FeatureFactory):

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 elapsed_time_dict: Union[Dict[str, list], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.dict_path = f"../feature_engineering/elapsed_time_mean_by_content_id.pickle"
        self.make_col_name = "elapsed_time_mean_by_content_id"
        if elapsed_time_dict is None:
            if not os.path.isfile(self.dict_path):
                print("make_new_dict")
                cols = ["user_id", "content_id", "task_container_id", "prior_question_elapsed_time"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]
                print("loaded")
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.elapsed_time_dict = pickle.load(f)
        else:
            self.elapsed_time_dict = elapsed_time_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        w_df = df.drop_duplicates(["user_id", "task_container_id"])
        w_df["elapsed_time"] = w_df.groupby("user_id")["prior_question_elapsed_time"].shift(-1)

        df = pd.merge(df, w_df[["user_id", "task_container_id", "elapsed_time"]], how="inner")[["content_id", "elapsed_time"]]
        df = df.dropna()
        ret_dict = df.groupby("content_id")["elapsed_time"].mean().to_dict()
        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        df["elapsed_time_content_id_mean"] = [self.elapsed_time_dict[x] for x in df["content_id"].values]
        df["elapsed_time_content_id_mean"] = df["elapsed_time_content_id_mean"].astype("int32")

        df["elapsed_time_mean_content_id_vs_prior_elapsed_time"] = df["prior_question_elapsed_time"] - df["elapsed_time_content_id_mean"]
        df["elapsed_time_mean_content_id_vs_prior_elapsed_time"] = df["elapsed_time_mean_content_id_vs_prior_elapsed_time"].fillna(0).astype("int32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):

        self.logger.info(f"elapsed_time")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class ElapsedTimeBinningEncoder(FeatureFactory):

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.make_col_name = "elapsed_time_mean_by_content_id"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        df["prior_question_elapsed_time_bin300"] = [x // 1000 if x // 1000 < 300 else 300 for x in
                                                    df["prior_question_elapsed_time"].fillna(0).values]
        df["duration_previous_content_bin300"] = [x // 1000 if x // 1000 < 300 else 300 for x in
                                                  df["duration_previous_content"].fillna(0).values]
        df["prior_question_elapsed_time_bin300"] = df["prior_question_elapsed_time_bin300"].astype("int16")
        df["duration_previous_content_bin300"] = df["duration_previous_content_bin300"].astype("int16")

        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):

        self.logger.info(f"elapsed_time_binning")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"

class PastNUserAnswerHistory(FeatureFactory):

    def __init__(self,
                 past_n: int,
                 min_size: int,
                 model_id: str = None,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 userans_te_dict: Union[Dict[str, list], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.past_n = past_n
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.dict_path = f"../feature_engineering/past{past_n}_min_size{min_size}_user_answer_history.pickle"
        self.make_col_name = f"past{past_n}_user_answer_history"
        if userans_te_dict is None:
            if not os.path.isfile(self.dict_path):
                print("make_new_dict")
                cols = ["user_id", "content_id", "user_answer", "answered_correctly"]
                df = pd.read_pickle(MERGE_FILE_PATH)[cols]
                print("loaded")
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.userans_te_dict = pickle.load(f)
        else:
            self.userans_te_dict = userans_te_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """
        
        df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
        cols = ["content_id"]
        for i in tqdm.tqdm(range(1, self.past_n+1)):
            col_content = f"target{i}_content"
            col_ans = f"target{i}_ans"

            df[col_content] = df.groupby("user_id")["content_id"].shift(i).fillna(-1)
            df[col_ans] = df.groupby("user_id")["user_answer"].shift(i).fillna(-1)

            cols.append(col_content)
            cols.append(col_ans)

        df["agg"] = [tuple(x) for x in df[cols].values]

        dict_size = df.groupby("agg")["answered_correctly"].size().to_dict()
        dict_sum = df.groupby("agg")["answered_correctly"].sum().to_dict()

        ret_dict = {}
        for key in dict_size.keys():
            if dict_size[key] > self.min_size:
                ret_dict[key] = dict_sum[key] / dict_size[key]

        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        group = df.groupby("user_id")

        for user_id, w_df in group:
            keys = w_df.tail(self.past_n)[["user_answer", "content_id"]].fillna(-1).values.reshape(-1)[::-1]
            if user_id not in self.data_dict:
                self.data_dict[user_id] = keys.tolist()
            else:
                self.data_dict[user_id] = (keys.tolist() + self.data_dict[user_id])[:self.past_n*2]
        return self

    def _all_predict_core(self,
                    df: pd.DataFrame):
        def f(x):
            if x in self.userans_te_dict:
                return self.userans_te_dict[x]
            else:
                return np.nan
        self.logger.info(f"past_n_user_answer_history")

        cols = ["content_id"]
        for i in range(1, self.past_n+1):
            col_content = f"target{i}_content"
            col_ans = f"target{i}_ans"

            df[col_content] = df.groupby("user_id")["content_id"].shift(i).fillna(-1)
            df[col_ans] = df.groupby("user_id")["user_answer"].shift(i).fillna(-1)

            cols.append(col_content)
            cols.append(col_ans)

        df[self.make_col_name] = [f(tuple(x)) for x in df[cols].values]
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        df = df.drop(cols[1:], axis=1)
        return df

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        def f(user_id, content_id):
            ret = [content_id]

            if user_id in self.data_dict:
                ret.extend(self.data_dict[user_id])

            key_length = self.past_n*2 + 1
            if len(ret) < key_length:
                ret += [-1] * (key_length - len(ret))

            ret = tuple(ret)

            if ret in self.userans_te_dict:
                return self.userans_te_dict[ret]
            else:
                return np.nan
        df[self.make_col_name] = [f(x[0], x[1]) for x in df[["user_id", "content_id"]].values]
        df[self.make_col_name] = df[self.make_col_name].astype("float32")
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}"


class CorrectVsIncorrectMeanEncoder(FeatureFactory):

    def __init__(self,
                 groupby: str,
                 column: str,
                 min_size: int=0,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 target_dict: Union[Dict[tuple, list], None] = None,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.groupby = groupby
        self.column = column
        self.min_size = min_size
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.dict_path = f"../feature_engineering/correct_vs_incorrect_{groupby}_{column}_min{min_size}_{model_id}.pickle"
        self.make_col_name = "correct_vs_incorrect"
        self.target_dict = target_dict
        if self.target_dict is None:
            if os.path.isfile(self.dict_path):
                with open(self.dict_path, "rb") as f:
                    self.target_dict = pickle.load(f)

    def make_dict(self,
                  df: pd.DataFrame,
                  output_dir: str = None):
        """
        question_lecture_dictを作って, 所定の場所に保存する
        :param df:
        :param is_output:
        :return:
        """

        size_dict = df.groupby([self.groupby, "answered_correctly"])[self.column].size().to_dict()
        sum_dict = df.groupby([self.groupby, "answered_correctly"])[self.column].sum().to_dict()

        ret_dict = {}
        for key in size_dict.keys():
            if size_dict[key] > self.min_size:
                ret_dict[key] = sum_dict[key] / size_dict[key]
        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        def f(x, target):
            if (x, target) in self.target_dict:
                return self.target_dict[(x, target)]
            else:
                return np.nan

        df[f"diff_{self.column}_groupby_{self.groupby}_vs_incorrect_mean"] = \
            (df[self.column] - [f(x, 0) for x in df[self.groupby]]).astype("float32")
        df[f"diff_{self.column}_groupby_{self.groupby}_vs_correct_mean"] = \
            (df[self.column] - [f(x, 1) for x in df[self.groupby]]).astype("float32")
        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):

        if self.target_dict is None:
            if not os.path.isfile(self.dict_path):
                print("make_new_dict")
                self.make_dict(df)
            with open(self.dict_path, "rb") as f:
                self.target_dict = pickle.load(f)

        self.logger.info(f"correct_vs_incorrect")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class PostProcessSumEncoder(FeatureFactory):

    def __init__(self,
                 model_id: str = None,
                 load_feature: bool = False,
                 save_feature: bool = False,
                 logger: Union[Logger, None] = None,
                 is_partial_fit: bool = False,
                 is_debug: bool = False):
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
        self.logger = logger
        self.is_partial_fit = is_partial_fit
        self.is_debug = is_debug
        self.data_dict = {}
        self.make_col_name = "elapsed_time_mean_by_content_id"

    def fit(self,
            df: pd.DataFrame,
            feature_factory_dict: Dict[str,
                                       Dict[str, FeatureFactory]],
            is_first_fit: bool):
        return self

    def _predict(self,
                 df: pd.DataFrame):
        df["sum_enc_user_id"] = (df["target_enc_user_id"] * df["count_enc_user_id"]).astype("float32")
        df["sum_enc_['user_id', 'part']"] = (df["target_enc_['user_id', 'part']"] * df["count_enc_['user_id', 'part']"]).astype("float32")

        return df

    def _all_predict_core(self,
                    df: pd.DataFrame):

        self.logger.info(f"elapsed_time_binning")

        return self._predict(df)

    def partial_predict(self,
                        df: pd.DataFrame,
                        is_update: bool=True):
        return self._predict(df)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class FeatureFactoryManager:
    def __init__(self,
                 feature_factory_dict: Dict[Union[str, tuple],
                                            Dict[str, FeatureFactory]],
                 logger: Logger,
                 load_feature: bool = None,
                 save_feature: bool = None,
                 model_id: str = None,
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
        self.load_feature = load_feature
        self.save_feature = save_feature
        self.model_id = model_id
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
                factory.model_id = model_id
                if self.load_feature is not None and factory.load_feature is None:
                    factory.load_feature = load_feature
                if self.save_feature is not None and factory.save_feature is None:
                    factory.save_feature = save_feature


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

            for factory in dicts.values():
                if factory.is_partial_fit:
                    if not is_first_fit:
                        df = factory.partial_predict(df, is_update=False)
                    else:
                        df = factory.all_predict(df)
                    df = factory.make_feature(df)
                    factory.fit(df=df,
                                feature_factory_dict=self.feature_factory_dict,
                                is_first_fit=is_first_fit)

        # not partial_fit
        for column, dicts in self.feature_factory_dict.items():
            # カラム(ex: user_idなど)ごとに処理

            for factory in dicts.values():
                if not factory.is_partial_fit:
                    factory.fit(df=df,
                                feature_factory_dict=self.feature_factory_dict,
                                is_first_fit=is_first_fit)

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
                    self.logger.info(factory)
                    df = factory.all_predict(df=df)

        # partial_predictなし
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if not factory.is_partial_fit:
                    self.logger.info(factory)
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
                    df = factory.partial_predict(df)

        # partial_predictなし
        for dicts in self.feature_factory_dict.values():
            for factory in dicts.values():
                if not factory.is_partial_fit:
                    df = factory.partial_predict(df)

        return df

