from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import os
import pickle
import tqdm
import glob
from logging import Logger

class FeatureFactoryForTransformer:
    def __init__(self,
                 column_config: Dict[Union[str, int, tuple], dict],
                 dict_path: str,
                 sequence_length: int,
                 logger: Logger,
                 embbed_dict: Dict[Union[str, int, tuple], int] = None):
        self.column_config = column_config
        self.embbed_dict_path = dict_path
        self.embbed_dict = embbed_dict
        self.sequence_length = sequence_length
        self.logger = logger
        self.data_dict = {}

        self.embbed_dict = {}
        if dict_path is not None:
            for key, value in column_config.items():
                if value["type"] == "category":
                    dict_dir = f"{dict_path}/{key}.pickle"
                    if not os.path.isfile(dict_dir):
                        print("make_new_dict")
                        files = glob.glob("../input/riiid-test-answer-prediction/split10/*.pickle")
                        df = pd.concat([pd.read_pickle(f).sort_values(["user_id", "timestamp"])[
                                            ["user_id", "content_id", "content_type_id", "answered_correctly", "user_answer"]] for f in files])
                        print("loaded")
                        self.make_dict(df=df, column=key, output_dir=dict_dir)
                        with open(self.dict_dir, "rb") as f:
                            self.embbed_dict[key] = pickle.load(f)
                else:
                    raise NotImplementedError
        else:
            self.embbed_dict = embbed_dict

    def make_dict(self,
                  df: pd.DataFrame,
                  column: Union[str, tuple],
                  output_dir: str):

        ret_dict = {}
        if type(column) == tuple:
            column = list(column)
            for i, key in enumerate(df[column].fillna(-1).sort_values(column).drop_duplicates().values):
                ret_dict[tuple(key)] = i + 1
        else:
            for i, key in enumerate(df[column].fillna(-1).sort_values().drop_duplicates().values):
                ret_dict[key] = i + 1

        if output_dir is None:
            output_dir = self.dict_path
        with open(output_dir, "wb") as f:
            pickle.dump(ret_dict, f)

    def _fit(self,
             df: pd.DataFrame):
        for user_id, w_df in df.groupby("user_id"):
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                for key, embbed_dict in self.embbed_dict.items():
                    self.data_dict[user_id][key] = [embbed_dict[x] for x in w_df[key].values]
                self.data_dict[user_id]["answered_correctly"] = w_df["answered_correctly"].values.tolist()
            else:
                for key, embbed_dict in self.embbed_dict.items():
                    self.data_dict[user_id][key] = \
                        self.data_dict[user_id][key] + [embbed_dict[x] for x in w_df[key].values]
                    self.data_dict[user_id][key] = self.data_dict[user_id][key][-self.sequence_length:]
                self.data_dict[user_id]["answered_correctly"] = \
                    self.data_dict[user_id]["answered_correctly"] + w_df["answered_correctly"].values.tolist()
            self.data_dict[user_id]["answered_correctly"] = self.data_dict[user_id]["answered_correctly"][-self.sequence_length:]
        return self

    def fit(self,
            df: pd.DataFrame,
            is_first_fit=True):
        if is_first_fit:
            self._fit(df)
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info("all_predict_for_transformer")
        group = {}
        for user_id, w_df in tqdm.tqdm(df.groupby("user_id")):
            w_dict = {}
            for key, embbed_dict in self.embbed_dict.items():
                w_dict[key] = [embbed_dict[x] for x in w_df[key].values]
            w_dict["answered_correctly"] = w_df["answered_correctly"].values.tolist()
            w_dict["is_val"] = w_df["is_val"].values.tolist()
            group[user_id] = w_dict
        return group

    def partial_predict(self,
                        df: pd.DataFrame):

        def f(x, index_dict):
            user_id = x[0]

            if user_id not in self.data_dict:
                ret_dict = {x: [] for x in index_dict.keys()}
                ret_dict["answered_correctly"] = []
            else:
                ret_dict = self.data_dict[user_id]

            for key, index in index_dict.items():
                ret_dict[key] = ret_dict[key] + [self.embbed_dict[key][x[index]]]
                ret_dict[key] = ret_dict[key][-self.sequence_length:]

            ret_dict["answered_correctly"] = ret_dict["answered_correctly"] + [-1]  # we can't see future
            ret_dict["answered_correctly"] = ret_dict["answered_correctly"][-self.sequence_length:]
            self.data_dict[user_id] = ret_dict
            return ret_dict

        index_dict = {}
        columns = list(self.column_config.keys())
        for index, key in enumerate(columns):
            index_dict[key] = index + 1

        # x[0]: user_id, x[1]~: keys
        groups = {x[0]: f(x, index_dict) for x in df[["user_id"] + columns].values}
        return groups

