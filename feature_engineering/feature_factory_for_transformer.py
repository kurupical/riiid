from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import os
import pickle
import tqdm
import glob
from logging import Logger
import copy
MERGE_FILE_PATH = "../input/riiid-test-answer-prediction/train_merged.pickle"


def calc_sec(x, max_sec=300):
    ret = x//1000
    return np.min([ret, max_sec])

class FeatureFactoryForTransformer:
    def __init__(self,
                 column_config: Dict[Union[str, int, tuple], dict],
                 dict_path: str,
                 sequence_length: int,
                 logger: Logger,
                 embbed_dict: Dict[Union[str, int, tuple], int] = {}):
        self.dict_path = dict_path
        self.column_config = column_config
        self.embbed_dict_path = dict_path
        self.embbed_dict = embbed_dict
        self.sequence_length = sequence_length
        self.logger = logger
        self.data_dict = {}

        self._init_for_partial_predict()

    def _init_for_partial_predict(self):
        self.target_cols = ["user_id"]
        self.index_dict = {}
        for k in self.column_config.keys():
            if self.column_config[k]["type"] == "leakage_feature":
                continue
            if type(k) == str:
                self.target_cols.append(k)
            else:  # tuple
                self.target_cols.extend(list(k))

        columns = list(self.column_config.keys())
        for index, key in enumerate(columns):
            if self.column_config[key]["type"] == "leakage_feature":
                continue
            if type(key) == str:
                self.index_dict[key] = self.target_cols.index(key)
            if type(key) == tuple:
                self.index_dict[key] = [self.target_cols.index(x) for x in key]
        print(self.target_cols)

    def make_dict(self, df):
        for key, value in self.column_config.items():
            if value["type"] == "category":
                dict_dir = f"{self.dict_path}/{key}_for_transformer.pickle"
                if not os.path.isfile(dict_dir):
                    if df is None:
                        print("make_new_dict")
                        target_cols = []
                        for k in self.column_config.keys():
                            if type(k) == str:
                                target_cols.append(k)
                            else: # tuple
                                target_cols.extend(list(k))

                        df = df[target_cols]
                        print("loaded")
                    self._make_dict(df=df, column=key, output_dir=dict_dir)
                with open(dict_dir, "rb") as f:
                    self.embbed_dict[key] = pickle.load(f)

    def _make_dict(self,
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
            print(output_dir)
            pickle.dump(ret_dict, f)

    def _fit(self,
             df: pd.DataFrame):
        for user_id, w_df in df.groupby("user_id"):
            if user_id not in self.data_dict:
                self.data_dict[user_id] = {}
                for key in self.column_config.keys():
                    if self.column_config[key]["type"] == "category":
                        embbed_dict = self.embbed_dict[key]
                        if type(key) == tuple:
                            self.data_dict[user_id][key] = [embbed_dict[tuple(x)] for x in w_df[list(key)].values][-self.sequence_length:]
                        else:
                            self.data_dict[user_id][key] = [embbed_dict[x] for x in w_df[key].values][-self.sequence_length:]
                    else:
                        self.data_dict[user_id][key] = w_df[key].values[-self.sequence_length:].tolist()

            else:
                for key in self.column_config.keys():
                    if self.column_config[key]["type"] == "category":
                        embbed_dict = self.embbed_dict[key]
                        if type(key) == tuple:
                            self.data_dict[user_id][key] = \
                                self.data_dict[user_id][key] + [embbed_dict[tuple(x)] for x in w_df[list(key)].values]
                        else:
                            self.data_dict[user_id][key] = \
                                self.data_dict[user_id][key] + [embbed_dict[x] for x in w_df[key].values]
                    else:
                        self.data_dict[user_id][key] = self.data_dict[user_id][key] + w_df[key].values.tolist()
                    self.data_dict[user_id][key] = self.data_dict[user_id][key][-self.sequence_length:]

        return self

    def fit(self,
            df: pd.DataFrame):
        self._fit(df)
        return self

    def all_predict(self,
                    df: pd.DataFrame):
        self.logger.info("all_predict_for_transformer")

        if self.dict_path is not None:
            self.make_dict(df)

        group = {}
        for user_id, w_df in tqdm.tqdm(df.groupby("user_id")):
            w_dict = {}
            for key in self.column_config.keys():
                if self.column_config[key]["type"] == "category":
                    embbed_dict = self.embbed_dict[key]
                    if type(key) == tuple:
                        w_dict[tuple(key)] = [embbed_dict[tuple(x)] for x in w_df[list(key)].values]
                    else:
                        w_dict[key] = [embbed_dict[x] for x in w_df[key].values]
                else:
                    w_dict[key] = w_df[key].values.tolist()
            w_dict["is_val"] = w_df["is_val"].values.tolist()
            group[user_id] = w_dict
        return group

    def partial_predict(self,
                        df: pd.DataFrame):

        def f(x):
            user_id = x[0]

            if user_id not in self.data_dict:
                ret_dict = {x: [] for x in self.column_config.keys()}
            else:
                ret_dict = copy.copy(self.data_dict[user_id])

            for key in self.column_config.keys():
                if self.column_config[key]["type"] == "category":
                    index = self.index_dict[key]
                    if type(key) == tuple:
                        ret_dict[key] = ret_dict[key] + [self.embbed_dict[key][tuple(x[index])]]
                    else:
                        ret_dict[key] = ret_dict[key] + [self.embbed_dict[key][x[index]]]
                elif self.column_config[key]["type"] == "numeric":
                    index = self.index_dict[key]
                    ret_dict[key] = ret_dict[key] + [x[index]]
                elif self.column_config[key]["type"] == "leakage_feature":
                    ret_dict[key] = ret_dict[key] + [-1]
                ret_dict[key] = ret_dict[key][-self.sequence_length:]
            # self.data_dict[user_id] = ret_dict
            return ret_dict

        # x[0]: user_id, x[1]~: keys
        groups = {i: f(x) for i, x in enumerate(df[self.target_cols].values)}
        return groups

