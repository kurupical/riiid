import numpy as np
import pandas as pd

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datetime import datetime as dt
import os
import glob
import pickle
import json
from feature_engineering.feature_factory_for_transformer import FeatureFactoryForTransformer
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    DurationPreviousContent, \
    ElapsedTimeBinningEncoder, \
    UserContentRateEncoder, \
    QuestionQuestionTableEncoder2, \
    PreviousAnswer2, \
    PreviousAnswer3, \
    StudyTermEncoder2, \
    MeanAggregator, \
    ElapsedTimeMeanByContentIdEncoder, \
    DurationFeaturePostProcess
from experiment.common import get_logger
import time
from transformers import AdamW, get_linear_schedule_with_warmup

torch.manual_seed(0)
np.random.seed(0)
is_debug = False
is_make_feature_factory = True
load_pickle = True
epochs = 12
device = torch.device("cuda")

wait_time = 0

def main(params: dict,
         output_dir: str):
    import mlflow
    print("start params={}".format(params))
    model_id = "all"
    logger = get_logger()
    column_config = {
        ("content_id", "content_type_id"): {"type": "category", "dtype": np.int16},
        "user_answer": {"type": "leakage_feature", "dtype": np.int8},
        "answered_correctly": {"type": "leakage_feature", "dtype": np.int8},
        "part": {"type": "category", "dtype": np.int8},
        "prior_question_elapsed_time_bin300": {"type": "category", "dtype": np.int16},
        "duration_previous_content_bin300": {"type": "category", "dtype": np.int16},
        "prior_question_had_explanation": {"type": "category", "dtype": np.int8},
        "rating_diff_content_user_id": {"type": "numeric", "dtype": np.float16},
        "task_container_id_bin300": {"type": "category", "dtype": np.int16},
        "previous_answer_index_question_id": {"type": "category", "dtype": np.int16},
        "previous_answer_question_id": {"type": "category", "dtype": np.int8},
        "timediff-elapsedtime_bin500": {"type": "category", "dtype": np.int16},
        "timedelta_log10": {"type": "category", "dtype": np.int8},
        "previous_answer_index_question_id_0": {"type": "category", "dtype": np.int16},
        "previous_answer_index_question_id_1": {"type": "category", "dtype": np.int16},
        "previous_answer_index_question_id_2": {"type": "category", "dtype": np.int16},
        "previous_answer_index_question_id_3": {"type": "category", "dtype": np.int16},
        "previous_answer_index_question_id_4": {"type": "category", "dtype": np.int16},
        "previous_answer_question_id_0": {"type": "category", "dtype": np.int8},
        "previous_answer_question_id_1": {"type": "category", "dtype": np.int8},
        "previous_answer_question_id_2": {"type": "category", "dtype": np.int8},
        "previous_answer_question_id_3": {"type": "category", "dtype": np.int8},
        "previous_answer_question_id_4": {"type": "category", "dtype": np.int8},
    }


    if is_make_feature_factory:
        # feature factory
        feature_factory_dict = {"user_id": {}}
        feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
        feature_factory_dict["user_id"]["ElapsedTimeBinningEncoder"] = ElapsedTimeBinningEncoder()
        feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(rate_func="elo",
                                                                                           column="user_id")
        feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                             column="question_id",
                                                                             is_debug=is_debug,
                                                                             model_id=model_id,
                                                                             n=300)
        feature_factory_dict["user_id"]["PreviousAnswer3"] = PreviousAnswer3(groupby="user_id",
                                                                             column="question_id",
                                                                             is_debug=is_debug,
                                                                             model_id=model_id,
                                                                             n=500)
        feature_factory_dict["user_id"]["StudyTermEncoder2"] = StudyTermEncoder2(is_partial_fit=True)
        feature_factory_dict["user_id"][f"MeanAggregatorStudyTimebyUserId"] = MeanAggregator(column="user_id",
                                                                                             agg_column="study_time",
                                                                                             remove_now=False)

        feature_factory_dict["user_id"]["ElapsedTimeMeanByContentIdEncoder"] = ElapsedTimeMeanByContentIdEncoder()
        feature_factory_dict["post"] = {
            "DurationFeaturePostProcess": DurationFeaturePostProcess()
        }
        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger,
                                                        split_num=1,
                                                        model_id="all",
                                                        load_feature=not is_debug,
                                                        save_feature=not is_debug)

        ff_for_transformer = FeatureFactoryForTransformer(column_config=column_config,
                                                          dict_path="../feature_engineering/",
                                                          sequence_length=params["max_seq"],
                                                          logger=logger)
        df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
        df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1)
        df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
        if is_debug:
            df = df.head(10000)
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        ff_for_transformer.make_dict(df=df)
        feature_factory_manager.fit(df)
        df = feature_factory_manager.all_predict(df)

        def f(x):
            x = x // 1000
            if x < -90:
                return -90
            if x > 90:
                return 90
            return x
        df["task_container_id_bin300"] = [x if x < 300 else 300 for x in df["task_container_id"].values]
        df["timediff-elapsedtime_bin500"] = [f(x) for x in df["timediff-elapsedtime"].values]
        df["timedelta_log10"] = np.log10(df["duration_previous_content"].values)
        df["timedelta_log10"] = df["timedelta_log10"].replace(-np.inf, -1).replace(np.inf, -1).fillna(-1).astype("int8")
        df = df[["user_id", "content_id", "content_type_id", "part", "user_answer", "answered_correctly",
                 "prior_question_elapsed_time_bin300", "duration_previous_content_bin300",
                 "prior_question_had_explanation", "rating_diff_content_user_id", "task_container_id_bin300",
                 "previous_answer_index_question_id", "previous_answer_question_id", "row_id",
                 "timediff-elapsedtime_bin500", "timedelta_log10",
                 "previous_answer_index_question_id_0", "previous_answer_question_id_0",
                 "previous_answer_index_question_id_1", "previous_answer_question_id_1",
                 "previous_answer_index_question_id_2", "previous_answer_question_id_2",
                 "previous_answer_index_question_id_3", "previous_answer_question_id_3",
                 "previous_answer_index_question_id_4", "previous_answer_question_id_4",
                 ]]
        for dicts in feature_factory_manager.feature_factory_dict.values():
            for factory in dicts.values():
                factory.logger = None
        feature_factory_manager.logger = None
        with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
            pickle.dump(feature_factory_manager, f)

        ff_for_transformer.fit(df)
        ff_for_transformer.logger = None
        with open(f"{output_dir}/feature_factory_manager_for_transformer.pickle", "wb") as f:
            pickle.dump(ff_for_transformer, f)

if __name__ == "__main__":
    if not is_debug:
        for _ in tqdm(range(wait_time)):
            time.sleep(1)
    output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
    os.makedirs(output_dir, exist_ok=True)
    for cont_emb in [8]:
        for cat_emb in [256]:
            dropout = 0.2
            lr = 0.9e-3
            if is_debug:
                batch_size = 8
            else:
                batch_size = 512
            params = {"embed_dim": cat_emb,
                      "cont_emb": cont_emb,
                      "max_seq": 100,
                      "batch_size": batch_size,
                      "num_warmup_steps": 1000,
                      "lr": lr,
                      "dropout": dropout}
            main(params, output_dir=output_dir)