from datetime import datetime as dt
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    TagsSeparator, \
    UserLevelEncoder2, \
    NUniqueEncoder, \
    ShiftDiffEncoder, \
    TargetEncodeVsUserId, \
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder
import pandas as pd
import glob
import os
import tqdm
import lightgbm as lgb
import pickle
import riiideducation
import numpy as np
from logging import Logger, StreamHandler, Formatter
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

model_dir = "../output/ex_039/20201101200838"

data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
}
prior_columns = ["prior_group_responses", "prior_group_answers_correct"]

def get_logger():
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run(debug,
        model_dir,
        kaggle=False):

    if kaggle:
        files_dir = "/kaggle/input/riiid-split10/*.pickle"
    else:
        files_dir = "../input/riiid-test-answer-prediction/split10_base/*.pickle"

    logger = get_logger()
    # environment
    env = riiideducation.make_env()

    df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                              dtype={"bundle_id": "int32",
                                     "question_id": "int32",
                                     "correct_answer": "int8",
                                     "part": "int8"})
    df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                             dtype={"lecture_id": "int32",
                                    "tag": "int16",
                                    "part": "int8"})
    # model loading
    models = []
    for model_path in glob.glob(f"{model_dir}/*model*.pickle"):
        with open(model_path, "rb") as f:
            models.append(pickle.load(f))

    # data preprocessing
    logger = get_logger()
    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator()
    }
    for column in ["content_id", "user_id", "part", "prior_question_had_explanation",
                   "tags1", "tags2",
                   ("user_id", "prior_question_had_explanation"), ("user_id", "part")]:
        is_partial_fit = (column == "content_id" or column == "user_id")

        if type(column) == str:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=column, is_partial_fit=is_partial_fit),
                "TargetEncoder": TargetEncoder(column=column, is_partial_fit=is_partial_fit)
            }
        else:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=list(column), is_partial_fit=is_partial_fit),
                "TargetEncoder": TargetEncoder(column=list(column), is_partial_fit=is_partial_fit)
            }
    feature_factory_dict["user_id"]["ShiftDiffEncoderTimestamp"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="timestamp",
                                                                                    is_partial_fit=True)
    feature_factory_dict["user_id"]["ShiftDiffEncoderContentId"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="content_id")
    for column in ["user_id", "content_id"]:
        feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                           agg_column="prior_question_elapsed_time",
                                                                                                           remove_now=True)

    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    feature_factory_dict["user_id"]["UserCountBinningEncoder"] = UserCountBinningEncoder(is_partial_fit=True)
    feature_factory_dict["user_count_bin"] = {}
    feature_factory_dict["user_count_bin"]["CountEncoder"] = CountEncoder(column="user_count_bin")
    feature_factory_dict["user_count_bin"]["TargetEncoder"] = TargetEncoder(column="user_count_bin")
    feature_factory_dict[("user_id", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["user_id", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["user_id", "user_count_bin"])
    }
    feature_factory_dict[("content_id", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["content_id", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["content_id", "user_count_bin"])
    }

    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[1, 2, 3, 4, 5, 6, 7])
    feature_factory_dict["user_count_bin"]["CategoryLevelEncoderUserCountBin"] = \
        CategoryLevelEncoder(groupby_column="user_id",
                             agg_column="user_count_bin",
                             categories=[0, 1, 2, 3, 4, 5])
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger)
    for model_id, fname in enumerate(glob.glob(files_dir)):
        logger.info(f"loading... {fname}")
        df = pd.read_pickle(fname)
        df = df[df["answered_correctly"] != -1]
        df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
        if debug:
            df = df.head(1000)
        df = pd.concat([pd.merge(df[df["content_type_id"] == 0], df_question,
                                 how="left", left_on="content_id", right_on="question_id"),
                        pd.merge(df[df["content_type_id"] == 1], df_lecture,
                                 how="left", left_on="content_id", right_on="lecture_id")]).sort_values(["user_id", "timestamp"])
        # df = feature_factory_manager.feature_factory_dict["content_id"]["TargetEncoder"].all_predict(df)
        feature_factory_manager.fit(df, is_first_fit=True)

    for dicts in feature_factory_manager.feature_factory_dict.values():
        for factory in dicts.values():
            factory.logger = None
    feature_factory_manager.logger = None
    with open(f"feature_factory_manager.pickle", "wb") as f:
        pickle.dump(feature_factory_manager, f)
    return

if __name__ == "__main__":
    run(debug=False,
        model_dir=model_dir)