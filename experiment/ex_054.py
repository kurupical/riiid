from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    NUniqueEncoder, \
    TagsSeparator, \
    ShiftDiffEncoder, \
    UserLevelEncoder2, \
    TargetEncodeVsUserId, \
    Counter, \
    PreviousAnswer, \
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder, \
    PriorQuestionElapsedTimeBinningEncoder, \
    PreviousAnswer2
from experiment.common import get_logger
import pandas as pd
from model.cboost import train_catboost_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm
import pickle

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

is_debug = False
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)

def make_feature_factory_manager(split_num):
    logger = get_logger()

    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator()
    }
    for column in ["content_id", "user_id", "part", "prior_question_had_explanation",
                   "tags1", "tags2",
                   ("user_id", "prior_question_had_explanation"), ("user_id", "part"),
                   ("content_id", "prior_question_had_explanation")]:
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
    feature_factory_dict[("prior_question_had_explanation", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["prior_question_had_explanation", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["prior_question_had_explanation", "user_count_bin"])
    }

    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[2, 5])
    feature_factory_dict["user_count_bin"]["CategoryLevelEncoderUserCountBin"] = \
        CategoryLevelEncoder(groupby_column="user_id",
                             agg_column="user_count_bin",
                             categories=[0])

    feature_factory_dict["prior_question_elapsed_time"] = {
        "PriorQuestionElapsedTimeBinningEncoder": PriorQuestionElapsedTimeBinningEncoder()
    }
    feature_factory_dict[("part", "prior_question_elapsed_time_bin")] = {
        "CountEncoder": CountEncoder(column=["part", "prior_question_elapsed_time_bin"]),
        "TargetEncoder": TargetEncoder(column=["part", "prior_question_elapsed_time_bin"])
    }
    feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id", column="content_id", is_debug=is_debug)
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num)
    return feature_factory_manager

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10)
    df = feature_factory_manager.all_predict(df)
    df.tail(1000).to_csv("exp028.csv", index=False)

    df = df.drop(["user_answer", "tags", "type_of"], axis=1)
    df = df[df["answered_correctly"].notnull()]
    print(df.columns)
    print(df.shape)
    model_id = os.path.basename(fname).replace(".pickle", "")
    print(model_id)

    for lr in [0.03, 0.05]:
        for depth in [6, 7, 8, 9, 10]:
            params = {
                'n_estimators': 12000,
                'learning_rate': lr,
                'eval_metric': 'AUC',
                'loss_function': 'Logloss',
                'metric_period': 50,
                'od_wait': 400,
                'task_type': 'GPU',
                'max_depth': depth,
                "verbose": 100
            }
            train_catboost_cv(df,
                              params=params,
                              output_dir=output_dir,
                              model_id=model_id,
                              exp_name=model_id,
                              is_debug=is_debug,
                              drop_user_id=True)
    break