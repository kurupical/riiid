from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    NUniqueEncoder, \
    TagsSeparator, \
    ShiftDiffEncoder, \
    UserLevelEncoder, \
    TargetEncodeVsUserId
from experiment.common import get_logger
import pandas as pd
from model.lgbm import train_lgbm_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    # df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    logger = get_logger()
    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator()
    }
    for column in ["content_id", "user_id", "content_type_id", "prior_question_had_explanation",
                   "tags1", "tags2", "tags3", "tags4", "tags5", "tags6",
                   ("user_id", "content_type_id"), ("user_id", "prior_question_had_explanation")]:
        is_partial_fit = column == "content_id"

        if type(column) == str:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=column),
                "TargetEncoder": TargetEncoder(column=column, is_partial_fit=is_partial_fit)
            }
        else:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=list(column)),
                "TargetEncoder": TargetEncoder(column=list(column), is_partial_fit=is_partial_fit)
            }

    for column in ["part", ("user_id", "tag"), ("user_id", "part"), ("content_type_id", "part")]:
        if type(column) == str:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=column)
            }
        else:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=list(column))
            }

    feature_factory_dict["user_id"]["MeanAggregatorTimestamp"] = MeanAggregator(column="user_id",
                                                                                agg_column="timestamp",
                                                                                remove_now=False)
    feature_factory_dict["user_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="user_id",
                                                                                               agg_column="prior_question_elapsed_time",
                                                                                               remove_now=True)
    feature_factory_dict["user_id"]["ShiftDiffEncoder"] = ShiftDiffEncoder(groupby="user_id",
                                                                           column="timestamp")
    feature_factory_dict["content_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="content_id",
                                                                                                  agg_column="prior_question_elapsed_time",
                                                                                                  remove_now=True)

    feature_factory_dict["postprocess"] = {
        "TargetEncodeVsUserId": TargetEncodeVsUserId()
    }
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=10)
    df = feature_factory_manager.all_predict(df)
    os.makedirs(output_dir, exist_ok=True)
    params = {
        'objective': 'binary',
        'num_leaves': 32,
        'min_data_in_leaf': 15,  # 42,
        'max_depth': -1,
        'learning_rate': 0.3,
        'boosting': 'gbdt',
        'bagging_fraction': 0.7,  # 0.5,
        'feature_fraction': 0.9,
        'bagging_seed': 0,
        'reg_alpha': 5,  # 1.728910519108444,
        'reg_lambda': 5,
        'random_state': 0,
        'metric': 'auc',
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 100
    }
    df.tail(1000).to_csv("exp028.csv", index=False)

    df = df.drop(["user_answer", "tags", "type_of"], axis=1)
    df = df[df["answered_correctly"].notnull()]
    print(df.columns)
    print(df.shape)

    model_id = os.path.basename(fname).replace(".pickle", "")
    print(model_id)
    train_lgbm_cv(df,
                  params=params,
                  output_dir=output_dir,
                  model_id=model_id,
                  exp_name=model_id,
                  drop_user_id=True)
