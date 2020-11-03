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
    PriorQuestionElapsedTimeBinningEncoder
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
import pickle

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

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
    for column in [("content_id", "prior_question_had_explanation"),
                   "content_id", "user_id", "part", "prior_question_had_explanation",
                   "tags1", "tags2",
                   ("user_id", "prior_question_had_explanation"), ("user_id", "part")]:
        is_partial_fit = (column == ("content_id", "prior_question_had_explanation") or column == "user_id")

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

    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column=["content_id", "prior_question_had_explanation"])
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
                                                                                       vs_columns=["content_id", "prior_question_had_explanation"],
                                                                                       categories=[2, 5])
    feature_factory_dict["user_count_bin"]["CategoryLevelEncoderUserCountBin"] = \
        CategoryLevelEncoder(groupby_column="user_id",
                             agg_column="user_count_bin",
                             vs_columns=["content_id", "prior_question_had_explanation"],
                             categories=[0])

    feature_factory_dict["prior_question_elapsed_time"] = {
        "PriorQuestionElapsedTimeBinningEncoder": PriorQuestionElapsedTimeBinningEncoder()
    }
    feature_factory_dict[("part", "prior_question_elapsed_time_bin")] = {
        "CountEncoder": CountEncoder(column=["part", "prior_question_elapsed_time_bin"]),
        "TargetEncoder": TargetEncoder(column=["part", "prior_question_elapsed_time_bin"])
    }

    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num)
    return feature_factory_manager

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10)
    df = feature_factory_manager.all_predict(df)
    os.makedirs(output_dir, exist_ok=True)
    params = {
        'objective': 'binary',
        'num_leaves': 32,
        'max_depth': -1,
        'learning_rate': 0.3,
        'boosting': 'gbdt',
        'bagging_fraction': 0.7,
        'feature_fraction': 0.3,
        'bagging_seed': 0,
        'reg_alpha': 5,  # 1.728910519108444,
        'reg_lambda': 5,
        'random_state': 0,
        'metric': 'auc',
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 50
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
                  is_debug=is_debug,
                  drop_user_id=True)

# fit
df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                          dtype={"bundle_id": "int32",
                                 "question_id": "int32",
                                 "correct_answer": "int8",
                                 "part": "int8"})
df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                         dtype={"lecture_id": "int32",
                                "tag": "int16",
                                "part": "int8"})
feature_factory_manager = make_feature_factory_manager(split_num=1)

for fname in glob.glob("../input/riiid-test-answer-prediction/split10_base/*"):
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

    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    df = pd.concat([pd.merge(df[df["content_type_id"] == 0], df_question,
                             how="left", left_on="content_id", right_on="question_id"),
                    pd.merge(df[df["content_type_id"] == 1], df_lecture,
                             how="left", left_on="content_id", right_on="lecture_id")]).sort_values(
        ["user_id", "timestamp"])
    # df = feature_factory_manager.feature_factory_dict["content_id"]["TargetEncoder"].all_predict(df)
    feature_factory_manager.fit(df, is_first_fit=True)
for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = None
feature_factory_manager.logger = None
with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
    pickle.dump(feature_factory_manager, f)