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
    TargetEncodeVsUserId
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

model_dir = "../output/ex_035/20201031164620"

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
    for column in ["content_id", "content_type_id", "user_id", "prior_question_had_explanation",
                   "tags1", "tags2", "tags3", "tags4", "tags5", "tags6",
                   ("user_id", "content_type_id"), ("user_id", "prior_question_had_explanation")]:
        is_partial_fit = (column == "content_id" or column == "content_type_id")

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
    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    feature_factory_dict["user_id"]["UserLevelEncoder2ContentTypeId"] = UserLevelEncoder2(vs_column="content_type_id")
    feature_factory_dict["content_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="content_id",
                                                                                                  agg_column="prior_question_elapsed_time",
                                                                                                  remove_now=True)
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

    iter_test = env.iter_test()
    df_test_prev = pd.DataFrame()
    answered_correctlies = []
    user_answers = []
    i = 0
    t = time.time()
    for (df_test, df_sample_prediction) in iter_test:
        i += 1
        logger.info(f"[time: {int(time.time() - t)}iteration {i}: data_length: {len(df_test)}")
        # 前回のデータ更新
        if len(df_test_prev) > 0: # 初回のみパスするためのif
            answered_correctly = df_test.iloc[0]["prior_group_answers_correct"]
            user_answer = df_test.iloc[0]["prior_group_responses"]
            answered_correctlies.extend([int(x) for x in answered_correctly.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")])
            user_answers.extend([int(x) for x in user_answer.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")])

        if debug:
            update_record = 1
        else:
            update_record = 50
        if len(df_test_prev) > update_record:
            df_test_prev["answered_correctly"] = answered_correctlies
            df_test_prev["user_answer"] = user_answers
            # df_test_prev = df_test_prev.drop(prior_columns, axis=1)
            df_test_prev = df_test_prev[df_test_prev["answered_correctly"] != -1]
            df_test_prev["answered_correctly"] = df_test_prev["answered_correctly"].replace(-1, np.nan)
            df_test_prev["prior_question_had_explanation"] = df_test_prev["prior_question_had_explanation"].fillna(-1).astype("int8")

            feature_factory_manager.fit(df_test_prev)

            df_test_prev = pd.DataFrame()
            answered_correctlies = []
            user_answers = []
        # 今回のデータ取得&計算

        # logger.info(f"[time: {int(time.time() - t)}dataload")
        logger.info(f"merge... ")
        w_df1 = pd.merge(df_test[df_test["content_type_id"] == 0], df_question, how="left", left_on="content_id",
                         right_on="question_id")
        w_df2 = pd.merge(df_test[df_test["content_type_id"] == 1], df_lecture, how="left", left_on="content_id",
                         right_on="lecture_id")
        df_test = pd.concat([w_df1, w_df2]).sort_values(["user_id", "timestamp"]).sort_index()
        df_test["tag"] = df_test["tag"].fillna(-1)
        df_test["correct_answer"] = df_test["correct_answer"].fillna(-1)
        df_test["bundle_id"] = df_test["bundle_id"].fillna(-1)

        logger.info(f"transform... ")
        df_test["prior_question_had_explanation"] = df_test["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")

        df = feature_factory_manager.partial_predict(df_test)
        df.columns = [x.replace(" ", "_") for x in df.columns]
        logger.info(f"other... ")

        # predict
        predicts = []
        cols = models[0].feature_name()
        for model in models:
            predicts.append(model.predict(df[cols]))

        df["answered_correctly"] = np.array(predicts).transpose().mean(axis=1)
        df_sample_prediction = pd.merge(df_sample_prediction[["row_id"]],
                                        df[["row_id", "answered_correctly"]],
                                        how="inner")
        env.predict(df_sample_prediction)
        df_test_prev = df_test_prev.append(df[cols + ["user_id", "tags"]])
        if i < 5:
            df_test_prev.to_csv(f"{i}.csv")

if __name__ == "__main__":
    run(debug=True,
        model_dir=model_dir)