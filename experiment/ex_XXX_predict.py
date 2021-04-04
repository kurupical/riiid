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
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder, \
    PriorQuestionElapsedTimeBinningEncoder
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

model_dir = "../output/ex_253/20201214130401"

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
        update_record,
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

    # load feature_factory_manager
    logger = get_logger()
    ff_manager_path = f"{model_dir}/feature_factory_manager.pickle"
    with open(ff_manager_path, "rb") as f:
        feature_factory_manager = pickle.load(f)
    for dicts in feature_factory_manager.feature_factory_dict.values():
        for factory in dicts.values():
            factory.logger = logger
    feature_factory_manager.logger = logger

    iter_test = env.iter_test()
    df_test_prev = []
    df_test_prev_rows = 0
    answered_correctlies = []
    user_answers = []
    i = 0
    for (df_test, df_sample_prediction) in iter_test:
        i += 1
        logger.info(f"[iteration {i}: data_length: {len(df_test)}")
        # 前回のデータ更新
        if df_test_prev_rows > 0: # 初回のみパスするためのif
            answered_correctly = df_test.iloc[0]["prior_group_answers_correct"]
            user_answer = df_test.iloc[0]["prior_group_responses"]
            answered_correctlies.extend([int(x) for x in answered_correctly.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")])
            user_answers.extend([int(x) for x in user_answer.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")])

        if debug:
            update_record = 1
        if df_test_prev_rows > update_record:
            logger.info("------ fitting ------")
            logger.info("concat df")
            df_test_prev = pd.concat(df_test_prev)
            df_test_prev["answered_correctly"] = answered_correctlies
            df_test_prev["user_answer"] = user_answers
            # df_test_prev = df_test_prev.drop(prior_columns, axis=1)
            # df_test_prev = df_test_prev[df_test_prev["answered_correctly"] != -1]
            df_test_prev["answered_correctly"] = df_test_prev["answered_correctly"].replace(-1, np.nan)
            df_test_prev["prior_question_had_explanation"] = df_test_prev["prior_question_had_explanation"].fillna(-1).astype("int8")

            logger.info("fit data")
            feature_factory_manager.fit(df_test_prev)

            df_test_prev = []
            df_test_prev_rows = 0
            answered_correctlies = []
            user_answers = []
        # 今回のデータ取得&計算

        # logger.info(f"[time: {int(time.time() - t)}dataload")
        logger.info(f"------ question&lecture merge ------")
        w_df1 = pd.merge(df_test[df_test["content_type_id"] == 0], df_question, how="left", left_on="content_id",
                         right_on="question_id")
        w_df2 = pd.merge(df_test[df_test["content_type_id"] == 1], df_lecture, how="left", left_on="content_id",
                         right_on="lecture_id")
        df_test = pd.concat([w_df1, w_df2]).sort_values(["user_id", "timestamp"]).sort_index()
        df_test["tag"] = df_test["tag"].fillna(-1)
        df_test["correct_answer"] = df_test["correct_answer"].fillna(-1)
        df_test["bundle_id"] = df_test["bundle_id"].fillna(-1)

        logger.info(f"------ transform ------ ")
        df_test["prior_question_had_explanation"] = df_test["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")

        df = feature_factory_manager.partial_predict(df_test)
        df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df.columns]
        logger.info(f"------ predict ------")


        # predict
        predicts = []
        cols = models[0].feature_name()
        w_df = df[cols]
        for model in models:
            predicts.append(model.predict(w_df))

        logger.info("------ other ------")
        df["answered_correctly"] = np.array(predicts).transpose().mean(axis=1)
        df_sample_prediction = df[df["content_type_id"] == 0][["row_id", "answered_correctly"]]
        env.predict(df_sample_prediction)
        df_test_prev.append(df[cols + ["user_id", "tags", "content_type_id"]])
        df_test_prev_rows += len(df)
        if i < 5:
            df.to_csv(f"{i}.csv")
        if i == 3:
            class EmptyLogger:
                def __init__(self):
                    pass

                def info(self, s):
                    pass

            logger = EmptyLogger()


if __name__ == "__main__":
    run(debug=True,
        model_dir=model_dir,
        update_record=50)