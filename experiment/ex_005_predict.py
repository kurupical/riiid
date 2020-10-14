from datetime import datetime as dt
from pipeline.p_005_partialfit import Pipeline
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
        files_dir = "../input/riiid-test-answer-prediction/split10/*.pickle"

    logger = get_logger()
    # environment
    env = riiideducation.make_env()

    # model loading
    models = []
    for model_path in glob.glob(f"{model_dir}/*model*.pickle"):
        with open(model_path, "rb") as f:
            models.append(pickle.load(f))

    # data preprocessing
    pipeline = Pipeline(logger=logger)
    for model_id, fname in enumerate(glob.glob(files_dir)):
        logger.info(f"loading... {fname}")
        df = pd.read_pickle(fname)
        if debug:
            df = df.head(1000)
        pipeline.fit(df)

    iter_test = env.iter_test()
    df_test_prev = pd.DataFrame()
    i = 0
    t = time.time()
    for (df_test, df_sample_prediction) in iter_test:
        i += 1
        logger.info(f"[time: {int(time.time() - t)}iteration {i}: data_length: {len(df_test)}")
        # 前回のデータ更新
        if len(df_test_prev) > 0:
            answered_correctly = df_test.iloc[0]["prior_group_answers_correct"]
            user_answer = df_test.iloc[0]["prior_group_responses"]

            df_test_prev["answered_correctly"] = [int(x) for x in answered_correctly.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")]
            df_test_prev["user_answer"] = [int(x) for x in user_answer.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")]
            # df_test_prev = df_test_prev.drop(prior_columns, axis=1)

            pipeline.fit(df_test_prev)
        # 今回のデータ取得&計算

        # logger.info(f"[time: {int(time.time() - t)}dataload")
        logger.info(f"transform... ")
        df = pipeline.partial_transform(df_test)
        logger.info(f"other... ")
        cols = models[0].feature_name()
        for col in cols:
            if col not in df.columns:
                df[col] = -99999

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
        df_test_prev = df[cols]

if __name__ == "__main__":
    run(debug=False,
        model_dir="../output/ex_005/20201013223614")