from datetime import datetime as dt
from pipeline.p_001_baseline import transform
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

data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float16',
    'prior_question_had_explanation': 'float16'
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
        kaggle=False,
        rewrite=False):

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
    data_dir = "../work_csv"
    if rewrite:
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        print(glob.glob(files_dir))
        for model_id, fname in enumerate(glob.glob(files_dir)):
            logger.info(f"loading... {fname}")
            df = pd.read_pickle(fname)
            if debug:
                df = df.head(1000)
            df = transform(df)

            for user_id, w_df in tqdm.tqdm(df.groupby("user_id")):
                w_df.to_pickle(f"{data_dir}/{user_id}.pickle")

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

            for user_id, df_prev in df_test_prev.groupby("user_id"):
                fname = f"{data_dir}/{user_id}.pickle"
                if os.path.isfile(fname):
                    df = pd.read_pickle(f"{data_dir}/{user_id}.pickle")
                    df = pd.concat([df, df_prev[df.columns]])
                else:
                    df = df_prev
                df.to_pickle(f"{data_dir}/{user_id}.pickle")

        # 今回のデータ取得&計算
        dfs = []

        df_nows = []
        for user_id, df_now in df_test.groupby("user_id"):
            fname = f"{data_dir}/{user_id}.pickle"
            if os.path.isfile(fname):
                df = pd.read_pickle(fname)
                df = pd.concat([df, df_now])
            else:
                df = df_now[:]
                df["user_answer"] = -1
                df["answered_correctly"] = -1
                df = df.astype(data_types_dict)
            df = transform(df)
            predicts = []
            cols = models[0].feature_name()
            for col in cols:
                if col not in df.columns:
                    df[col] = -99999
            for model in models:
                predicts.append(model.predict(df[cols].iloc[-len(df_now):]))

            df_now["answered_correctly"] = np.array(predicts).transpose().mean(axis=1)
            dfs.append(df[cols].iloc[-len(df_now):].drop("row_id", axis=1, errors="ignore"))
            df_nows.append(df_now)

        # predict
        df_nows = pd.concat(df_nows)
        df_sample_prediction = pd.merge(df_sample_prediction[["row_id"]],
                                        df_nows[["row_id", "answered_correctly"]],
                                        how="inner")
        env.predict(df_sample_prediction)
        df_test_prev = pd.concat(dfs)
        print(df_test_prev)

if __name__ == "__main__":
    run(debug=False,
        model_dir="../output/ex_001/20201011201811")