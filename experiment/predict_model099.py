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
import json
import torch
from torch.utils.data import Dataset, DataLoader

from experiment.model099 import SAKTModel, SAKTDataset
warnings.filterwarnings("ignore")

model_dir = "../output/model051/20201214105333"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


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
    # params
    with open(f"{model_dir}/transformer_param.json", "r") as f:
        params = json.load(f)
    # model loading
    model_path = f"{model_dir}/transformers.pth"
    model = SAKTModel(13938, embed_dim=params["embed_dim"], max_seq=params["max_seq"])
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # load feature_factory_manager
    logger = get_logger()
    ff_manager_path_for_transformer = f"{model_dir}/feature_factory_manager_for_transformer.pickle"
    with open(ff_manager_path_for_transformer, "rb") as f:
        feature_factory_manager_for_transformer = pickle.load(f)
    feature_factory_manager_for_transformer.logger = logger

    ff_manager_path = f"{model_dir}/feature_factory_manager.pickle"
    with open(ff_manager_path, "rb") as f:
        feature_factory_manager = pickle.load(f)
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
            df_test_prev = df_test_prev[df_test_prev["answered_correctly"] != -1]
            df_test_prev["answered_correctly"] = df_test_prev["answered_correctly"].replace(-1, np.nan)
            df_test_prev["prior_question_had_explanation"] = df_test_prev["prior_question_had_explanation"].fillna(-1).astype("int8")

            logger.info("fit data")
            feature_factory_manager_for_transformer.fit(df_test_prev)

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

        df_test["previous_answer_content_id"] = -1
        df_test = feature_factory_manager.partial_predict(df_test)
        df_test["qq_table2_mean"] = df_test["qq_table2_mean"].fillna(0.65)
        df_test["qq_table2_min"] = df_test["qq_table2_min"].fillna(0.6)
        df_test["previous_answer_content_id"] = -1
        group = feature_factory_manager_for_transformer.partial_predict(df_test[df_test["content_type_id"] == 0])
        logger.info(f"------ predict ------")

        dataset_val = SAKTDataset(group, 13939, predict_mode=True, max_seq=params["max_seq"])
        dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=1)

        predicts = []
        for item in dataloader_val:
            x = item["x"].to(device).long()
            target_id = item["target_id"].to(device).long()
            part = item["part"].to(device).long()
            label = item["label"].to(device).float()
            elapsed_time = item["elapsed_time"].to(device).long()
            duration_previous_content = item["duration_previous_content"].to(device).long()
            prior_question_had_explanation = item["prior_q"].to(device).long()
            user_answer = item["user_answer"].to(device).long()
            rate_diff = item["rate_diff"].to(device).float()
            qq_table_mean = item["qq_table_mean"].to(device).float()
            qq_table_min = item["qq_table_min"].to(device).float()

            output = model(x, target_id, part, elapsed_time,
                           duration_previous_content, prior_question_had_explanation, user_answer,
                           rate_diff, qq_table_mean, qq_table_min)
            predicts.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())

        logger.info("------ other ------")
        df_sample_prediction = df_test[df_test["content_type_id"] == 0][["row_id"]]
        df_sample_prediction["answered_correctly"] = predicts
        env.predict(df_sample_prediction)
        df_test_prev.append(df_test)
        df_test_prev_rows += len(df_test)
        if i < 5:
            df_test.to_csv(f"{i}.csv")
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