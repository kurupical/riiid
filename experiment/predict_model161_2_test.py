import pandas as pd
import pickle
import numpy as np
from logging import Logger, StreamHandler, Formatter
import warnings
import json
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

model_dir = "../output/model161/20201222082506_2"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from experiment.model161 import SAKTModel, SAKTDataset

def get_logger():
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class EmptyLogger:
    def __init__(self):
        pass

    def info(self, s):
        pass


import psutil
import os
import time
import sys
import math
import gc
from contextlib import contextmanager

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


class KurupicalModel:
    def __init__(self,
                 model_dir: str,
                 verbose: bool):
        if verbose:
            logger = get_logger()
        else:
            logger = EmptyLogger()
        with trace("load csv"):
            self.df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                          dtype={"bundle_id": "int32",
                                                 "question_id": "int32",
                                                 "correct_answer": "int8",
                                                 "part": "int8"})
            self.df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                                         dtype={"lecture_id": "int32",
                                                "tag": "int16",
                                                "part": "int8"})
            # params
        with trace("load model1"):
            with open(f"{model_dir}/transformer_param.json", "r") as f:
                self.params = json.load(f)
            # model loading
            model_path = f"{model_dir}/transformers.pth"
            self.model = SAKTModel(13938, embed_dim=self.params["embed_dim"], max_seq=self.params["max_seq"], cont_emb=8)

        with trace("model.load_state_dict"):
            self.model.load_state_dict(torch.load(model_path))

        with trace("model.to(cuda)"):
            self.model.to(device)

        with trace("load model4"):
            self.model.eval()



        with trace("load seq"):
            # load feature_factory_manager
            ff_manager_path_for_transformer = f"{model_dir}/feature_factory_manager_for_transformer.pickle"
            with open(ff_manager_path_for_transformer, "rb") as f:
                self.feature_factory_manager_for_transformer = pickle.load(f)
            self.feature_factory_manager_for_transformer.logger = logger

        with trace("load manager"):
            ff_manager_path = f"{model_dir}/feature_factory_manager.pickle"
            with open(ff_manager_path, "rb") as f:
                self.feature_factory_manager = pickle.load(f)
            self.feature_factory_manager.logger = logger

        with trace("gc collect"):
            gc.collect()

    def update(self, df_test_prev):
        """
        辞書のupdate
        前回のdf_testをそのまま与える
        """

        df_test_prev["answered_correctly"] = df_test_prev["answered_correctly"].replace(-1, np.nan)
        df_test_prev["prior_question_had_explanation"] = df_test_prev["prior_question_had_explanation"].fillna(-1).astype("int8")

        self.feature_factory_manager.fit(df_test_prev)
        df_test_prev["answered_correctly"] = df_test_prev["answered_correctly"].replace(np.nan, -1)
        self.feature_factory_manager_for_transformer.fit(df_test_prev)

    def predict(self, df_test):
        """
        予測
        df_testをそのまま与える
        return: np.array(batch_size)
        """
        w_df1 = pd.merge(df_test[df_test["content_type_id"] == 0], self.df_question, how="left", left_on="content_id",
                         right_on="question_id")
        w_df2 = pd.merge(df_test[df_test["content_type_id"] == 1], self.df_lecture, how="left", left_on="content_id",
                         right_on="lecture_id")
        df_test = pd.concat([w_df1, w_df2]).sort_values(["user_id", "timestamp"]).sort_index()
        df_test["tag"] = df_test["tag"].fillna(-1)
        df_test["correct_answer"] = df_test["correct_answer"].fillna(-1)
        df_test["bundle_id"] = df_test["bundle_id"].fillna(-1)
        df_test["task_container_id_bin300"] = [x if x < 300 else 300 for x in df_test["task_container_id"].values]

        df_test["prior_question_had_explanation"] = df_test["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")

        df_test = self.feature_factory_manager.partial_predict(df_test)
        group = self.feature_factory_manager_for_transformer.partial_predict(df_test[df_test["content_type_id"] == 0])

        dataset_val = SAKTDataset(group, 13939, predict_mode=True, max_seq=self.params["max_seq"])
        dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=1)

        predicts = []

        with torch.no_grad():
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
                container_id = item["container_id"].to(device).long()
                prev_ans_idx = item["previous_answer_index_content_id"].to(device).long()
                prev_answer_content_id = item["previous_answer_content_id"].to(device).long()

                print(item)
                output = self.model(x, target_id, part, elapsed_time,
                                    duration_previous_content, prior_question_had_explanation, user_answer,
                                    rate_diff, container_id, prev_ans_idx, prev_answer_content_id)
                predicts.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
                print(predicts)

        return np.array(predicts), df_test

def run(model_dir,
        verbose=False):
    model = KurupicalModel(model_dir=model_dir,
                           verbose=verbose)
    # environment
    logger = get_logger()
    df_test_prev = None
    df_all = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", nrows=1000)
    df_all = df_all.reset_index()
    df_all["index"] = df_all["index"] // 100

    for _, df_test in df_all.groupby(["index"]):
        if verbose: logger.info("inference!")
        if df_test_prev is not None:
            model.update(df_test_prev)

        predicts, df_test_prev = model.predict(df_test)

        df_sample_prediction = df_test[df_test["content_type_id"] == 0][["row_id"]]
        df_sample_prediction["answered_correctly"] = predicts

if __name__ == "__main__":
    run(model_dir=model_dir,
        verbose=True)