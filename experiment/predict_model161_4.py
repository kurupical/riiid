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

warnings.filterwarnings("ignore")

model_dir = "../output/model161/20201222082506_2"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from experiment.model208_4 import SAKTModel, SAKTDataset

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


class KurupicalModel:
    def __init__(self,
                 model_dir: str,
                 verbose: bool=False):
        if verbose:
            self.logger = get_logger()
        else:
            self.logger = EmptyLogger()
        self.df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                                      dtype={"bundle_id": "int32",
                                             "question_id": "int32",
                                             "correct_answer": "int8",
                                             "part": "int8"})
        self.df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                                     dtype={"lecture_id": "int32",
                                            "tag": "int16",
                                            "part": "int8"})

        self.part_dict = {} # key: (content_id, content_type_id), value: part
        for x in self.df_question[["question_id", "part"]].values:
            question_id = x[0]
            part = x[1]
            self.part_dict[(question_id, 0)] = part

        for x in self.df_lecture[["lecture_id", "part"]].values:
            lecture_id = x[0]
            part = x[1]
            self.part_dict[(lecture_id, 1)] = part

        # params
        with open(f"{model_dir}/transformer_param.json", "r") as f:
            self.params = json.load(f)
        # model loading
        model_path = f"{model_dir}/transformers.pth"
        self.model = SAKTModel(13938, embed_dim=self.params["embed_dim"], max_seq=self.params["max_seq"], cont_emb=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

        # load feature_factory_manager
        ff_manager_path_for_transformer = f"{model_dir}/feature_factory_manager_for_transformer.pickle"
        with open(ff_manager_path_for_transformer, "rb") as f:
            self.feature_factory_manager_for_transformer = pickle.load(f)
        self.feature_factory_manager_for_transformer.logger = self.logger

        ff_manager_path = f"{model_dir}/feature_factory_manager.pickle"
        with open(ff_manager_path, "rb") as f:
            self.feature_factory_manager = pickle.load(f)
        self.feature_factory_manager.logger = self.logger

    def update(self,
               df_test_prev,
               df_test):
        """
        辞書のupdate
        前回のdf_testをそのまま与える
        """
        df_test_prev = df_test_prev.copy()
        answered_correctly = df_test.iloc[0]["prior_group_answers_correct"]
        user_answer = df_test.iloc[0]["prior_group_responses"]
        answered_correctly = \
            [int(x) for x in answered_correctly.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")]
        user_answer = \
            [int(x) for x in user_answer.replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")]

        df_test_prev["answered_correctly"] = answered_correctly
        df_test_prev["user_answer"] = user_answer
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
        self.logger.info("------------ start! ------------")
        df_test = df_test.copy()

        self.logger.info("preprocess")
        df_test["part"] = [self.part_dict[(x[0], x[1])] for x in df_test[["content_id", "content_type_id"]].values]
        df_test["task_container_id_bin300"] = [x if x < 300 else 300 for x in df_test["task_container_id"].values]
        df_test["prior_question_had_explanation"] = df_test["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")

        self.logger.info("feature_factory_manager.partial_predict")
        df_test = self.feature_factory_manager.partial_predict(df_test)
        self.logger.info("feature_factory_manager_for_transformer.partial_predict")
        group = self.feature_factory_manager_for_transformer.partial_predict(df_test[df_test["content_type_id"] == 0])

        self.logger.info("make_dataset")
        dataset_val = SAKTDataset(group, 13939, predict_mode=True, max_seq=self.params["max_seq"])
        self.logger.info("make_dataloader")
        dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False)

        predicts = []

        self.logger.info("no_grad_setting")
        with torch.no_grad():
            self.logger.info("item")
            for item in dataloader_val:
                self.logger.info("data_preparing")
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

                self.logger.info("predict")
                output = self.model(x, target_id, part, elapsed_time,
                                    duration_previous_content, prior_question_had_explanation, user_answer,
                                    rate_diff, container_id, prev_ans_idx, prev_answer_content_id)

                self.logger.info("postprocessing")
                predicts.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())

        return np.array(predicts), df_test

def run(model_dir,
        verbose=False):
    model = KurupicalModel(model_dir=model_dir, verbose=verbose)
    # environment
    env = riiideducation.make_env()

    logger = get_logger()
    iter_test = env.iter_test()
    df_test_prev = None
    for (df_test, df_sample_prediction) in iter_test:
        if verbose: logger.info("inference!")
        if df_test_prev is not None:
            model.update(df_test_prev, df_test)

        predicts, df_test_prev = model.predict(df_test)

        df_sample_prediction = df_test[df_test["content_type_id"] == 0][["row_id"]]
        df_sample_prediction["answered_correctly"] = predicts
        env.predict(df_sample_prediction)

if __name__ == "__main__":
    run(model_dir=model_dir,
        verbose=True)