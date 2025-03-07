import numpy as np
import pandas as pd

import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datetime import datetime as dt
import os
import glob
import pickle
import json
from feature_engineering.feature_factory_for_transformer import FeatureFactoryForTransformer
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    DurationPreviousContent, \
    ElapsedTimeBinningEncoder, \
    UserContentRateEncoder, \
    QuestionQuestionTableEncoder2, \
    PreviousAnswer2
from experiment.common import get_logger
import time
from transformers import AdamW, get_linear_schedule_with_warmup

torch.manual_seed(0)
np.random.seed(0)
is_debug = False
is_make_feature_factory = False
load_pickle = True
epochs = 10
device = torch.device("cuda")

wait_time = 0
model_id = "train_0"


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, n_part=8, max_seq=100, is_test=False, predict_mode=False):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.is_test = is_test
        self.n_part = n_part
        self.predict_mode = predict_mode

        self.user_ids = []
        for user_id in group.keys():
            q = group[user_id][("content_id", "content_type_id")]
            if not is_test:
                self.user_ids.append([user_id, -1])
            else:
                is_val = group[user_id]["is_val"]
                for i in range(len(q)):
                    if is_val[i]:
                        self.user_ids.append([user_id, i+1])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index][0]
        end = self.user_ids[index][1]
        q_ = self.samples[user_id][("content_id", "content_type_id")]
        ua_ = self.samples[user_id]["user_answer"]
        part_ = self.samples[user_id]["part"]
        elapsed_time_ = self.samples[user_id]["prior_question_elapsed_time_bin300"]
        duration_previous_content_ = self.samples[user_id]["duration_previous_content_bin300"]
        qa_ = self.samples[user_id]["answered_correctly"]
        prior_q_ = self.samples[user_id]["prior_question_had_explanation"]
        rate_diff_ = self.samples[user_id]["rating_diff_content_user_id"]

        if not self.is_test:
            seq_len = len(q_)
        else:
            start = np.max([0, end - self.max_seq])
            q_ = q_[start:end]
            part_ = part_[start:end]
            qa_ = qa_[start:end]
            ua_ = ua_[start:end]
            elapsed_time_ = elapsed_time_[start:end]
            duration_previous_content_ = duration_previous_content_[start:end]
            prior_q_ = prior_q_[start:end]
            rate_diff_ = rate_diff_[start:end]
            seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        ua = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        duration_previous_content = np.zeros(self.max_seq, dtype=int)
        prior_q = np.zeros(self.max_seq, dtype=int)
        rate_diff = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            part[:] = part_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
            ua[:] = ua_[-self.max_seq:]
            elapsed_time[:] = elapsed_time_[-self.max_seq:]
            duration_previous_content[:] = duration_previous_content_[-self.max_seq:]
            prior_q[:] = prior_q_[-self.max_seq:]
            rate_diff[:] = rate_diff_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            part[-seq_len:] = part_
            qa[-seq_len:] = qa_
            ua[-seq_len:] = ua_
            elapsed_time[-seq_len:] = elapsed_time_
            duration_previous_content[-seq_len:] = duration_previous_content_
            prior_q[-seq_len:] = prior_q_
            rate_diff[-seq_len:] = rate_diff_

        target_id = q[1:]
        part = part[1:]
        elapsed_time = elapsed_time[1:]
        duration_previous_content = duration_previous_content[1:]
        label = qa[1:]
        prior_q = prior_q[1:] # 0: nan, 1: lecture 2: False 3: True
        ua = ua[:-1] + 1 # 0: nan, 1: lecture, 2~5: answer
        x = qa[:-1].copy() + 2 # 1: lecture 2: not correct 3: correct
        rate_diff = rate_diff[1:]

        return {
            "x": x,
            "user_answer": ua,
            "target_id": target_id,
            "part": part,
            "elapsed_time": elapsed_time,
            "duration_previous_content": duration_previous_content,
            "label": label,
            "prior_q": prior_q,
            "rate_diff": rate_diff,
        }

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.ln1 = nn.LayerNorm(state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.ln2 = nn.LayerNorm(state_size)

    def forward(self, x):
        x = self.lr1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.lr2(x)
        x = self.ln2(x)
        return x


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, num_heads=8, dropout=0.2):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim
        embed_dim = 32*6 + 256

        self.embedding = nn.Embedding(4, 32)
        self.user_answer_embedding = nn.Embedding(6, 32)
        self.prior_question_had_explanation_embedding = nn.Embedding(4, 32)
        self.e_embedding = nn.Embedding(n_skill + 1, 256)
        self.part_embedding = nn.Embedding(8, 32)
        self.elapsed_time_embedding = nn.Embedding(302, 32)
        self.duration_previous_content_embedding = nn.Embedding(302, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim)

        self.continuous_embedding = nn.Sequential(
            nn.BatchNorm1d(99),
            nn.Linear(1, embed_dim//2),
            nn.LayerNorm(embed_dim//2)
        )
        self.cat_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2)
        )

        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.dropout = nn.Dropout(dropout/2)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids, parts, elapsed_time, duration_previous_content, prior_q, user_answer,
                rate_diff):
        device = x.device
        att_mask = future_mask(x.size(1)).to(device)

        e = self.e_embedding(question_ids)
        p = self.part_embedding(parts)
        prior_q_emb = self.prior_question_had_explanation_embedding(prior_q)
        user_answer_emb = self.user_answer_embedding(user_answer)

        # decoder
        x = self.embedding(x)
        el_time_emb = self.elapsed_time_embedding(elapsed_time)
        dur_emb = self.duration_previous_content_embedding(duration_previous_content)
        x = torch.cat([x, el_time_emb, dur_emb, e, p, prior_q_emb, user_answer_emb], dim=2)

        cont = rate_diff

        cont_emb = self.continuous_embedding(cont.view(x.size(0), x.size(1), -1))
        x = self.cat_embedding(x)
        x = torch.cat([x, cont_emb], dim=2)

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]

        att_dec = self.transformer_enc(x,
                                       mask=att_mask)
        att_dec, _ = self.gru(att_dec)
        att_dec = att_dec.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.layer_normal(att_dec)
        x = self.ffn(x) + att_dec
        x = self.dropout(x)
        x = self.pred(x)

        return x.squeeze(-1)


def train_epoch(model, train_iterator, val_iterator, optim, criterion, scheduler, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        optim.zero_grad()
        x = item["x"].to(device).long()
        target_id = item["target_id"].to(device).long()
        part = item["part"].to(device).long()
        label = item["label"].to(device).float()
        elapsed_time = item["elapsed_time"].to(device).long()
        duration_previous_content = item["duration_previous_content"].to(device).long()
        prior_question_had_explanation = item["prior_q"].to(device).long()
        user_answer = item["user_answer"].to(device).long()
        rate_diff = item["rate_diff"].to(device).float()

        output = model(x, target_id, part, elapsed_time,
                       duration_previous_content, prior_question_had_explanation, user_answer,
                       rate_diff)
        target_idx = (label.view(-1) >= 0).nonzero()
        loss = criterion(output.view(-1)[target_idx], label.view(-1)[target_idx])
        loss.backward()
        optim.step()
        scheduler.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1]
        target_idx = (label.view(-1) >= 0).nonzero()

        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred.view(-1)[target_idx] == label.view(-1)[target_idx]).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1)[target_idx].data.cpu().numpy())
        outs.extend(output.view(-1)[target_idx].data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    preds = []
    labels = []
    model.eval()
    i = 0
    with torch.no_grad():
        for item in tqdm(val_iterator):
            x = item["x"].to(device).long()
            target_id = item["target_id"].to(device).long()
            part = item["part"].to(device).long()
            label = item["label"].to(device).float()
            elapsed_time = item["elapsed_time"].to(device).long()
            duration_previous_content = item["duration_previous_content"].to(device).long()
            prior_question_had_explanation = item["prior_q"].to(device).long()
            user_answer = item["user_answer"].to(device).long()
            rate_diff = item["rate_diff"].to(device).float()

            output = model(x, target_id, part, elapsed_time,
                           duration_previous_content, prior_question_had_explanation, user_answer,
                           rate_diff)

            preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
            labels.extend(label[:, -1].view(-1).data.cpu().numpy())
            i += 1
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val

def main(params: dict,
         output_dir: str):
    import mlflow
    print("start params={}".format(params))
    logger = get_logger()
    df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
    # df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(30000)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1)
    column_config = {
        ("content_id", "content_type_id"): {"type": "category"},
        "user_answer": {"type": "leakage_feature"},
        "answered_correctly": {"type": "leakage_feature"},
        "part": {"type": "category"},
        "prior_question_elapsed_time_bin300": {"type": "category"},
        "duration_previous_content_bin300": {"type": "category"},
        "prior_question_had_explanation": {"type": "category"},
        "rating_diff_content_user_id": {"type": "numeric"}
    }


    if not load_pickle or is_debug:
        feature_factory_dict = {"user_id": {}}
        feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent()
        feature_factory_dict["user_id"]["ElapsedTimeBinningEncoder"] = ElapsedTimeBinningEncoder()
        feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(rate_func="elo",
                                                                                           column="user_id")
        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger,
                                                        split_num=1,
                                                        model_id=model_id,
                                                        load_feature=not is_debug,
                                                        save_feature=not is_debug)

        print("all_predict")
        df = feature_factory_manager.all_predict(df)
        df = df[["user_id", "content_id", "content_type_id", "part", "user_answer", "answered_correctly",
                 "prior_question_elapsed_time_bin300", "duration_previous_content_bin300",
                 "prior_question_had_explanation", "rating_diff_content_user_id"]]
        df["rating_diff_content_user_id"] = df["rating_diff_content_user_id"].fillna(1500)
        print(df.head(10))

        print("data preprocess")

        train_idx = []
        val_idx = []
        np.random.seed(0)
        for _, w_df in df[df["content_type_id"] == 0].groupby("user_id"):
            if np.random.random() < 0.01:
                # all val
                val_idx.extend(w_df.index.tolist())
            else:
                train_num = int(len(w_df) * 0.95)
                train_idx.extend(w_df[:train_num].index.tolist())
                val_idx.extend(w_df[train_num:].index.tolist())
    ff_for_transformer = FeatureFactoryForTransformer(column_config=column_config,
                                                      dict_path="../feature_engineering/",
                                                      sequence_length=params["max_seq"],
                                                      logger=logger)
    ff_for_transformer.make_dict(df=df)
    n_skill = len(ff_for_transformer.embbed_dict[("content_id", "content_type_id")])
    if not load_pickle or is_debug:
        df["is_val"] = 0
        df["is_val"].loc[val_idx] = 1
        w_df = df[df["is_val"] == 0]
        w_df["group"] = (w_df.groupby("user_id")["user_id"].transform("count") - w_df.groupby("user_id").cumcount()) // params["max_seq"]
        w_df["user_id"] = w_df["user_id"].astype(str) + "_" + w_df["group"].astype(str)

        group = ff_for_transformer.all_predict(w_df)

        dataset_train = SAKTDataset(group,
                                    n_skill=n_skill,
                                    max_seq=params["max_seq"])

        del w_df
        gc.collect()

    ff_for_transformer = FeatureFactoryForTransformer(column_config=column_config,
                                                      dict_path="../feature_engineering/",
                                                      sequence_length=params["max_seq"],
                                                      logger=logger)
    if not load_pickle or is_debug:
        group = ff_for_transformer.all_predict(df[df["content_type_id"] == 0])
        dataset_val = SAKTDataset(group,
                                  is_test=True,
                                  n_skill=n_skill,
                                  max_seq=params["max_seq"])

    os.makedirs("../input/feature_engineering/model114", exist_ok=True)
    if not is_debug and not load_pickle:
        with open(f"../input/feature_engineering/model114/train.pickle", "wb") as f:
            pickle.dump(dataset_train, f)
        with open(f"../input/feature_engineering/model114/val.pickle", "wb") as f:
            pickle.dump(dataset_val, f)

    if not is_debug and load_pickle:
        with open(f"../input/feature_engineering/model114/train.pickle", "rb") as f:
            dataset_train = pickle.load(f)
        with open(f"../input/feature_engineering/model114/val.pickle", "rb") as f:
            dataset_val = pickle.load(f)
        print("loaded!")
    dataloader_train = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=params["batch_size"], shuffle=False, num_workers=1)

    model = SAKTModel(n_skill, embed_dim=params["embed_dim"], max_seq=params["max_seq"], dropout=dropout)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=params["lr"],
                      weight_decay=0.01,
                      )
    num_train_optimization_steps = int(len(dataloader_train) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params["num_warmup_steps"],
                                                num_training_steps=num_train_optimization_steps)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        loss, acc, auc, auc_val = train_epoch(model, dataloader_train, dataloader_val, optimizer, criterion, scheduler, device)
        print("epoch - {} train_loss - {:.3f} auc - {:.4f} auc-val: {:.4f}".format(epoch, loss, auc, auc_val))

    preds = []
    labels = []
    with torch.no_grad():
        for item in tqdm(dataloader_val):
            x = item["x"].to(device).long()
            target_id = item["target_id"].to(device).long()
            part = item["part"].to(device).long()
            label = item["label"].to(device).float()
            elapsed_time = item["elapsed_time"].to(device).long()
            duration_previous_content = item["duration_previous_content"].to(device).long()
            prior_question_had_explanation = item["prior_q"].to(device).long()
            user_answer = item["user_answer"].to(device).long()
            rate_diff = item["rate_diff"].to(device).float()

            output = model(x, target_id, part, elapsed_time,
                           duration_previous_content, prior_question_had_explanation, user_answer,
                           rate_diff)

            preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
            labels.extend(label[:, -1].view(-1).data.cpu().numpy().tolist())

    auc_transformer = roc_auc_score(labels, preds)
    print("single transformer: {:.4f}".format(auc_transformer))
    df_oof = pd.DataFrame()
    # df_oof["row_id"] = df.loc[val_idx].index
    print(len(dataloader_val))
    print(len(preds))
    df_oof["predict"] = preds
    df_oof["target"] = labels

    df_oof.to_csv(f"{output_dir}/transformers1.csv", index=False)
    """
    df_oof2 = pd.read_csv("../output/ex_237/20201213110353/oof_train_0_lgbm.csv")
    df_oof2.columns = ["row_id", "predict_lgbm", "target"]
    df_oof2 = pd.merge(df_oof, df_oof2, how="inner")

    auc_lgbm = roc_auc_score(df_oof2["target"].values, df_oof2["predict_lgbm"].values)
    print("lgbm: {:.4f}".format(auc_lgbm))

    print("ensemble")
    max_auc = 0
    max_nn_ratio = 0
    for r in np.arange(0, 1.05, 0.05):
        auc = roc_auc_score(df_oof2["target"].values, df_oof2["predict_lgbm"].values*(1-r) + df_oof2["predict"].values*r)
        print("[nn_ratio: {:.2f}] AUC: {:.4f}".format(r, auc))

        if max_auc < auc:
            max_auc = auc
            max_nn_ratio = r
    print(len(df_oof2))
    """
    if not is_debug:
        mlflow.start_run(experiment_id=10,
                         run_name=os.path.basename(__file__))

        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("auc_val", auc_transformer)
        mlflow.end_run()
    torch.save(model.state_dict(), f"{output_dir}/transformers.pth")
    del model
    torch.cuda.empty_cache()
    with open(f"{output_dir}/transformer_param.json", "w") as f:
        json.dump(params, f)
    if is_make_feature_factory:
        # feature factory
        feature_factory_dict = {"user_id": {}}
        feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
        feature_factory_dict["user_id"]["ElapsedTimeBinningEncoder"] = ElapsedTimeBinningEncoder()
        feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(rate_func="elo",
                                                                                           column="user_id")
        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger,
                                                        split_num=1,
                                                        model_id=model_id,
                                                        load_feature=not is_debug,
                                                        save_feature=not is_debug)

        ff_for_transformer = FeatureFactoryForTransformer(column_config=column_config,
                                                          dict_path="../feature_engineering/",
                                                          sequence_length=params["max_seq"],
                                                          logger=logger)
        df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
        if is_debug:
            df = df.head(10000)
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        feature_factory_manager.fit(df)
        df = feature_factory_manager.all_predict(df)
        for dicts in feature_factory_manager.feature_factory_dict.values():
            for factory in dicts.values():
                factory.logger = None
        feature_factory_manager.logger = None
        with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
            pickle.dump(feature_factory_manager, f)

        ff_for_transformer.fit(df)
        ff_for_transformer.logger = None
        with open(f"{output_dir}/feature_factory_manager_for_transformer.pickle", "wb") as f:
            pickle.dump(ff_for_transformer, f)

if __name__ == "__main__":
    if not is_debug:
        for _ in tqdm(range(wait_time)):
            time.sleep(1)
    output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
    os.makedirs(output_dir, exist_ok=True)
    for lr in [1e-3]:
        for dropout in [0.5]:
            if is_debug:
                batch_size = 8
            else:
                batch_size = 128
            params = {"embed_dim": 256,
                      "max_seq": 100,
                      "batch_size": batch_size,
                      "num_warmup_steps": 1000,
                      "lr": lr,
                      "dropout": dropout}
            main(params, output_dir=output_dir)