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
from model.optim import AdaBelief

torch.manual_seed(0)
np.random.seed(0)
is_debug = False
is_make_feature_factory = False
load_pickle = True
epochs = 12
device = torch.device("cuda")

wait_time = 0
len_train = 10**9  # ALL

load_feature_dir = [
    "../../riiid_takoi/notebook/fe/fe018_part.feather",
    "../../riiid_takoi/notebook/fe/local_fe019.feather",
    "../../riiid_takoi/notebook/fe/local_fe095.feather",
    "../../riiid_takoi/notebook/fe/fe032.feather",
]

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
            q = group[user_id]["content_id_with_lecture"]
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
        q_ = self.samples[user_id]["content_id_with_lecture"]
        part_ = self.samples[user_id]["part"]
        elapsed_time_ = self.samples[user_id]["prior_question_elapsed_time"]
        elapsed_time_ = np.clip(np.array(elapsed_time_)//1000, None, 300)
        timestamp_delta_ = self.samples[user_id]["timestamp_delta"]
        timestamp_delta_ = np.clip(np.array(timestamp_delta_)//1000, None, 300)
        qa_ = self.samples[user_id]["answered_correctly"]
        prior_q_ = self.samples[user_id]["prior_question_had_explanation"]
        uid_win_rate_ = self.samples[user_id]["uid_win_rate"]
        container_id_ = self.samples[user_id]["task_container_id"]
        container_id_ = np.clip(np.array(container_id_), None, 300)
        content_id_delta_ = self.samples[user_id]["content_id_delta"]
        last_content_id_acc_ = self.samples[user_id]["last_content_id_acc"]

        if not self.is_test:
            seq_len = len(q_)
        else:
            start = np.max([0, end - self.max_seq])
            q_ = q_[start:end]
            part_ = part_[start:end]
            qa_ = qa_[start:end]
            elapsed_time_ = elapsed_time_[start:end]
            timestamp_delta_ = timestamp_delta_[start:end]
            prior_q_ = prior_q_[start:end]
            uid_win_rate_ = uid_win_rate_[start:end]
            container_id_ = container_id_[start:end]
            content_id_delta_ = content_id_delta_[start:end]
            last_content_id_acc_ = last_content_id_acc_[start:end]
            seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        elapsed_time = np.zeros(self.max_seq, dtype=int)
        timestamp_delta = np.zeros(self.max_seq, dtype=int)
        prior_q = np.zeros(self.max_seq, dtype=int)
        uid_win_rate = np.zeros(self.max_seq, dtype=int)
        container_id = np.zeros(self.max_seq, dtype=int)
        content_id_delta = np.zeros(self.max_seq, dtype=int)
        last_content_id_acc = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            part[:] = part_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
            elapsed_time[:] = elapsed_time_[-self.max_seq:]
            timestamp_delta[:] = timestamp_delta_[-self.max_seq:]
            prior_q[:] = prior_q_[-self.max_seq:]
            uid_win_rate[:] = uid_win_rate_[-self.max_seq:]
            container_id[:] = container_id_[-self.max_seq:]
            content_id_delta[:] = content_id_delta_[-self.max_seq:]
            last_content_id_acc[:] = last_content_id_acc_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            part[-seq_len:] = part_
            qa[-seq_len:] = qa_
            elapsed_time[-seq_len:] = elapsed_time_
            timestamp_delta[-seq_len:] = timestamp_delta_
            prior_q[-seq_len:] = prior_q_
            uid_win_rate[-seq_len:] = uid_win_rate_
            container_id[-seq_len:] = container_id_
            content_id_delta[-seq_len:] = content_id_delta_
            last_content_id_acc[-seq_len:] = last_content_id_acc_

        target_id = q[1:]
        part = part[1:]
        elapsed_time = elapsed_time[1:]
        timestamp_delta = timestamp_delta[1:]
        prior_q = prior_q[1:] # 0: nan, 1: lecture 2: False 3: True
        x = qa[:-1].copy() # 1: lecture 2: not correct 3: correct
        uid_win_rate = uid_win_rate[1:]
        container_id = container_id[1:]
        content_id_delta = content_id_delta[1:]
        last_content_id_acc = last_content_id_acc[1:]
        label = qa[1:].copy() - 3

        return {
            "x": x,
            "target_id": target_id,
            "part": part,
            "elapsed_time": elapsed_time,
            "timestamp_delta": timestamp_delta,
            "label": label,
            "prior_q": prior_q,
            "uid_win_rate": uid_win_rate,
            "container_id": container_id,
            "content_id_delta": content_id_delta,
            "last_content_id_acc": last_content_id_acc
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

class ContEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_len):
        super(ContEmbedding, self).__init__()
        self.embed_dim = embed_dim

        self.bn = nn.BatchNorm1d(seq_len-1)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.bn(x)
        x, _ = self.gru(x)
        x = self.ln2(x)
        return x


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

class CatEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(CatEmbedding, self).__init__()
        self.embed_dim = embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.lr = nn.Linear(embed_dim, embed_dim // 2)
        self.ln2 = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.ln1(x)
        x = self.lr(x)
        x = self.ln2(x)
        return x


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, num_heads=8, dropout=0.2,
                 cont_emb=None):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim_cat = embed_dim
        embed_dim_cat_all = 32*8 + embed_dim
        embed_dim_all = embed_dim_cat_all + cont_emb

        self.embedding = nn.Embedding(5, 32)
        self.prior_question_had_explanation_embedding = nn.Embedding(4, 32)
        self.e_embedding = nn.Embedding(n_skill + 1, self.embed_dim_cat)
        self.part_embedding = nn.Embedding(8, 32)
        self.elapsed_time_embedding = nn.Embedding(302, 32)
        self.timestamp_delta_embedding = nn.Embedding(302, 32)
        self.container_embedding = nn.Embedding(302, 32)
        self.content_id_delta_embedding = nn.Embedding(302, 32)
        self.last_content_id_acc_embedding = nn.Embedding(4, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim_all, nhead=num_heads, dropout=dropout)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.gru = nn.GRU(input_size=embed_dim_all, hidden_size=embed_dim_all)

        self.continuous_embedding = ContEmbedding(input_dim=1, embed_dim=cont_emb, seq_len=max_seq)
        self.cat_embedding = nn.Sequential(
            nn.Linear(embed_dim_cat_all, embed_dim_cat_all),
            nn.LayerNorm(embed_dim_cat_all)
        )

        self.layer_normal = nn.LayerNorm(embed_dim_all)

        self.ffn = FFN(embed_dim_all)
        self.dropout = nn.Dropout(dropout/2)
        self.pred = nn.Linear(embed_dim_all, 1)

    def forward(self, item, device):
        x = item["x"].to(device).long()
        question_ids = item["target_id"].to(device).long()
        parts = item["part"].to(device).long()
        elapsed_time = item["elapsed_time"].to(device).long()
        timestamp_delta = item["timestamp_delta"].to(device).long()
        prior_question_had_explanation = item["prior_q"].to(device).long()
        uid_win_rate = item["uid_win_rate"].to(device).float()
        container_id = item["container_id"].to(device).long()
        content_id_delta = item["content_id_delta"].to(device).long()
        last_content_id_acc = item["last_content_id_acc"].to(device).long()

        device = x.device
        att_mask = future_mask(x.size(1)).to(device)

        e = self.e_embedding(question_ids)
        p = self.part_embedding(parts)
        prior_q_emb = self.prior_question_had_explanation_embedding(prior_question_had_explanation)

        # decoder
        x = self.embedding(x)
        el_time_emb = self.elapsed_time_embedding(elapsed_time)
        dur_emb = self.timestamp_delta_embedding(timestamp_delta)
        container_emb = self.container_embedding(container_id)
        content_id_delta_emb = self.content_id_delta_embedding(content_id_delta)
        last_content_id_acc_emb = self.last_content_id_acc_embedding(last_content_id_acc)
        x = torch.cat([x, el_time_emb, dur_emb, e, p, prior_q_emb, container_emb,
                       content_id_delta_emb, last_content_id_acc_emb], dim=2)

        cont = uid_win_rate

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


def train_epoch(model, train_iterator, val_iterator, optim, criterion, scheduler, epoch, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        optim.zero_grad()
        label = item["label"].to(device).float()

        output = model(item, device)
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
            label = item["label"].to(device).float()
            output = model(item, device)
            preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
            labels.extend(label[:, -1].view(-1).data.cpu().numpy())
            i += 1
            if i > 100 and epoch < 3:
                break
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val

def main(params: dict,
         output_dir: str):
    import mlflow
    print("start params={}".format(params))
    df = pd.read_feather("../../riiid_takoi/notebook/data/train_sort.feather").head(len_train)
    # df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(30000)
    for d in load_feature_dir:
        df_ = pd.read_feather(d).head(len_train)
        if is_debug:
            df_ = df_.head(30000)
        df = pd.concat([df, df_], axis=1)

    print(df.isnull().sum())
    # ====================
    # preprocess
    # ====================
    df["content_id"] = df["content_id"] + 2
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1) + 2
    df["content_id_with_lecture"] = df["content_id"]
    df.loc[df["content_type_id"] == 1, "content_id_with_lecture"] = df["content_id"] + 14000
    df["answered_correctly"] += 3
    df["task_container_id"] += 1
    df["part"] += 1
    df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].fillna(0)
    df["timestamp_delta"] = df["timestamp_delta"].fillna(0)
    df["uid_win_rate"] = df["uid_win_rate"].fillna(0.65)  # target_encoding(user_id)

    # 以下は特徴作成時に処理済
    # df["content_id_delta"] = df["content_id_delta"].fillna(-1) + 2
    # df["last_content_id_acc"] = df["last_content_id_acc"].fillna(-1) + 2

    # ====================
    # data prepare
    # ====================
    agg_dict = {
        "content_id_with_lecture": list,
        "prior_question_had_explanation": list,
        "prior_question_elapsed_time": list,
        "answered_correctly": list,
        "task_container_id": list,
        "part": list,
        "content_id_delta": list,
        "last_content_id_acc": list,
        "uid_win_rate": list,
        "is_val": list,
        "timestamp_delta": list,
    }
    df_val_row = pd.read_feather("../../riiid_takoi/notebook/fe/validation_row_id.feather").head(len_train//10)
    if is_debug:
        df_val_row = df_val_row.head(3000)
    df_val_row["is_val"] = 1

    df = pd.merge(df, df_val_row, how="left", on="row_id")
    df["is_val"] = df["is_val"].fillna(0)

    print(df["is_val"].value_counts())

    if not load_pickle or is_debug:
        # 100件ずつgroupを作る. 例えば950件あったら、 1~50, 51~150, 151~250 のように、先頭が端数になるように
        w_df = df[df["is_val"] == 0]
        w_df["group"] = (w_df.groupby("user_id")["user_id"].transform("count") - w_df.groupby("user_id").cumcount()) // params["max_seq"]

        group = w_df.groupby(["user_id", "group"]).agg(agg_dict).T.to_dict()

        dataset_train = SAKTDataset(group,
                                    n_skill=60000,
                                    max_seq=params["max_seq"])

        del w_df
        gc.collect()

        group = df[df["content_type_id"] == 0].groupby("user_id").agg(agg_dict).T.to_dict()
        dataset_val = SAKTDataset(group,
                                  is_test=True,
                                  n_skill=60000,
                                  max_seq=params["max_seq"])

    os.makedirs("../input/feature_engineering/model200", exist_ok=True)
    if not is_debug and not load_pickle:
        with open(f"../input/feature_engineering/model200/train.pickle", "wb") as f:
            pickle.dump(dataset_train, f)
        with open(f"../input/feature_engineering/model200/val.pickle", "wb") as f:
            pickle.dump(dataset_val, f)

    if not is_debug and load_pickle:
        with open(f"../input/feature_engineering/model200/train.pickle", "rb") as f:
            dataset_train = pickle.load(f)
        with open(f"../input/feature_engineering/model200/val.pickle", "rb") as f:
            dataset_val = pickle.load(f)
        print("loaded!")
    dataloader_train = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=params["batch_size"], shuffle=False, num_workers=1)

    model = SAKTModel(n_skill=60000, embed_dim=params["embed_dim"], max_seq=params["max_seq"], dropout=dropout,
                      cont_emb=params["cont_emb"])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdaBelief(optimizer_grouped_parameters,
                          lr=params["lr"],
                          )
    num_train_optimization_steps = int(len(dataloader_train) * 20)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=params["num_warmup_steps"],
                                                num_training_steps=num_train_optimization_steps)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        loss, acc, auc, auc_val = train_epoch(model, dataloader_train, dataloader_val, optimizer, criterion, scheduler,
                                              epoch, device)
        print("epoch - {} train_loss - {:.3f} auc - {:.4f} auc-val: {:.4f}".format(epoch, loss, auc, auc_val))
        torch.save(model.state_dict(), f"{output_dir}/transformers_epoch{epoch}_auc{round(auc_val, 4)}.pth")

    preds = []
    labels = []
    with torch.no_grad():
        for item in tqdm(dataloader_val):
            label = item["label"].to(device).float()

            output = model(item, device)

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

if __name__ == "__main__":
    if not is_debug:
        for _ in tqdm(range(wait_time)):
            time.sleep(1)
    output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
    os.makedirs(output_dir, exist_ok=True)
    for lr in [0.9e-3]:
        cont_emb = 16
        cat_emb = 256
        dropout = 0.3
        if is_debug:
            batch_size = 8
        else:
            batch_size = 512
        params = {"embed_dim": cat_emb,
                  "cont_emb": cont_emb,
                  "max_seq": 100,
                  "batch_size": batch_size,
                  "num_warmup_steps": 3000,
                  "lr": lr,
                  "dropout": dropout}
        main(params, output_dir=output_dir)