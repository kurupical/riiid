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
import mlflow

torch.manual_seed(0)
np.random.seed(0)
is_debug = False
max_lens = [100]
emb_nums = [128, 256]

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, n_part=8, max_seq=100, is_test=False):
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.is_test = is_test
        self.n_part = n_part

        self.user_ids = []
        for user_id in group.index:
            q, part, qa, is_val = group[user_id]
            if not is_test:
                self.user_ids.append([user_id, -1])
            else:
                for i in range(len(q)):
                    if is_val[i]:
                        self.user_ids.append([user_id, i+1])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index][0]
        end = self.user_ids[index][1]
        q_, part_, qa_, _ = self.samples[user_id]

        if not self.is_test:
            seq_len = len(q_)
        else:
            start = np.max([0, end - self.max_seq])
            q_ = q_[start:end]
            part_ = part_[start:end]
            qa_ = qa_[start:end]
            seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            part[:] = part_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            part[-seq_len:] = part_
            qa[-seq_len:] = qa_

        target_id = q[1:]
        part = part[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq - 1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, part, label


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)
        self.part_embedding = nn.Embedding(8, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids, parts):
        device = x.device
        x = self.embedding(x)

        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)
        p = self.part_embedding(parts)
        e = e + p

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight


def train_epoch(model, train_iterator, val_iterator, optim, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        part = item[2].to(device).long()
        label = item[3].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(x, target_id, part)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1]
        pred = (torch.sigmoid(output) >= 0.5).long()

        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    preds = []
    labels = []
    model.eval()
    i = 0
    for d in tqdm(val_iterator):
        x = d[0].to(device).long()
        target_id = d[1].to(device).long()
        part = d[2].to(device).long()
        label = d[3].to(device).long()

        output, atten_weight = model(x, target_id, part)

        preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
        labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        i += 1
        if i > 100:
            break
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val

def main(params: dict):
    print("start params={}".format(params))
    df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(10000)
    df = df[df.content_type_id == False]

    train_idx = []
    val_idx = []
    np.random.seed(0)
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    df["is_val"] = 0
    df["is_val"].loc[val_idx] = 1
    df["group"] = (df.groupby("user_id")["user_id"].transform("count") - df.groupby("user_id").cumcount()) // max_len
    print(df["group"].value_counts())

    print(len(train_idx))
    print(len(val_idx))
    group = df[df["is_val"] == 0][['user_id', 'content_id', 'answered_correctly', "group", "part", 'is_val']].groupby(['user_id', "group"]).apply(lambda r: (
                r['content_id'].values,
                r['part'].values,
                r['answered_correctly'].values,
                r["is_val"].values))
    dataset_train = SAKTDataset(group, 13523, max_seq=params["max_seq"])

    group = df[['user_id', 'content_id', 'answered_correctly', "part", 'is_val']].groupby(['user_id']).apply(lambda r: (
                r['content_id'].values,
                r['part'].values,
                r['answered_correctly'].values,
                r["is_val"].values))
    dataset_val = SAKTDataset(group, 13523, is_test=True, max_seq=params["max_seq"])

    dataloader_train = DataLoader(dataset_train, batch_size=1024, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=1)

    device = torch.device("cuda")

    model = SAKTModel(13523, embed_dim=params["embed_dim"], max_seq=params["max_seq"])
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    epochs = 12
    for epoch in range(epochs):
        loss, acc, auc, auc_val = train_epoch(model, dataloader_train, dataloader_val, optimizer, criterion, device)
        print("epoch - {} train_loss - {:.3f} auc - {:.4f} auc-val: {:.4f}".format(epoch, loss, auc, auc_val))

    preds = []
    labels = []
    for d in tqdm(dataloader_val):
        x = d[0].to(device).long()
        target_id = d[1].to(device).long()
        part = d[2].to(device).long()
        label = d[3].to(device).long()

        output, atten_weight = model(x, target_id, part)

        preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
        labels.extend(label[:, -1].view(-1).data.cpu().numpy())
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = preds
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values
    df_oof.to_csv(f"{output_dir}/transformers1.csv", index=False)

    df_oof2 = pd.read_csv("../output/ex_172/20201202080625/oof_train_0_lgbm.csv")
    df_oof2.columns = ["row_id", "predict_lgbm", "target"]
    df_oof2 = pd.merge(df_oof, df_oof2, how="inner")

    auc_transformer = roc_auc_score(df_oof2["target"].values, df_oof2["predict"].values)
    auc_lgbm = roc_auc_score(df_oof2["target"].values, df_oof2["predict_lgbm"].values)
    print("single transformer: {:.4f}".format(auc_transformer))
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

    if not is_debug:
        mlflow.start_run(experiment_id=10,
                         run_name=os.path.basename(__file__))

        mlflow.log_param("count_row", len(df))

        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("auc_val", auc_transformer)
        mlflow.log_metric("auc_lgbm", auc_lgbm)
        mlflow.log_metric("auc_ensemble", max_auc)
        mlflow.log_metric("ensemble_nn_ratio", max_nn_ratio)
        mlflow.end_run()

if __name__ == "__main__":
    for emb_num in emb_nums:
        for max_len in max_lens:
            params = {"embed_dim": emb_num,
                      "max_seq": max_len}
            main(params)