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
from experiment.common import get_logger

torch.manual_seed(0)
np.random.seed(0)
is_debug = True
is_make_feature_factory = True
epochs = 25
device = torch.device("cuda")

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)


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
                        self.user_ids.append([user_id, i + 1])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index][0]
        end = self.user_ids[index][1]
        q_ = self.samples[user_id][("content_id", "content_type_id")]
        part_ = self.samples[user_id]["part"]
        qa_ = self.samples[user_id]["answered_correctly"]

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
        target_idx = (label.view(-1) >= 0).nonzero()
        loss = criterion(output.view(-1)[target_idx], label.view(-1)[target_idx])
        loss.backward()
        optim.step()
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
    for d in tqdm(val_iterator):
        x = d[0].to(device).long()
        target_id = d[1].to(device).long()
        part = d[2].to(device).long()
        label = d[3].to(device).long()

        output, atten_weight = model(x, target_id, part)
        target_idx = (label[:, -1] >= 0).nonzero()
        preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1)[target_idx].data.cpu().numpy().tolist())
        labels.extend(label[:, -1].view(-1)[target_idx].data.cpu().numpy())
        i += 1
        if i > 100:
            break
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val


def main(params: dict):
    import mlflow
    logger = get_logger()
    print("start params={}".format(params))
    # df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle").head(30_000_000)
    df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(
        ["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(3000)
        df.to_csv("aaa.csv")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df = df[["user_id", "content_id", "content_type_id", "part", "answered_correctly"]]

    train_idx = []
    val_idx = []
    np.random.seed(0)
#    for _, w_df in df[df["content_type_id"] == 0].groupby("user_id"):
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

    ff_for_transformer = FeatureFactoryForTransformer(
        column_config={("content_id", "content_type_id"): {"type": "category"},
                       "part": {"type": "category"}},
        dict_path="../feature_engineering/",
        sequence_length=params["max_seq"],
        logger=logger)
    n_skill = len(ff_for_transformer.embbed_dict[("content_id", "content_type_id")])

    group = ff_for_transformer.all_predict(df)
    dataset_val = SAKTDataset(group,
                              is_test=True,
                              n_skill=n_skill,
                              max_seq=params["max_seq"])

    dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=1)

    ## all_predict
    xs = []
    target_ids = []
    parts = []
    labels = []
    for d in tqdm(dataloader_val):
        xs.extend(d[0].to(device).long().data.cpu().numpy().tolist())
        target_ids.extend(d[1].to(device).long().data.cpu().numpy().tolist())
        parts.extend(d[2].to(device).long().data.cpu().numpy().tolist())
        labels.extend(d[3].to(device).long().data.cpu().numpy().tolist())
    print(xs[0])
    print(target_ids[0])
    print(parts[0])
    print(labels[0])

    ## partial_predict
    ff_for_transformer = FeatureFactoryForTransformer(
        column_config={("content_id", "content_type_id"): {"type": "category"},
                       "part": {"type": "category"}},
        dict_path="../feature_engineering/",
        sequence_length=params["max_seq"],
        logger=logger)

    ff_for_transformer.fit(df.loc[train_idx])

    xs_test = []
    target_ids_test = []
    parts_test = []
    labels_test = []
    for i in tqdm(range(len(df.loc[val_idx]))):
        w_df = df.loc[val_idx[i:i+1]]
        group = ff_for_transformer.partial_predict(w_df.copy())

        dataset_val = SAKTDataset(group,
                                  predict_mode=True,
                                  n_skill=n_skill,
                                  max_seq=params["max_seq"])

        dataloader_val = DataLoader(dataset_val, batch_size=1024, shuffle=False, num_workers=1)
        for d in dataloader_val:
            xs_test.extend(d[0].to(device).long().data.cpu().numpy().tolist())
            target_ids_test.extend(d[1].to(device).long().data.cpu().numpy().tolist())
            parts_test.extend(d[2].to(device).long().data.cpu().numpy().tolist())
            labels_test.extend(d[3].to(device).long().data.cpu().numpy().tolist())
        ff_for_transformer.fit(w_df.copy())

    pd.DataFrame(xs).to_csv("all_xs.csv")
    pd.DataFrame(xs_test).to_csv("partial_xs.csv")
    pd.DataFrame(labels).to_csv("all_labels.csv")
    pd.DataFrame(labels_test).to_csv("partial_labels.csv")
    df.loc[val_idx].to_csv("raw_data.csv")
    assert xs[0] == xs_test[0]
    assert xs == xs_test
    assert target_ids == target_ids_test
    assert parts == parts_test

if __name__ == "__main__":
    params = {"embed_dim": 256,
              "max_seq": 10,
              "lr": 1e-3}
    main(params)