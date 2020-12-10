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
is_full_data = False
epochs = 1
device = torch.device("cpu")

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
                        self.user_ids.append([user_id, i+1])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index][0]
        end = self.user_ids[index][1]
        q_ = self.samples[user_id][("content_id", "content_type_id")]
        ua_ = self.samples[user_id]["user_answer"]
        part_ = self.samples[user_id]["part"]
        qa_ = self.samples[user_id]["answered_correctly"]

        if not self.is_test:
            seq_len = len(q_)
        else:
            start = np.max([0, end - self.max_seq])
            q_ = q_[start:end]
            part_ = part_[start:end]
            qa_ = qa_[start:end]
            ua_ = ua_[start:end]
            seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        part = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        ua = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            part[:] = part_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
            ua[:] = ua_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            part[-seq_len:] = part_
            qa[-seq_len:] = qa_
            ua[-seq_len:] = ua_

        target_id = q[1:]
        part = part[1:]
        label = qa[1:]

        x = q[:-1].copy()
        past_label = qa[:-1].copy() + 1
        x[x < 0] = 0

        return x, past_label, target_id, part, label


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

        self.content_embedding = nn.Embedding(n_skill + 1, embed_dim)
        self.qa_embedding = nn.Embedding(4, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill + 1, embed_dim)
        self.part_embedding = nn.Embedding(8, embed_dim)

        self.self_att_enc = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)
        self.ffn_enc = FFN(embed_dim)
        self.self_att_dec = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)
        self.multi_att_dec = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)
        self.ffn_dec_self1 = FFN(embed_dim)
        self.ffn_dec = FFN(embed_dim)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, content, qa, question_ids, parts):
        device = content.device
        att_mask = future_mask(content.size(1)).to(device)

        content_emb = self.content_embedding(content)
        qa_emb = self.qa_embedding(qa)
        pos_id = torch.arange(content_emb.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = content_emb + pos_x + qa_emb

        e = self.e_embedding(question_ids)
        p = self.part_embedding(parts)
        e = e + p
        e = e.permute(1, 0, 2)

        att_enc, _ = self.self_att_enc(e, e, e, attn_mask=att_mask)
        att_enc = att_enc.permute(1, 0, 2)
        att_enc = self.ffn_enc(att_enc)
        att_enc = self.layer_normal(att_enc + e.permute(1, 0, 2))

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        att_enc = att_enc.permute(1, 0, 2)

        att_self_dec, _ = self.self_att_dec(x, x, x, attn_mask=att_mask)
        att_self_dec = att_self_dec.permute(1, 0, 2)
        att_self_dec = self.ffn_enc(att_self_dec)
        att_self_dec = self.layer_normal(att_self_dec + x.permute(1, 0, 2))

        att_self_dec = att_self_dec.permute(1, 0, 2)
        att_output, att_weight = self.multi_att_dec(att_enc, att_enc, att_self_dec, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + att_self_dec)
        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn_dec(att_output)
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
        content = item[0].to(device).long()
        qa = item[1].to(device).long()
        target_id = item[2].to(device).long()
        part = item[3].to(device).long()
        label = item[4].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(content, qa, target_id, part)
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
        qa = d[1].to(device).long()
        target_id = d[2].to(device).long()
        part = d[3].to(device).long()
        label = d[4].to(device).long()

        output, atten_weight = model(x, qa, target_id, part)
        preds.extend(torch.nn.Sigmoid()(output[:, -1]).view(-1).data.cpu().numpy().tolist())
        labels.extend(label[:, -1].view(-1).data.cpu().numpy())
        i += 1
        if i > 100:
            break
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val

def main(params: dict):
    import mlflow
    logger = get_logger()
    print("start params={}".format(params))
    if is_full_data:
        df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
    else:
        df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    # df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(30000)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    df = df[["user_id", "content_id", "content_type_id", "part", "user_answer", "answered_correctly"]]

    train_idx = []
    val_idx = []
    np.random.seed(0)
    for _, w_df in df[df["content_type_id"] == 0].groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    df["is_val"] = 0
    df["is_val"].loc[val_idx] = 1
    w_df = df[df["is_val"] == 0]
    w_df["group"] = (w_df.groupby("user_id")["user_id"].transform("count") - w_df.groupby("user_id").cumcount()) // params["max_seq"]
    w_df["user_id"] = w_df["user_id"].astype(str) + "_" + w_df["group"].astype(str)
    ff_for_transformer = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category"},
                                                                     "user_answer": {"type": "category"},
                                                                     "part": {"type": "category"}},
                                                      dict_path="../feature_engineering/",
                                                      sequence_length=params["max_seq"],
                                                      logger=logger)
    group = ff_for_transformer.all_predict(w_df)
    del w_df
    gc.collect()

    n_skill = len(ff_for_transformer.embbed_dict[("content_id", "content_type_id")])
    print(group)
    dataset_train = SAKTDataset(group,
                                n_skill=n_skill,
                                max_seq=params["max_seq"])

    ff_for_transformer = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category"},
                                                                     "user_answer": {"type": "category"},
                                                                     "part": {"type": "category"}},
                                                      dict_path="../feature_engineering/",
                                                      sequence_length=params["max_seq"],
                                                      logger=logger)
    df["group"] = (df.groupby("user_id")["user_id"].transform("count") - df.groupby("user_id").cumcount()) // params["max_seq"]
    group = ff_for_transformer.all_predict(df)
    dataset_val = SAKTDataset(group,
                              is_test=True,
                              n_skill=n_skill,
                              max_seq=params["max_seq"])

    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=1)

    model = SAKTModel(n_skill, embed_dim=params["embed_dim"], max_seq=params["max_seq"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        loss, acc, auc, auc_val = train_epoch(model, dataloader_train, dataloader_val, optimizer, criterion, device)
        print("epoch - {} train_loss - {:.3f} auc - {:.4f} auc-val: {:.4f}".format(epoch, loss, auc, auc_val))

    preds = []
    labels = []
    for d in tqdm(dataloader_val):
        x = d[0].to(device).long()
        qa = d[1].to(device).long()
        target_id = d[2].to(device).long()
        part = d[3].to(device).long()
        label = d[4].to(device).long()

        output, atten_weight = model(x, qa, target_id, part)

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
    torch.save(model.state_dict(), f"{output_dir}/transformers.pth")

    del df, dataset_train, dataset_val, dataloader_train, dataloader_val
    gc.collect()

    with open(f"{output_dir}/transformer_param.json", "w") as f:
        json.dump(params, f)
    if is_make_feature_factory:
        ff_for_transformer = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category"},
                                                                          "user_answer": {"type": "category"},
                                                                          "part": {"type": "category"}},
                                                          dict_path="../feature_engineering/",
                                                          sequence_length=params["max_seq"],
                                                          logger=logger)
        df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
        if is_debug:
            df = df.head(10000)
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        ff_for_transformer.fit(df)
        ff_for_transformer.logger = None
        with open(f"{output_dir}/feature_factory_manager_for_transformer.pickle", "wb") as f:
            pickle.dump(ff_for_transformer, f)


if __name__ == "__main__":
    for embed_dim in [256]:
        for max_seq in [100]:
            params = {"embed_dim": embed_dim,
                      "max_seq": max_seq,
                      "lr": 1e-3}
            main(params)