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
    PreviousAnswer2, \
    StudyTermEncoder2, \
    MeanAggregator, \
    ElapsedTimeMeanByContentIdEncoder, \
    DurationFeaturePostProcess
from experiment.common import get_logger
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Optional, Any

torch.manual_seed(0)
np.random.seed(0)
is_debug = False
is_make_feature_factory = False
load_pickle = False
epochs = 13
device = torch.device("cuda")

wait_time = 0

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

        # =============================
        # setting index
        # =============================

        idx_dict = {
            ("content_id", "content_type_id"): 0,
            "user_answer": 1,
            "part": 2,
            "prior_question_elapsed_time_bin300": 3,
            "duration_previous_content_bin300": 4,
            "answered_correctly": 5,
            "prior_question_had_explanation": 6,
            "rating_diff_content_user_id": 7,
            "task_container_id_bin300": 8,
            "previous_answer_index_question_id": 9,
            "previous_answer_question_id": 10,
            "timediff-elapsedtime_bin500": 11,
            "timedelta_log10": 12
        }
        num_sequence = len(idx_dict)

        item_ary = np.zeros((num_sequence, self.max_seq))
        item_ary[idx_dict["answered_correctly"], :] = -1
        data_length = len(self.samples[user_id][("content_id", "content_type_id")])

        if self.is_test:
            start = np.max([0, end - self.max_seq])
        else:
            start = 0
            end = data_length
        seq_length = end - start

        for item_name, idx in idx_dict.items():
            item_ary[idx, -seq_length:] = self.samples[user_id][item_name][start:end]

        def get_data(key, remove_now=False):
            if remove_now:
                return item_ary[idx_dict[key], :][:-1]
            else:
                return item_ary[idx_dict[key], :][1:]

        return {
            "x": get_data(key="answered_correctly", remove_now=True) + 2, # 1: lecture, 2: not correct, 3: correct
            "user_answer": get_data(key="user_answer", remove_now=True) + 1,
            "target_id": get_data(key=("content_id", "content_type_id")),
            "part": get_data(key="part"),
            "elapsed_time": get_data(key="prior_question_elapsed_time_bin300"),
            "duration_previous_content": get_data(key="duration_previous_content_bin300"),
            "label": get_data(key="answered_correctly"),
            "prior_q": get_data(key="prior_question_had_explanation"),
            "rate_diff": get_data(key="rating_diff_content_user_id"),
            "container_id": get_data(key="task_container_id_bin300"),
            "previous_answer_index_question_id": get_data(key="previous_answer_index_question_id"),
            "previous_answer_question_id": get_data(key="previous_answer_question_id"),
            "timediff-elapsedtime_bin500": get_data(key="timediff-elapsedtime_bin500"),
            "prior_content_id": get_data(key=("content_id", "content_type_id"), remove_now=True),
            "timedelta_log10": get_data(key="timedelta_log10") + 2
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

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class LITLayer(nn.Module):
    """
    https://arxiv.org/pdf/2012.14164.pdf
    """
    def __init__(self, input_dim, embed_dim, dropout, activation):
        super(LITLayer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.activation = activation

        self.linear1 = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, input_dim)

        self.norm_lstm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.norm_lstm(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.lit_layer = LITLayer(input_dim=d_model, embed_dim=dim_feedforward, dropout=dropout, activation=self.activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.ReLU
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.lit_layer(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ContEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, seq_len):
        super(ContEmbedding, self).__init__()
        self.embed_dim = embed_dim

        self.bn = nn.BatchNorm1d(seq_len-1)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=embed_dim // 2)
        self.ln2 = nn.LayerNorm(embed_dim // 2)

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
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim // 2)
        self.ln2 = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.ln1(x)
        x, _ = self.gru(x)
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


class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.padding = (kernel_size-1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=self.padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=self.padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        h = self.res(x)
        h = h[:, :, :-self.padding]
        return x + h


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, num_heads=8, dropout=0.2,
                 cont_emb=None):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim_cat = embed_dim
        embed_dim_small_cat = 32
        embed_dim_middle_cat = 32
        embed_dim_cat_all = embed_dim_small_cat*5 + embed_dim_middle_cat*5 + embed_dim
        embed_dim_all = embed_dim_cat_all + cont_emb

        self.embedding = nn.Embedding(4, embed_dim_small_cat)
        self.user_answer_embedding = nn.Embedding(6, self.embed_dim_cat)
        self.prior_question_had_explanation_embedding = nn.Embedding(4, embed_dim_small_cat)
        self.e_embedding = nn.Embedding(n_skill + 1, self.embed_dim_cat)
        self.part_embedding = nn.Embedding(8, self.embed_dim_cat)
        self.elapsed_time_embedding = nn.Embedding(302, embed_dim_middle_cat)
        self.duration_previous_content_embedding = nn.Embedding(302, embed_dim_middle_cat)
        self.container_embedding = nn.Embedding(302, embed_dim_middle_cat)
        self.prev_ans_idx_embedding = nn.Embedding(302, embed_dim_middle_cat)
        self.prev_ans_content_id_embedding = nn.Embedding(4, embed_dim_small_cat)
        self.timediff_elapsedtime_embedding = nn.Embedding(502, embed_dim_middle_cat)
        self.timedelta_log10_embedding = nn.Embedding(15, embed_dim_small_cat)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim_all, nhead=num_heads, dropout=dropout)
        self.conv1d_stack = Conv1dStack(in_dim=embed_dim_all, out_dim=embed_dim_all)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=4)
        self.gru = nn.GRU(input_size=embed_dim_all, hidden_size=embed_dim_all)

        self.continuous_embedding = ContEmbedding(input_dim=1, embed_dim=cont_emb, seq_len=max_seq)
        self.prior_content_embedding = nn.Sequential(
            nn.Linear(self.embed_dim_cat, embed_dim_small_cat),
            nn.LayerNorm(embed_dim_small_cat)
        )
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
        label = item["label"].to(device).float()
        elapsed_time = item["elapsed_time"].to(device).long()
        duration_previous_content = item["duration_previous_content"].to(device).long()
        prior_q = item["prior_q"].to(device).long()
        user_answer = item["user_answer"].to(device).long()
        rate_diff = item["rate_diff"].to(device).float()
        container_id = item["container_id"].to(device).long()
        prev_ans_idx = item["previous_answer_index_question_id"].to(device).long()
        prior_content_id_ans_correctly = item["previous_answer_question_id"].to(device).long()
        prior_content_id = item["prior_content_id"].to(device).long()
        timediff_elapsedtime = item["timediff-elapsedtime_bin500"].to(device).long()
        timedelta_log10 = item["timedelta_log10"].to(device).long()

        att_mask = future_mask(x.size(1)).to(device)

        e = self.e_embedding(question_ids)
        p = self.part_embedding(parts)

        e = e + p

        prior_q_emb = self.prior_question_had_explanation_embedding(prior_q)

        user_answer_emb = self.user_answer_embedding(user_answer)
        prior_content_id_emb = self.e_embedding(prior_content_id)
        prior_content_user_answer_emb = self.prior_content_embedding(user_answer_emb + prior_content_id_emb)
        timediff_elapsedtime_emb = self.timediff_elapsedtime_embedding(timediff_elapsedtime)
        timedelta_log10_emb = self.timedelta_log10_embedding(timedelta_log10)

        # decoder
        x = self.embedding(x)
        el_time_emb = self.elapsed_time_embedding(elapsed_time)
        dur_emb = self.duration_previous_content_embedding(duration_previous_content)
        container_emb = self.container_embedding(container_id)
        prev_ans_idx_emb = self.prev_ans_idx_embedding(prev_ans_idx)
        prev_ans_content_id_emb = self.prev_ans_content_id_embedding(prior_content_id_ans_correctly)
        x = torch.cat([x, el_time_emb, dur_emb, e, prior_q_emb, container_emb,
                       prev_ans_idx_emb, prev_ans_content_id_emb,
                       prior_content_user_answer_emb,
                       timediff_elapsedtime_emb, timedelta_log10_emb], dim=2)

        cont = rate_diff

        cont_emb = self.continuous_embedding(cont.view(x.size(0), x.size(1), -1))
        x = self.cat_embedding(x)
        x = torch.cat([x, cont_emb], dim=2)

        x = x.permute(0, 2, 1)  # x: [bs, s_len, embed] => [bs, embed, s_len]
        x = self.conv1d_stack(x)

        x = x.permute(2, 0, 1)  # x: [bs, embed, s_len] => [s_len, bs, embed]

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
            if i > 100 and epoch < 12:
                break
    auc_val = roc_auc_score(labels, preds)

    return loss, acc, auc, auc_val

def main(params: dict,
         output_dir: str):
    import mlflow
    print("start params={}".format(params))
    model_id = "train_0"
    logger = get_logger()
    # df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")
    df = pd.read_pickle("../input/riiid-test-answer-prediction/split10/train_0.pickle").sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    if is_debug:
        df = df.head(30000)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1)
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    column_config = {
        ("content_id", "content_type_id"): {"type": "category"},
        "user_answer": {"type": "leakage_feature"},
        "answered_correctly": {"type": "leakage_feature"},
        "part": {"type": "category"},
        "prior_question_elapsed_time_bin300": {"type": "category"},
        "duration_previous_content_bin300": {"type": "category"},
        "prior_question_had_explanation": {"type": "category"},
        "rating_diff_content_user_id": {"type": "numeric"},
        "task_container_id_bin300": {"type": "category"},
        "previous_answer_index_question_id": {"type": "category"},
        "previous_answer_question_id": {"type": "category"},
        "timediff-elapsedtime_bin500": {"type": "category"},
        "timedelta_log10": {"type": "category"}
    }


    if not load_pickle or is_debug:
        feature_factory_dict = {"user_id": {}}
        feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
        feature_factory_dict["user_id"]["ElapsedTimeBinningEncoder"] = ElapsedTimeBinningEncoder()
        feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(rate_func="elo",
                                                                                           column="user_id")
        feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                             column="question_id",
                                                                             is_debug=is_debug,
                                                                             model_id=model_id,
                                                                             n=300)
        feature_factory_dict["user_id"]["StudyTermEncoder2"] = StudyTermEncoder2(is_partial_fit=True)
        feature_factory_dict["user_id"][f"MeanAggregatorStudyTimebyUserId"] = MeanAggregator(column="user_id",
                                                                                             agg_column="study_time",
                                                                                             remove_now=False)

        feature_factory_dict["user_id"]["ElapsedTimeMeanByContentIdEncoder"] = ElapsedTimeMeanByContentIdEncoder()
        feature_factory_dict["post"] = {
            "DurationFeaturePostProcess": DurationFeaturePostProcess()
        }

        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger,
                                                        split_num=1,
                                                        model_id=model_id,
                                                        load_feature=not is_debug,
                                                        save_feature=not is_debug)
        print("all_predict")
        df = feature_factory_manager.all_predict(df)
        def f(x):
            x = x // 1000
            if x < -100:
                return -100
            if x > 400:
                return 400
            return x
        df["task_container_id_bin300"] = [x if x < 300 else 300 for x in df["task_container_id"]]
        df["timediff-elapsedtime_bin500"] = [f(x) for x in df["timediff-elapsedtime"].values]
        df["timedelta_log10"] = np.log10(df["duration_previous_content"].values)
        df["timedelta_log10"] = df["timedelta_log10"].replace(-np.inf, -1).replace(np.inf, -1).fillna(-1).astype("int8")
        df = df[["user_id", "content_id", "content_type_id", "part", "user_answer", "answered_correctly",
                 "prior_question_elapsed_time_bin300", "duration_previous_content_bin300",
                 "prior_question_had_explanation", "rating_diff_content_user_id", "task_container_id_bin300",
                 "previous_answer_index_question_id", "previous_answer_question_id", "row_id",
                 "timediff-elapsedtime_bin500", "timedelta_log10"]]
        print(df.head(10))

        print("data preprocess")



    ff_for_transformer = FeatureFactoryForTransformer(column_config=column_config,
                                                      dict_path="../feature_engineering/",
                                                      sequence_length=params["max_seq"],
                                                      logger=logger)
    ff_for_transformer.make_dict(df=df)
    n_skill = len(ff_for_transformer.embbed_dict[("content_id", "content_type_id")])

    if not load_pickle or is_debug:
        df_val_row = pd.read_feather("../../riiid_takoi/notebook/fe/validation_row_id.feather").head(len(df))
        if is_debug:
            df_val_row = df_val_row.head(3000)
        df_val_row["is_val"] = 1

        df = pd.merge(df, df_val_row, how="left", on="row_id")
        df["is_val"] = df["is_val"].fillna(0)

        print(df["is_val"].value_counts())

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

    os.makedirs("../input/feature_engineering/model284", exist_ok=True)
    if not is_debug and not load_pickle:
        with open(f"../input/feature_engineering/model284/train.pickle", "wb") as f:
            pickle.dump(dataset_train, f)
        with open(f"../input/feature_engineering/model284/val.pickle", "wb") as f:
            pickle.dump(dataset_val, f)

    if not is_debug and load_pickle:
        with open(f"../input/feature_engineering/model284/train.pickle", "rb") as f:
            dataset_train = pickle.load(f)
        with open(f"../input/feature_engineering/model284/val.pickle", "rb") as f:
            dataset_val = pickle.load(f)
        print("loaded!")
    dataloader_train = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=params["batch_size"], shuffle=False)

    model = SAKTModel(n_skill, embed_dim=params["embed_dim"], max_seq=params["max_seq"], dropout=dropout,
                      cont_emb=params["cont_emb"])

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
    if is_make_feature_factory:
        # feature factory
        feature_factory_dict = {"user_id": {}}
        feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
        feature_factory_dict["user_id"]["ElapsedTimeBinningEncoder"] = ElapsedTimeBinningEncoder()
        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger,
                                                        split_num=1,
                                                        model_id="all",
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
    for cont_emb in [8]:
        for cat_emb in [256]:
            dropout = 0.5
            lr = 1e-3
            if is_debug:
                batch_size = 8
            else:
                batch_size = 128
            params = {"embed_dim": cat_emb,
                      "cont_emb": cont_emb,
                      "max_seq": 100,
                      "batch_size": batch_size,
                      "num_warmup_steps": 3000,
                      "lr": lr,
                      "dropout": dropout}
            main(params, output_dir=output_dir)