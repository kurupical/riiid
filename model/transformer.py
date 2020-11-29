import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import tqdm
import copy
from sklearn.metrics import roc_auc_score
from logging import Logger

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class TransformerModel(nn.Module):

    def __init__(self,
                 use_cols_config: Dict[str, dict],
                 n_emb: int = 32,
                 n_head: int = 4,
                 n_hidden: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=n_hidden, dropout=dropout,
                                                 activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=n_layers)

        self.embeddings_dict = {}
        for column, config_dict in use_cols_config.items():
            self.embeddings_dict[column] = nn.Embedding(num_embeddings=config_dict["num_embeddings"], embedding_dim=n_emb)
        self.device = "gpu"
        self.n_emb = n_emb
        self.decoder = nn.Linear(n_emb, 1)

    def forward(self,
                data: Dict[str, dict]):
        '''
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        '''

        embedded_src = None

        for col, embeedding_layer in self.embeddings_dict.items():
            if embedded_src is None:
                embedded_src = embeedding_layer(data[col])
            else:
                embedded_src += embeedding_layer(data[col])

        _src = embedded_src * np.sqrt(self.n_emb)
        print(embedded_src.shape)
        # TODO: ここにpositional_encoderをいれる

        output = self.transformer_encoder(src=_src,
                                          #src_key_padding_mask=data["mask"].view(-1, 50, 1)
                                          ).transpose(1, 2)
        output = torch.nn.AdaptiveAvgPool1d(output_size=1)(output).view(-1, self.n_emb)

        output = self.decoder(output)
        return output

def pad_seq(seq: List[int],
            max_batch_len: int = 50,
            pad_value: int = 0) -> List[int]:
    if len(seq) >= max_batch_len:
        return seq
    else:
        return seq + (max_batch_len - len(seq)) * [pad_value]

class RiiidDataset(torch.utils.data.Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 indice: List[int],
                 use_cols_config: Dict[str, dict],
                 window: int):

        self.index_dict = {}  # {index: {user_id, index_start}}
        self.data_dict = {} # user_id: {use_col[0], use_col[1], ...}
        self.use_cols_config = use_cols_config
        self.window = window
        self.indice = indice

        current_idx = 0
        for user_id, w_df in tqdm.tqdm(df.groupby("user_id")):

            self.data_dict[user_id] = {}
            w_df["answered_correctly"] = w_df["answered_correctly"].replace(np.nan, 2)
            # 辞書作成
            for col in use_cols_config.keys():
                self.data_dict[user_id][col] = pad_seq(w_df[col].values.tolist())

            idxs = w_df.index
            indice = set(self.indice)
            content_ids = w_df["content_type_id"].values
            for i in range(len(w_df)):
                if content_ids[i] == 1:
                    continue
                if idxs[i] not in indice:
                    continue
                if i < self.window:
                    self.index_dict[current_idx] = {"user_id": user_id,
                                                    "index_start": 0,
                                                    "index_end": i+1}
                else:
                    self.index_dict[current_idx] = {"user_id": user_id,
                                                    "index_start": i-self.window+1,
                                                    "index_end": i+1}
                current_idx += 1


    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        user_id = self.index_dict[idx]["user_id"]
        start_idx = self.index_dict[idx]["index_start"]
        end_idx = self.index_dict[idx]["index_end"]
        data_dict = copy.copy(self.data_dict[user_id])

        # データ作成
        for k, v in data_dict.items():
            data_dict[k] = pad_seq(data_dict[k][start_idx:end_idx][::-1], max_batch_len=self.window)

        # mask作成
        if end_idx - start_idx == self.window:
            data_dict["mask"] = [False] * self.window
        else:
            not_mask_length = end_idx - start_idx
            mask_length = self.window - not_mask_length
            data_dict["mask"] = [False] * not_mask_length + [True] * mask_length

        return data_dict

def collate_fn(batch):

    ret_dict = {}

    for data_dict in batch:
        for col, values in data_dict.items():
            if col not in ret_dict:
                ret_dict[col] = [values]
            else:
                ret_dict[col] += [values]

    for col, values in ret_dict.items():
        if col == "mask":
            ret_dict[col] = torch.Tensor(values).bool()
        elif col == "answered_correctly":
            ret_dict[col] = torch.Tensor(values).long()
        else:
            ret_dict[col] = torch.Tensor(values).long()
    return ret_dict

def train_transformer(df: pd.DataFrame,
                      use_cols_config: Dict[str, dict],
                      window: int,
                      criterion: Union,
                      optimizer: Union,
                      optimizer_params: dict,
                      scheduler: Union,
                      scheduler_params: dict,
                      n_emb: int,
                      n_head: int,
                      n_hidden: int,
                      n_layers: int,
                      batch_size: int,
                      epochs: int,
                      dropout: float,
                      logger: Logger,
                      output_dir: str,
                      model_id: str):
    """

    :param df:
    :param use_cols:
        {col_name: {"embedding_num": int}}
    :param n_emb:
    :param n_head:
    :param n_hidden:
    :param n_layers:
    :param batch_size:
    :param dropout:
    :return:
    """

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

    dataset_train = RiiidDataset(df=df,
                                 indice=train_idx,
                                 use_cols_config=use_cols_config,
                                 window=window)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                   batch_size=batch_size,
                                                   collate_fn=collate_fn,
                                                   num_workers=4,
                                                   shuffle=True)
    print(f"make_train_data len={len(dataset_train)}")
    dataset_val = RiiidDataset(df=df,
                               indice=val_idx,
                               use_cols_config=use_cols_config,
                               window=window)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                 batch_size=batch_size,
                                                 collate_fn=collate_fn,
                                                 num_workers=1,
                                                 shuffle=False)
    print(f"make_val_data len={len(dataset_val)}")

    model = TransformerModel(n_emb=n_emb,
                             use_cols_config=use_cols_config,
                             n_head=n_head,
                             n_hidden=n_hidden,
                             n_layers=n_layers,
                             dropout=dropout)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # TODO: 後で直す。汚い。。
    scheduler_params["num_training_steps"] = len(dataset_train) // batch_size * scheduler_params["num_training_epochs"]
    del scheduler_params["num_training_epochs"]

    optimizer = optimizer(optimizer_grouped_parameters, **optimizer_params)
    scheduler = scheduler(optimizer, **scheduler_params)
    model.train()
    losses = []
    predict = []
    label = []
    for epoch in range(epochs):
        logger.info(f"--- epoch {epoch+1} ---")

        for batch in tqdm.tqdm(dataloader_train):
            with torch.set_grad_enabled(mode=True):
                output = model(batch)
                loss = criterion(output.flatten().float(),
                                 batch["answered_correctly"][:, -1].flatten().float())
                loss.backward()
                losses.append(loss.detach().data.numpy())
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

        predict = []
        label = []
        for batch in tqdm.tqdm(dataloader_val):
            output = nn.functional.sigmoid(model(batch))
            predict.extend(output.flatten().detach().data.numpy().tolist())
            label.extend(batch["answered_correctly"][:, 0].flatten().detach().data.numpy().tolist())
        logger.info(f"AUC: {round(roc_auc_score(np.array(label), np.array(predict)), 4)}")

    df.loc[val_idx].to_csv(f"{output_dir}/val.csv")
    df.to_csv(f"{output_dir}/all.csv")
    df_ret = pd.DataFrame(index=val_idx)
    df_ret["predict"] = np.array(predict)
    df_ret["target"] = np.array(label)
    df_ret["target2"] = df.loc[val_idx]["answered_correctly"]
    df_ret.to_csv(f"{output_dir}/oof_{model_id}.csv")