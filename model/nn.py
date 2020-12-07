import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np
import tempfile

import pickle
import mlflow
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization, Input, Add, PReLU
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization, Input, Add, PReLU
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tfdeterminism import patch
from sklearn.metrics import roc_auc_score

def train_nn_cv(df: pd.DataFrame,
                model,
                params: dict,
                output_dir: str,
                model_id: int,
                exp_name: str,
                drop_user_id: bool,
                experiment_id: int=0,
                is_debug: bool=False):

    if not is_debug:
        mlflow.start_run(experiment_id=experiment_id, run_name=exp_name)

        mlflow.log_param("model_id", model_id)
        mlflow.log_param("count_row", len(df))
        mlflow.log_param("count_column", len(df.columns))

        for key, value in params.items():
            mlflow.log_param(key, value)

    if drop_user_id:
        features = [x for x in df.columns if x not in ["answered_correctly", "user_id"]]
    else:
        features = [x for x in df.columns if x not in ["answered_correctly"]]
    df_imp = pd.DataFrame()
    df_imp["feature"] = features

    train_idx = []
    val_idx = []
    np.random.seed(0)
    np.random.seed(0)
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    if is_debug:
        epochs = 3
    else:
        epochs = 1000

    model.fit(df[features].iloc[train_idx].values, df["answered_correctly"].iloc[train_idx].values.reshape(-1, 1),
              batch_size=2**17,
              epochs=epochs,
              verbose=True,
              validation_data=(df[features].iloc[val_idx].values,
                               df["answered_correctly"].iloc[val_idx].values.reshape(-1, 1)),
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=-1, mode='auto'),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=-1,
                                           mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
                         ModelCheckpoint(filepath=f"{output_dir}/best_nn_{model_id}.weight", monitor='val_loss', verbose=-1,
                                         save_best_only=True, mode='auto')])
    model = load_model(f"{output_dir}/best_nn_{model_id}.weight")
    pd.DataFrame(features, columns=["feature"]).to_csv(f"{output_dir}/nn_use_feature.csv", index=False)

    y_train = model.predict(df.iloc[train_idx][features])
    y_oof = model.predict(df.iloc[val_idx][features])

    auc_train = roc_auc_score(df.iloc[train_idx]["answered_correctly"].values.flatten(), y_train.flatten())
    auc_val = roc_auc_score(df.iloc[val_idx]["answered_correctly"].values.flatten(), y_oof.flatten())
    print(f"auc_train: {auc_train}, auc_val: {auc_val}")
    if not is_debug:
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_val", auc_val)
        mlflow.end_run()

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.iloc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.iloc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_nn.csv", index=False)

