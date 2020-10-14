import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np
import tempfile

import pickle
import mlflow
import random
import os

def train_lgbm_kfold(df: pd.DataFrame,
                     fold: BaseCrossValidator,
                     params: dict,
                     output_dir: str):

    y_oof = np.zeros(len(df))

    features = [x for x in df.columns if x != "answered_correctly"]

    df_imp = pd.DataFrame()
    df_imp["feature"] = features
    for i, (train_idx, val_idx) in enumerate(fold.split(df, df["answered_correctly"])):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        train_data = lgb.Dataset(df_train[features],
                                 label=df_train["answered_correctly"])
        valid_data = lgb.Dataset(df_train[features],
                                 label=df_train["answered_correctly"])

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            verbose_eval=100
        )
        y_oof[val_idx] = model.predict(df_val[features])

        df_imp[f"fold{i}"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
        with open(f"{output_dir}/model_fold{i}.pickle", "wb") as f:
            pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["predict"] = y_oof
    df_oof["target"] = df["answered_correctly"]

    df_oof.to_csv(f"{output_dir}/oof.csv", index=False)

    # feature importance
    df_imp["fold_mean"] = df_imp.drop("feature", axis=1).mean(axis=1)
    df_imp.sort_values("fold_mean", ascending=False).to_csv(f"{output_dir}/imp.csv")


def train_lgbm_cv(df: pd.DataFrame,
                  params: dict,
                  output_dir: str,
                  model_id: int,
                  exp_name: str):

    mlflow.start_run()

    mlflow.log_param("exp_name", exp_name)
    mlflow.log_param("model_id", model_id)
    mlflow.log_param("count_row", len(df))
    mlflow.log_param("count_column", len(df.columns))

    for key, value in params.items():
        mlflow.log_param(key, value)
    features = [x for x in df.columns if x != "answered_correctly"]

    df_imp = pd.DataFrame()
    df_imp["feature"] = features

    train_idx = []
    val_idx = []
    np.random.seed(0)
    for _, w_df in df.groupby("user_id"):
        train_num = (np.random.random(len(w_df)) < 0.8).sum()
        train_idx.extend(w_df[:train_num].index.tolist())
        val_idx.extend(w_df[train_num:].index.tolist())

    df_train, df_val = df.loc[train_idx], df.loc[val_idx]
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    valid_data = lgb.Dataset(df_val[features],
                             label=df_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
    mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
    y_oof = model.predict(df_val[features])

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}.pickle", "wb") as f:
        pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df_val.index
    df_oof["predict"] = y_oof
    df_oof["target"] = df_val["answered_correctly"]

    df_oof.to_csv(f"{output_dir}/oof_{model_id}.csv", index=False)

    mlflow.end_run()


def train_lgbm_cv_newuser(df: pd.DataFrame,
                          params: dict,
                          output_dir: str,
                          model_id: int,
                          exp_name: str,
                          new_user_ratio: float):

    if new_user_ratio > 0.2:
        raise ValueError("new_user_ratio>0.2だと完全にuserでsplitされちゃう")
    mlflow.start_run()

    mlflow.log_param("exp_name", exp_name)
    mlflow.log_param("model_id", model_id)
    mlflow.log_param("count_row", len(df))
    mlflow.log_param("count_column", len(df.columns))

    for key, value in params.items():
        mlflow.log_param(key, value)
    features = [x for x in df.columns if x != "answered_correctly"]

    df_imp = pd.DataFrame()
    df_imp["feature"] = features

    train_idx = []
    val_idx = []
    np.random.seed(0)
    for _, w_df in df.groupby("user_id"):
        if random.random() < new_user_ratio:
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = (np.random.random(len(w_df)) < 0.8/(1-new_user_ratio)).sum()
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    df_train, df_val = df.loc[train_idx], df.loc[val_idx]
    print(len(df_train), len(df_val))
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    valid_data = lgb.Dataset(df_val[features],
                             label=df_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
    mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
    y_oof = model.predict(df_val[features])

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}.pickle", "wb") as f:
        pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df_val.index
    df_oof["predict"] = y_oof
    df_oof["target"] = df_val["answered_correctly"]

    df_oof.to_csv(f"{output_dir}/oof_{model_id}.csv", index=False)

    mlflow.end_run()
