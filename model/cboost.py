from catboost import CatBoostClassifier
from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np
import tempfile

import pickle
import mlflow
import random
import os
from sklearn.metrics import roc_auc_score

def train_catboost_cv(df: pd.DataFrame,
                      params: dict,
                      output_dir: str,
                      model_id: int,
                      exp_name: str,
                      drop_user_id: bool,
                      experiment_id: int=6,
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
    for _, w_df in df.groupby("user_id"):
        train_num = (np.random.random(len(w_df)) < 0.8).sum()
        train_idx.extend(w_df[:train_num].index.tolist())
        val_idx.extend(w_df[train_num:].index.tolist())

    model = CatBoostClassifier(**params)
    model.fit(df.loc[train_idx][features],
              df.loc[train_idx]["answered_correctly"],
              eval_set=(df.loc[val_idx][features], df.loc[val_idx]["answered_correctly"]))

    y_train = model.predict_proba(df.loc[train_idx][features])[:, 1]
    y_oof = model.predict_proba(df.loc[val_idx][features])[:, 1]
    auc_train = roc_auc_score(df.loc[train_idx]["answered_correctly"].values.flatten(), y_train.flatten())
    auc_val = roc_auc_score(df.loc[val_idx]["answered_correctly"].values.flatten(), y_oof.flatten())

    if not is_debug:
        mlflow.log_metric("auc_train", auc_train)
        mlflow.log_metric("auc_val", auc_val)
        mlflow.end_run()

    model.save_model(f"{output_dir}/model_{model_id}_catboost")

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_catboost.csv", index=False)

