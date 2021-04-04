import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import tempfile
from feature_engineering.feature_factory import FeatureFactoryManager

import pickle
import mlflow
import random
import os
import gc
import tqdm

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
    for _, w_df in df.groupby("user_id"):
        train_num = (np.random.random(len(w_df)) < 0.8).sum()
        train_idx.extend(w_df[:train_num].index.tolist())
        val_idx.extend(w_df[train_num:].index.tolist())

    print("make_train_data")
    train_data = lgb.Dataset(df.loc[train_idx][features],
                             label=df.loc[train_idx]["answered_correctly"])
    print("make_test_data")
    valid_data = lgb.Dataset(df.loc[val_idx][features],
                             label=df.loc[val_idx]["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    y_oof = model.predict(df.loc[val_idx][features])
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)


def train_lgbm_cv_newuser(df: pd.DataFrame,
                          params: dict,
                          output_dir: str,
                          model_id: int,
                          exp_name: str,
                          drop_user_id: bool,
                          categorical_feature: list=[],
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
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df.loc[train_idx][features],
                             label=df.loc[train_idx]["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df.loc[val_idx][features],
                             label=df.loc[val_idx]["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof = model.predict(df.loc[val_idx][features])
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)

def train_lgbm_cv_newuser_alldata(df: pd.DataFrame,
                                  params: dict,
                                  output_dir: str,
                                  model_id: int,
                                  exp_name: str,
                                  drop_user_id: bool,
                                  categorical_feature: list=[],
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
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.01:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.98)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df.loc[train_idx][features],
                             label=df.loc[train_idx]["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df.loc[val_idx][features],
                             label=df.loc[val_idx]["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof = model.predict(df.loc[val_idx][features])
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)


def train_lgbm_cv_newuser_with_tta(df: pd.DataFrame,
                                   params: dict,
                                   output_dir: str,
                                   model_id: int,
                                   exp_name: str,
                                   drop_user_id: bool,
                                   categorical_feature: list=[],
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
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df.loc[train_idx][features],
                             label=df.loc[train_idx]["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df.loc[val_idx][features],
                             label=df.loc[val_idx]["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof1 = model.predict(df.loc[val_idx][features])
    w_df = df.loc[val_idx][features]
    w_df["rating_diff_content_user_id"] = w_df["rating_diff_content_user_id"] - 15
    y_oof2 = model.predict(w_df[features])
    w_df = df.loc[val_idx][features]
    w_df["rating_diff_content_user_id"] = w_df["rating_diff_content_user_id"] + 15
    y_oof3 = model.predict(w_df[features])

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict1"] = y_oof1
    df_oof["predict2"] = y_oof2
    df_oof["predict3"] = y_oof3
    df_oof["predict"] = y_oof1*0.5 * y_oof2*0.25 + y_oof3*0.25
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values
    print("--- tta(rating) ---")
    print("rate-15: {}".format(roc_auc_score(df_oof["target"].values, df_oof["predict2"].values)))
    print("rate+15: {}".format(roc_auc_score(df_oof["target"].values, df_oof["predict3"].values)))
    print("tta(3): {}".format(roc_auc_score(df_oof["target"].values, df_oof["predict"].values)))


    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)


def train_lgbm_cv_newuser_condition(df: pd.DataFrame,
                                    condition: str,
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
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.1:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.9)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df.loc[train_idx].query(condition)[features],
                             label=df.loc[train_idx].query(condition)["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df.loc[val_idx].query(condition)[features],
                             label=df.loc[val_idx].query(condition)["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    y_oof = model.predict(df.loc[val_idx][features])
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)




def train_lgbm_cv_newuser_user_answer(df: pd.DataFrame,
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
        features = [x for x in df.columns if x not in ["answered_correctly", "user_id", "user_answer"]]
    else:
        features = [x for x in df.columns if x not in ["answered_correctly", "user_answer"]]
    df_imp = pd.DataFrame()
    df_imp["feature"] = features

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

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df.loc[train_idx][features],
                             label=df.loc[train_idx]["user_answer"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df.loc[val_idx][features],
                             label=df.loc[val_idx]["user_answer"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        verbose_eval=10
    )
    y_oof = model.predict(df.loc[val_idx][features])
    print(model.best_score)
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["multi_logloss"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["multi_logloss"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict_0"] = y_oof[:, 0]
    df_oof["predict_1"] = y_oof[:, 1]
    df_oof["predict_2"] = y_oof[:, 2]
    df_oof["predict_3"] = y_oof[:, 3]
    df_oof["target"] = df.loc[val_idx]["user_answer"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)

def train_lgbm_cv_newuser_train95(df: pd.DataFrame,
                                  params: dict,
                                  output_dir: str,
                                  model_id: int,
                                  exp_name: str,
                                  drop_user_id: bool,
                                  categorical_feature: list=[],
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
    for _, w_df in df.groupby("user_id"):
        if np.random.random() < 0.01:
            # all val
            val_idx.extend(w_df.index.tolist())
        else:
            train_num = int(len(w_df) * 0.95)
            train_idx.extend(w_df[:train_num].index.tolist())
            val_idx.extend(w_df[train_num:].index.tolist())

    print(f"make_train_data len={len(train_idx)}")
    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]

    del df
    gc.collect()
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data = lgb.Dataset(df_val[features],
                             label=df_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof = model.predict(df_val[features])
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df_val.index
    df_oof["predict"] = y_oof
    df_oof["target"] = df_val["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)


def train_lgbm_cv_newuser_with_iteration(df: pd.DataFrame,
                                         feature_factory_manager: FeatureFactoryManager,
                                         params: dict,
                                         output_dir: str,
                                         model_id: int,
                                         exp_name: str,
                                         drop_user_id: bool,
                                         categorical_feature: list=[],
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
        features = [x for x in df.columns if x not in ["answered_correctly", "user_id", "user_answer", "tags", "type_of", "bundle_id", "previous_5_ans"]]
    else:
        features = [x for x in df.columns if x not in ["answered_correctly", "user_answer", "tags", "type_of", "bundle_id", "previous_5_ans"]]
    df_imp = pd.DataFrame()
    df_imp["feature"] = features

    df1 = feature_factory_manager.all_predict(df.copy())

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

    val_idx = val_idx[:1000000]
    df1 = df1.drop(["user_answer", "tags", "type_of", "bundle_id", "previous_5_ans"], axis=1)
    df1.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df1.columns]
    df_train = df1.loc[train_idx]
    df_train = df_train[df_train["answered_correctly"].notnull()]
    df_val = df1.loc[val_idx]
    df_val = df_val[df_val["answered_correctly"].notnull()]

    # valid2
    feature_factory_manager.fit(df.loc[train_idx])

    df2 = []
    for i in tqdm.tqdm(range(len(val_idx)//100)):
        w_df = df.loc[val_idx[i*100:(i+1)*100]]
        df2.append(feature_factory_manager.partial_predict(w_df))
        feature_factory_manager.fit(w_df)
    df2 = pd.concat(df2)
    df2.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df2.columns]
    df2_val = df2[df2["answered_correctly"].notnull()]

    print(df_val)
    print(df2_val)
    assert len(df_val) == len(df2_val)

    print(f"make_train_data len={len(train_idx)}")
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    print(f"make_test_data len={len(val_idx)}")
    valid_data1 = lgb.Dataset(df_val[features],
                              label=df_val["answered_correctly"])
    valid_data2 = lgb.Dataset(df2_val[features],
                              label=df2_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data1, valid_data2],
        verbose_eval=100
    )
    print(roc_auc_score(df_val["answered_correctly"], model.predict(df_val[features])))
    print(roc_auc_score(df2_val["answered_correctly"], model.predict(df2_val[features])))

    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof = model.predict(df.loc[val_idx][features])
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df.loc[val_idx].index
    df_oof["predict"] = y_oof
    df_oof["target"] = df.loc[val_idx]["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)


def train_lgbm_cv_2500kval(df: pd.DataFrame,
                          params: dict,
                          output_dir: str,
                          model_id: int,
                          exp_name: str,
                          drop_user_id: bool,
                          categorical_feature: list=[],
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

    df_val_row = pd.read_feather(
        "../input/riiid-test-answer-prediction/train_transformer_last2500k_only_row_id.feather")
    if is_debug:
        df_val_row = df_val_row.head(3000)
    df_val_row["is_val"] = 1

    df = pd.merge(df, df_val_row, how="left", on="row_id")
    df["is_val"] = df["is_val"].fillna(0)

    df_train = df[df["is_val"] == 0]
    df_val = df[df["is_val"] == 1]
    print(f"make_train_data len={len(df_train)}")
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    print(f"make_test_data len={len(df_val)}")
    valid_data = lgb.Dataset(df_val[features],
                             label=df_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        categorical_feature=categorical_feature,
        valid_sets=[train_data, valid_data],
        verbose_eval=100
    )
    if not is_debug:
        mlflow.log_metric("auc_train", model.best_score["training"]["auc"])
        mlflow.log_metric("auc_val", model.best_score["valid_1"]["auc"])
        mlflow.end_run()

    df_imp["importance"] = model.feature_importance("gain") / model.feature_importance("gain").sum()
    df_imp.sort_values("importance", ascending=False).to_csv(f"{output_dir}/imp_{model_id}.csv")
    with open(f"{output_dir}/model_{model_id}_lgbm.pickle", "wb") as f:
        pickle.dump(model, f)

    y_oof = model.predict(df_val[features])
    df_oof = pd.DataFrame()
    df_oof["row_id"] = df_val["row_id"]
    df_oof["predict"] = y_oof
    df_oof["target"] = df_val["answered_correctly"].values

    df_oof.to_csv(f"{output_dir}/oof_{model_id}_lgbm.csv", index=False)
