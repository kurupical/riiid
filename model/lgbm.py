import lightgbm as lgb
from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np
import pickle

def train_lgbm(df: pd.DataFrame,
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