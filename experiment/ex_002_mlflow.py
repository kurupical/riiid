from pipeline.p_001_baseline import transform
import pandas as pd
from model.lgbm import train_lgbm_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import os
import glob

output_dir = f"../output/ex_001/{dt.now().strftime('%Y%m%d%H%M%S')}/"

for model_id, fname in enumerate(glob.glob("../input/riiid-test-answer-prediction/split10/*")):
    print(fname)
    df = pd.read_pickle(fname)

    df = transform(df)

    os.makedirs(output_dir, exist_ok=True)
    params = {
        'objective': 'binary',
        'num_leaves': 32,
        'min_data_in_leaf': 15,  # 42,
        'max_depth': -1,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'bagging_fraction': 0.7,  # 0.5,
        'feature_fraction': 0.5,
        'bagging_seed': 0,
        'reg_alpha': 0.1,  # 1.728910519108444,
        'reg_lambda': 1,
        'random_state': 0,
        'metric': 'auc',
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 100
    }

    df = df.drop(["user_answer"], axis=1)

    train_lgbm_cv(df,
                  params=params,
                  output_dir=output_dir,
                  model_id=model_id,
                  exp_name="exp_002")
    break