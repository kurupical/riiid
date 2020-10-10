from pipeline.p_001_baseline import transform
import pandas as pd
from model.lgbm import train_lgbm
from sklearn.model_selection import KFold
from datetime import datetime as dt
import os

# df = pd.read_pickle("../input/riiid-test-answer-prediction/train.pickle").head(1000000)
df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", nrows=1000000)
# df_questions = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv")
# df_lectures = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv")

# df["content_id"] = df["content_id"].astype(str) + "_" + df["content_type_id"].astype(str)
# df_questions["question_id"] = df_questions["question_id"].astype(str) + "-" + "0"
# df_lectures["lecture_id"] = df_lectures["lecture_id"].astype(str) + "-" + "0"

# df = pd.merge(df, df_questions, how="left", left_on="content_id", right_on="question_id")
# df = pd.merge(df, df_lectures, how="left", left_on="content_id", right_on="lecture_id")

# df.to_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")

# print(len(df))
# df = pd.read_pickle("../input/riiid-test-answer-prediction/train_merged.pickle")

df = transform(df)

output_dir = f"../output/ex_001/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir)
params = {
    'objective': 'binary',
    'num_leaves': 32,
    'min_data_in_leaf': 15,  # 42,
    'max_depth': -1,
    'learning_rate': 0.1,
    'boosting': 'gbdt',
    'bagging_fraction': 0.2,  # 0.5,
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

train_lgbm(df,
           fold=KFold(n_splits=5),
           params=params,
           output_dir=output_dir)