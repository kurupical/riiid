
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    TagsSeparator, \
    UserLevelEncoder, \
    NUniqueEncoder, \
    ShiftDiffEncoder
from feature_engineering.environment import MyEnvironment, EnvironmentManager
from experiment.common import get_logger, merge, read_data
import pandas as pd
from model.lgbm import train_lgbm_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm
import random
import lightgbm as lgb

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

np.random.seed(0)
df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                          dtype={"bundle_id": "int32",
                                 "question_id": "int32",
                                 "correct_answer": "int8",
                                 "part": "int8"})
df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                         dtype={"lecture_id": "int32",
                                "tag": "int16",
                                "part": "int8"})

df = read_data("../input/riiid-test-answer-prediction/split10_base/train_0.pickle").reset_index(drop=True)
df["row_id"] = np.arange(len(df))


logger = get_logger()
feature_factory_dict = {}
feature_factory_dict["tags"] = {
    "TagsSeparator": TagsSeparator()
}
for column in ["content_id", "user_id", "content_type_id", "prior_question_had_explanation",
               "tags1", "tags2", "tags3", "tags4", "tags5", "tags6",
               ("user_id", "content_type_id"), ("user_id", "prior_question_had_explanation")]:
    is_partial_fit = column == "content_id"

    if type(column) == str:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=column),
            "TargetEncoder": TargetEncoder(column=column, is_partial_fit=is_partial_fit)
        }
    else:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=list(column)),
            "TargetEncoder": TargetEncoder(column=list(column), is_partial_fit=is_partial_fit)
        }

for column in ["part", ("user_id", "tag"), ("user_id", "part"), ("content_type_id", "part")]:
    if type(column) == str:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=column)
        }
    else:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=list(column))
        }


feature_factory_dict["user_id"]["MeanAggregatorTimestamp"] = MeanAggregator(column="user_id",
                                                                            agg_column="timestamp",
                                                                            remove_now=False)
feature_factory_dict["user_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="user_id",
                                                                                           agg_column="prior_question_elapsed_time",
                                                                                           remove_now=True)
feature_factory_dict["user_id"]["UserLevelEncoder"] = UserLevelEncoder()
feature_factory_dict["user_id"]["NUniqueEncoderContentId"] = NUniqueEncoder(groupby="user_id",
                                                                            column="content_id")
feature_factory_dict["user_id"]["NUniqueEncoderTaskContainerId"] = NUniqueEncoder(groupby="user_id",
                                                                                  column="task_container_id")
feature_factory_dict["user_id"]["ShiftDiffEncoder"] = ShiftDiffEncoder(groupby="user_id",
                                                                       column="timestamp")
feature_factory_dict["content_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="content_id",
                                                                                              agg_column="prior_question_elapsed_time",
                                                                                              remove_now=True)

feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                logger=logger)

new_user_ratio = 0.01

train_idx = []
val_idx = []
for _, w_df in df.groupby("user_id"):
    if random.random() < new_user_ratio:
        val_idx.extend(w_df.index.tolist())
    else:
        train_num = (np.random.random(len(w_df)) < 0.9/(1-new_user_ratio)).sum()
        train_idx.extend(w_df[:train_num].index.tolist())
        val_idx.extend(w_df[train_num:].index.tolist())
print(len(train_idx))
print(len(df))
df_train = df.iloc[train_idx]
df_val = df.iloc[val_idx]

df = merge(df=df, df_question=df_question, df_lecture=df_lecture)
df_train = merge(df=df_train, df_question=df_question, df_lecture=df_lecture)

df_all_fit = feature_factory_manager.all_predict(df)
df_train_all_fit = df_all_fit[df_all_fit["row_id"].isin(df_train["row_id"].values)]
df_val_all_fit = df_all_fit[~df_all_fit["row_id"].isin(df_train["row_id"].values)]

feature_factory_manager.fit(df_train, all_predict_mode=True)

gen = MyEnvironment(df_test=df_val,
                    interval=1).iter_test()
env_manager = EnvironmentManager(feature_factory_manager=feature_factory_manager,
                                 gen=gen,
                                 fit_interval=150,
                                 df_question=df_question,
                                 df_lecture=df_lecture)

i = 0

df_val2 = pd.DataFrame()
while True:
    if i % 100 == 0: print(i)
    i += 1
    x = env_manager.step()
    if x is None:
        break
    df_test = x[0]
    df_sub = x[1]
    df_val2 = pd.concat([df_val2, df_test], axis=0)

df_train_all_fit.columns = [x.replace(" ", "_") for x in df_train_all_fit.columns]
df_val2.columns = [x.replace(" ", "_") for x in df_val2.columns]
df_val2 = pd.merge(df_val2, df_val[["row_id", "answered_correctly"]], how="left")

params = {
    'objective': 'binary',
    'num_leaves': 32,
    'min_data_in_leaf': 15,  # 42,
    'max_depth': -1,
    'learning_rate': 0.3,
    'boosting': 'gbdt',
    'bagging_fraction': 0.7,  # 0.5,
    'feature_fraction': 0.9,
    'bagging_seed': 0,
    'reg_alpha': 5,  # 1.728910519108444,
    'reg_lambda': 5,
    'random_state': 0,
    'metric': 'auc',
    'verbosity': -1,
    "n_estimators": 1000,
}

df_train_all_fit = df_train_all_fit.drop(["user_answer", "tags", "type_of"], axis=1)
df_train_all_fit = df_train_all_fit[df_train_all_fit["answered_correctly"].notnull()]
features = [x for x in df_train_all_fit.columns if x not in ["answered_correctly"]]
train_data = lgb.Dataset(df_train_all_fit[features],
                         label=df_train_all_fit["answered_correctly"])
valid_data1 = lgb.Dataset(df_val_all_fit[features],
                          label=df_val_all_fit["answered_correctly"])
valid_data2 = lgb.Dataset(df_val2[features],
                          label=df_val["answered_correctly"])

print(df_train_all_fit.shape)
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data1, valid_data2],
    verbose_eval=100
)