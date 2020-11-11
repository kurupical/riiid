from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    NUniqueEncoder, \
    TagsSeparator, \
    ShiftDiffEncoder, \
    UserLevelEncoder2, \
    TargetEncodeVsUserId, \
    Counter
from experiment.common import get_logger
import pandas as pd
from model.lgbm import train_lgbm_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm
import lightgbm as lgb

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)

    # all_predict
    df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    # df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)]).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    logger = get_logger()
    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator()
    }
    for column in ["content_id", "content_type_id", "user_id", "prior_question_had_explanation",
                   "tags1", "tags2", "tags3", "tags4", "tags5", "tags6",
                   ("user_id", "content_type_id"), ("user_id", "prior_question_had_explanation")]:
        is_partial_fit = (column == "content_id" or column == "content_type_id")

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
    feature_factory_dict["user_id"]["ShiftDiffEncoder"] = ShiftDiffEncoder(groupby="user_id",
                                                                           column="timestamp")
    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    feature_factory_dict["user_id"]["UserLevelEncoder2ContentTypeId"] = UserLevelEncoder2(vs_column="content_type_id")
    feature_factory_dict["content_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="content_id",
                                                                                                  agg_column="prior_question_elapsed_time",
                                                                                                  remove_now=True)
    for col in ["part", "content_type_id", "prior_question_had_explanation", "type_of"]:
        feature_factory_dict["user_id"][f"Counter{col}"] = Counter(groupby_column="user_id",
                                                                   agg_column=col,
                                                                   categories=df[col].drop_duplicates())
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=10)
    train_idx = []
    val_idx = []
    for _, w_df in df.groupby("user_id"):
        train_num = (np.random.random(len(w_df)) < 0.8).sum()
        train_idx.extend(w_df[:train_num].index.tolist())
        val_idx.extend(w_df[train_num:].index.tolist())

    df = feature_factory_manager.all_predict(pd.concat([df.iloc[train_idx], df.iloc[val_idx]]))
    df = df.drop(["user_answer", "tags", "type_of"], axis=1)
    df_train = df.iloc[:len(train_idx)]
    df_val = df.iloc[len(train_idx):]
    print(df_train)

    df2 = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    # df2 = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)]).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df2["answered_correctly"] = df2["answered_correctly"].replace(-1, np.nan)
    df2["prior_question_had_explanation"] = df2["prior_question_had_explanation"].fillna(-1).astype("int8")
    df2_train = feature_factory_manager.all_predict(df2.iloc[train_idx])
    print(df2_train)
    feature_factory_manager.fit(df2.iloc[train_idx], is_first_fit=True)
    df2_val = []
    for i in tqdm.tqdm(range(len(val_idx)//3)):
        w_df = df2.iloc[val_idx[i*3:(i+1)*3]]
        df2_val.append(feature_factory_manager.partial_predict(w_df))
        feature_factory_manager.fit(w_df)
    df2_val = pd.concat(df2_val)
    df2_val = df2_val.drop(["user_answer", "tags", "type_of"], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    df_val.to_csv("exp055_all.csv", index=False)
    df2_val.to_csv("exp055_partial.csv", index=False)
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
        "n_estimators": 10000,
        "early_stopping_rounds": 100
    }

    features = [x for x in df.columns if x not in ["answered_correctly"]]

    df_train = df_train[df_train["answered_correctly"].notnull()]
    df_val = df_val[df_val["answered_correctly"].notnull()]
    df2_val = df2_val[df2_val["answered_correctly"].notnull()]

    print("make_train_data")
    train_data = lgb.Dataset(df_train[features],
                             label=df_train["answered_correctly"])
    print("make_test_data")
    valid_data1 = lgb.Dataset(df_val[features],
                              label=df_val["answered_correctly"])
    valid_data2 = lgb.Dataset(df2_val[features],
                              label=df2_val["answered_correctly"])

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data1, valid_data2],
        verbose_eval=100
    )
    break
