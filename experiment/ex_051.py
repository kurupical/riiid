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
    Counter, \
    PreviousAnswer, \
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder, \
    PriorQuestionElapsedTimeBinningEncoder, \
    PreviousAnswer2
from experiment.common import get_logger
import pandas as pd
from model.lgbm import train_lgbm_cv
from model.nn import train_nn_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm
import pickle
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization, Input, Add, PReLU
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization, Input, Add, PReLU
from tensorflow.keras import regularizers
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import roc_auc_score
import mlflow

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

is_debug = False
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)

def calc_optimized_weight(df):
    best_score = 0
    best_nn_ratio = 0
    for nn_ratio in np.arange(0, 1.05, 0.05):
        pred = df["nn"] * nn_ratio + df["lgbm"] * (1 - nn_ratio)
        score = roc_auc_score(df["target"].values, pred)
        print("[nn_ratio: {:.2f}] AUC: {:.4f}".format(nn_ratio, score))
        if score > best_score:
            best_score = score
            best_nn_ratio = nn_ratio

    return best_score, best_nn_ratio

def get_model(input_len,
              reg,
              hidden1,
              hidden2
              ):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(hidden1,
                    kernel_regularizer=regularizers.l2(reg),
                    activity_regularizer=regularizers.l2(reg),
                    input_shape=(input_len,)))
    model.add(PReLU())
    model.add(Dense(hidden2,
                    kernel_regularizer=regularizers.l2(reg),
                    activity_regularizer=regularizers.l2(reg)))
    model.add(PReLU())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(),
                  metrics=tensorflow.keras.metrics.AUC())

    return model

def make_feature_factory_manager(split_num):
    logger = get_logger()

    feature_factory_dict = {}
    for column in ["content_id", "user_id", "prior_question_had_explanation", ("user_id", "part"),
                   ("content_id", "prior_question_had_explanation")]:
        is_partial_fit = (column == "content_id" or column == "user_id")

        if type(column) == str:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=column, is_partial_fit=is_partial_fit),
                "TargetEncoder": TargetEncoder(column=column, is_partial_fit=is_partial_fit)
            }
        else:
            feature_factory_dict[column] = {
                "CountEncoder": CountEncoder(column=list(column), is_partial_fit=is_partial_fit),
                "TargetEncoder": TargetEncoder(column=list(column), is_partial_fit=is_partial_fit)
            }
    feature_factory_dict["user_id"]["ShiftDiffEncoderTimestamp"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="timestamp",
                                                                                    is_partial_fit=True)
    feature_factory_dict["user_id"]["ShiftDiffEncoderContentId"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="content_id")
    for column in ["user_id", "content_id"]:
        feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                           agg_column="prior_question_elapsed_time",
                                                                                                           remove_now=True)

    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    feature_factory_dict["user_id"]["UserCountBinningEncoder"] = UserCountBinningEncoder(is_partial_fit=True)
    feature_factory_dict["user_count_bin"] = {}
    feature_factory_dict["user_count_bin"]["CountEncoder"] = CountEncoder(column="user_count_bin")
    feature_factory_dict["user_count_bin"]["TargetEncoder"] = TargetEncoder(column="user_count_bin")
    feature_factory_dict[("user_id", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["user_id", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["user_id", "user_count_bin"])
    }
    feature_factory_dict[("content_id", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["content_id", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["content_id", "user_count_bin"])
    }
    feature_factory_dict[("prior_question_had_explanation", "user_count_bin")] = {
        "CountEncoder": CountEncoder(column=["prior_question_had_explanation", "user_count_bin"]),
        "TargetEncoder": TargetEncoder(column=["prior_question_had_explanation", "user_count_bin"])
    }

    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[2, 5])
    feature_factory_dict["user_count_bin"]["CategoryLevelEncoderUserCountBin"] = \
        CategoryLevelEncoder(groupby_column="user_id",
                             agg_column="user_count_bin",
                             categories=[0])

    feature_factory_dict["prior_question_elapsed_time"] = {
        "PriorQuestionElapsedTimeBinningEncoder": PriorQuestionElapsedTimeBinningEncoder()
    }
    feature_factory_dict[("part", "prior_question_elapsed_time_bin")] = {
        "CountEncoder": CountEncoder(column=["part", "prior_question_elapsed_time_bin"]),
        "TargetEncoder": TargetEncoder(column=["part", "prior_question_elapsed_time_bin"])
    }
    feature_factory_dict[("user_id", "content_id")] = {
        "PreviousAnswer2": PreviousAnswer2(column=["user_id", "content_id"])
    }
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num)
    return feature_factory_manager

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10)
    df = feature_factory_manager.all_predict(df)
    os.makedirs(output_dir, exist_ok=True)
    params = {
        'objective': 'binary',
        'num_leaves': 96,
        'max_depth': -1,
        'learning_rate': 0.3,
        'boosting': 'gbdt',
        'bagging_fraction': 0.5,
        'feature_fraction': 0.7,
        'bagging_seed': 0,
        'reg_alpha': 100,  # 1.728910519108444,
        'reg_lambda': 20,
        'random_state': 0,
        'metric': 'auc',
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 50
    }
    df.tail(1000).to_csv("exp028.csv", index=False)

    df = df[df["answered_correctly"].notnull()]
    print("lgbm")
    print(df.columns)
    print(df.shape)

    model_id = os.path.basename(fname).replace(".pickle", "")
    print(model_id)
    train_lgbm_cv(df.drop(["user_answer", "tags", "type_of"], axis=1),
                  params=params,
                  output_dir=output_dir,
                  model_id=model_id,
                  exp_name=model_id,
                  is_debug=is_debug,
                  drop_user_id=True)
    useful_cols = [
        "target_enc_content_id",
        "target_enc_['content_id', 'prior_question_had_explanation']",
        "user_rate_mean_content_id",
        "user_rate_sum_content_id",
        "shiftdiff_timestamp_by_user_id",
        "previous_answer_['user_id', 'content_id']",
        "target_enc_['content_id', 'user_count_bin']",
        "target_enc_['user_id', 'part']",
        "diff_user_level_target_enc_content_id",
        "user_id",
        "answered_correctly"
    ]

    df = df[useful_cols]
    df = df.fillna(-1)
    print("nn")
    print(df.columns)
    print(df.shape)

    params = {
        "input_len": len(df.columns) - 2, # 2: answered_correctly, user_id
        "hidden1": 64,
        "hidden2": 32,
        "reg": 5e-5
    }
    model = get_model(**params)

    model_id = os.path.basename(fname).replace(".pickle", "")
    print(model_id)
    train_nn_cv(df,
                model=model,
                output_dir=output_dir,
                params=params,
                model_id=model_id,
                exp_name=model_id,
                is_debug=is_debug,
                drop_user_id=True,
                experiment_id=4)

    df_oof_lgbm = pd.read_csv(f"{output_dir}/oof_{model_id}_lgbm.csv")
    df_oof_nn = pd.read_csv(f"{output_dir}/oof_{model_id}_nn.csv")

    df_oof = pd.DataFrame()
    df_oof["target"] = df_oof_lgbm["target"]
    df_oof["lgbm"] = df_oof_lgbm["predict"]
    df_oof["nn"] = df_oof_nn["predict"]

    score, weight = calc_optimized_weight(df_oof)
    mlflow.start_run(experiment_id=5, run_name=model_id)

    mlflow.log_param("model_id", model_id)
    mlflow.log_param("count_row", len(df))
    mlflow.log_param("count_column", len(df.columns))
    mlflow.log_param("nn_weight", weight)
    mlflow.log_metric("auc", score)
    mlflow.end_run()

# fit
df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                          dtype={"bundle_id": "int32",
                                 "question_id": "int32",
                                 "correct_answer": "int8",
                                 "part": "int8"})
df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                         dtype={"lecture_id": "int32",
                                "tag": "int16",
                                "part": "int8"})
feature_factory_manager = make_feature_factory_manager(split_num=1)

for fname in glob.glob("../input/riiid-test-answer-prediction/split10_base/*"):
    data_types_dict = {
        'row_id': 'int64',
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'user_answer': 'int8',
        'answered_correctly': 'int8',
    }

    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    df = pd.concat([pd.merge(df[df["content_type_id"] == 0], df_question,
                             how="left", left_on="content_id", right_on="question_id"),
                    pd.merge(df[df["content_type_id"] == 1], df_lecture,
                             how="left", left_on="content_id", right_on="lecture_id")]).sort_values(
        ["user_id", "timestamp"])
    # df = feature_factory_manager.feature_factory_dict["content_id"]["TargetEncoder"].all_predict(df)
    feature_factory_manager.fit(df, is_first_fit=True)
for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = None
feature_factory_manager.logger = None
with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
    pickle.dump(feature_factory_manager, f)