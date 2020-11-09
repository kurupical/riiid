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
    PreviousAnswer2, \
    PreviousAnswer3
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
import pickle

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

is_debug = True
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)

def make_feature_factory_manager(split_num, model_id=None):
    logger = get_logger()

    feature_factory_dict = {}
    feature_factory_dict["user_id"] = {}
    feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                         column="content_id",
                                                                         is_debug=is_debug,
                                                                         model_id=model_id)

    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num)
    return feature_factory_manager

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True).head(50000)
    model_id = os.path.basename(fname).replace(".pickle", "")
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10, model_id=model_id)

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
    print(feature_factory_manager.feature_factory_dict["user_id"]["PreviousAnswer2"].data_dict[408250])
    df2_val = []
    for i in tqdm.tqdm(range(len(val_idx)//3)):
        w_df = df2.iloc[val_idx[i*3:(i+1)*3]]
        df2_val.append(feature_factory_manager.partial_predict(w_df))
        feature_factory_manager.fit(w_df)
    df2_val = pd.concat(df2_val)
    df2_val = df2_val.drop(["user_answer", "tags", "type_of"], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv("exp055_train.csv", index=False)
    df_val.to_csv("exp055_all.csv", index=False)
    df2_val.to_csv("exp055_partial.csv", index=False)
