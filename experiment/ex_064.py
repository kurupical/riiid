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
    PreviousLecture, \
    ContentLevelEncoder
from experiment.common import get_logger
import pandas as pd
from model.lgbm import train_lgbm_cv
from model.cboost import train_catboost_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import numpy as np
import os
import glob
import time
import tqdm
import pickle
from sklearn.metrics import roc_auc_score

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

is_debug = True
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)
def calc_optimized_weight(df):
    best_score = 0
    best_cat_ratio = 0
    for cat_ratio in np.arange(0, 1.05, 0.05):
        pred = df["cat"] * cat_ratio + df["lgbm"] * (1 - cat_ratio)
        score = roc_auc_score(df["target"].values, pred)
        print("[cat_ratio: {:.2f}] AUC: {:.4f}".format(cat_ratio, score))
        if score > best_score:
            best_score = score
            best_cat_ratio = cat_ratio

    return best_score, best_cat_ratio

def make_feature_factory_manager(split_num, model_id=None):
    logger = get_logger()

    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator(is_partial_fit=True)
    }
    for column in ["user_id", "content_id", "part", "prior_question_had_explanation",
                   "tags1", "tags2",
                   ("user_id", "prior_question_had_explanation"), ("user_id", "part"),
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
                                                                                    column="timestamp")
    feature_factory_dict["user_id"]["ShiftDiffEncoderContentId"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="content_id")
    for column in ["user_id", "content_id"]:
        feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                           agg_column="prior_question_elapsed_time",
                                                                                                           remove_now=True)

    feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    feature_factory_dict["content_id"]["ContentLevelEncoder2UserId"] = ContentLevelEncoder(vs_column="user_id", is_partial_fit=True)
    feature_factory_dict["user_id"]["MeanAggregatorContentLevel"] = MeanAggregator(column="user_id",
                                                                                   agg_column="content_level_user_id",
                                                                                   remove_now=False)
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
        "PriorQuestionElapsedTimeBinningEncoder": PriorQuestionElapsedTimeBinningEncoder(is_partial_fit=True)
    }
    feature_factory_dict[("part", "prior_question_elapsed_time_bin")] = {
        "CountEncoder": CountEncoder(column=["part", "prior_question_elapsed_time_bin"]),
        "TargetEncoder": TargetEncoder(column=["part", "prior_question_elapsed_time_bin"])
    }
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
    df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True).head(3000)
    model_id = os.path.basename(fname).replace(".pickle", "")
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=1, model_id=model_id)

    train_idx = []
    val_idx = []
    np.random.seed(0)
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
    feature_factory_manager = make_feature_factory_manager(split_num=1, model_id=model_id)
    feature_factory_manager.fit(df2.iloc[train_idx], is_first_fit=True)
    df2_val = []
    for i in tqdm.tqdm(range(len(val_idx))):
        w_df = df2.iloc[val_idx[i:i+1]]
        ww_df = feature_factory_manager.partial_predict(w_df.copy())
        df2_val.append(ww_df)
        feature_factory_manager.fit(ww_df)
    feature_factory_manager = make_feature_factory_manager(split_num=1, model_id=model_id)
    df2_train = feature_factory_manager.all_predict(df2.iloc[train_idx])
    df2_val = pd.concat(df2_val)
    df2_val = df2_val.drop(["user_answer", "tags", "type_of"], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv("exp064_train.csv", index=False)
    df_val.to_csv("exp064_all.csv", index=False)
    df2_val.to_csv("exp064_partial.csv", index=False)

    for col in df_val.columns:
        try:
            np.testing.assert_almost_equal(df_val[col].fillna(-999).values, df2_val[col].fillna(-999).values, decimal=4)
        except:
            print(f"diffあり: {col}")
            out = pd.DataFrame()
            out["user_id"] = df_val["user_id"].values
            out["content_id"] = df_val["content_id"].values
            out["all"] = df_val[col].values
            out["partial"] = df2_val[col].values
            out.to_csv(f"{output_dir}/{col}.csv", index=False)
    break