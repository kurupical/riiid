from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    NUniqueEncoder, \
    TagsSeparator, \
    ShiftDiffEncoder, \
    UserLevelEncoder2, \
    Counter, \
    PreviousAnswer, \
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder, \
    PriorQuestionElapsedTimeBinningEncoder, \
    PreviousAnswer2, \
    PreviousLecture, \
    ContentLevelEncoder, \
    FirstColumnEncoder, \
    FirstNAnsweredCorrectly, \
    TargetEncoderAggregator, \
    SessionEncoder, \
    PreviousNAnsweredCorrectly, \
    QuestionLectureTableEncoder, \
    QuestionLectureTableEncoder2, \
    QuestionQuestionTableEncoder2, \
    UserAnswerLevelEncoder, \
    WeightDecayTargetEncoder, \
    UserContentRateEncoder, \
    StudyTermEncoder, \
    ElapsedTimeVsShiftDiffEncoder, \
    PastNFeatureEncoder

from experiment.common import get_logger, total_size
import pandas as pd
from model.lgbm import train_lgbm_cv, train_lgbm_cv_newuser
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
import warnings


warnings.filterwarnings("ignore")
pd.set_option("max_rows", 100)

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

is_debug = False
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

    for column in ["user_id", "content_id"]:
        is_partial_fit = (column == "content_id" or column == "user_id")

        if type(column) == str:
            feature_factory_dict[column] = {
                "TargetEncoder": TargetEncoder(column=column, is_partial_fit=is_partial_fit)
            }
        else:
            feature_factory_dict[column] = {
                "TargetEncoder": TargetEncoder(column=list(column), is_partial_fit=is_partial_fit)
            }
    feature_factory_dict["user_id"]["ShiftDiffEncoderTimestamp"] = ShiftDiffEncoder(groupby="user_id",
                                                                                    column="timestamp",
                                                                                    is_partial_fit=True)
    feature_factory_dict["user_id"]["PastNTimestampEncoder"] = PastNFeatureEncoder(column="timestamp",
                                                                                   past_ns=[5, 50],
                                                                                   agg_funcs=["vslast"],
                                                                                   remove_now=False)
    feature_factory_dict["user_id"]["StudyTermEncoder"] = StudyTermEncoder(is_partial_fit=True)
    feature_factory_dict["user_id"]["ElapsedTimeVsShiftDiffEncoder"] = ElapsedTimeVsShiftDiffEncoder()
    feature_factory_dict["user_id"]["CountEncoder"] = CountEncoder(column="user_id", is_partial_fit=True)
    feature_factory_dict["user_id"]["UserCountBinningEncoder"] = UserCountBinningEncoder(is_partial_fit=True)
    feature_factory_dict["user_count_bin"] = {}
    feature_factory_dict["user_count_bin"]["TargetEncoder"] = TargetEncoder(column="user_count_bin")
    feature_factory_dict[("user_id", "user_count_bin")] = {
        "TargetEncoder": TargetEncoder(column=["user_id", "user_count_bin"])
    }
    feature_factory_dict[("content_id", "user_count_bin")] = {
        "TargetEncoder": TargetEncoder(column=["content_id", "user_count_bin"])
    }
    feature_factory_dict[("user_id", "part")] = {
        "UserContentRateEncoder": UserContentRateEncoder(column=["user_id", "part"],
                                                         rate_func="elo")
    }

    for column in ["user_id", "content_id", "part", ("user_id", "part")]:
        if column not in feature_factory_dict:
            feature_factory_dict[column] = {}
        if type(column) == str:
            feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                               agg_column="prior_question_elapsed_time",
                                                                                                               remove_now=True)
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                               agg_column="shiftdiff_timestamp_by_user_id_cap200k",
                                                                                                               remove_now=True)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=column,
                                                                                                agg_column="study_time",
                                                                                                remove_now=True)
        else:
            feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                               agg_column="prior_question_elapsed_time",
                                                                                                               remove_now=True)
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                               agg_column="shiftdiff_timestamp_by_user_id_cap200k",
                                                                                                               remove_now=True)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                agg_column="study_time",
                                                                                                remove_now=True)


    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[2, 5])

    feature_factory_dict["prior_question_elapsed_time"] = {
        "PriorQuestionElapsedTimeBinningEncoder": PriorQuestionElapsedTimeBinningEncoder(is_partial_fit=True)
    }
    feature_factory_dict[("part", "prior_question_elapsed_time_bin")] = {
        "TargetEncoder": TargetEncoder(column=["part", "prior_question_elapsed_time_bin"])
    }
    feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                         column="content_id",
                                                                         is_debug=is_debug,
                                                                         model_id=model_id,
                                                                         n=500)
    feature_factory_dict["user_id"]["PreviousNAnsweredCorrectly"] = PreviousNAnsweredCorrectly(n=3,
                                                                                               is_partial_fit=True)
    feature_factory_dict["user_id"]["Counter"] = Counter(groupby_column="user_id",
                                                         agg_column="prior_question_had_explanation",
                                                         categories=[0, 1])

    feature_factory_dict[f"previous_3_ans"] = {
        "TargetEncoder": TargetEncoder(column="previous_3_ans")
    }
    feature_factory_dict["user_id"]["QuestionLectureTableEncoder2"] = QuestionLectureTableEncoder2(model_id=model_id,
                                                                                                   is_debug=is_debug,
                                                                                                   past_n=100,
                                                                                                   min_size=100)
    feature_factory_dict["user_id"]["QuestionQuestionTableEncoder2"] = QuestionQuestionTableEncoder2(model_id=model_id,
                                                                                                     is_debug=is_debug,
                                                                                                     past_n=100,
                                                                                                     min_size=300)
    feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(column="user_id",
                                                                                       rate_func="elo")
    feature_factory_dict["post"] = {
        "ContentIdTargetEncoderAggregator": TargetEncoderAggregator()
    }

    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num,
                                                    model_id=model_id,
                                                    load_feature=not is_debug,
                                                    save_feature=not is_debug)
    return feature_factory_manager

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)]).reset_index(drop=True)
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    model_id = os.path.basename(fname).replace(".pickle", "")
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10, model_id=model_id)
    df = feature_factory_manager.all_predict(df)
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

    df = df.drop(["user_answer", "tags", "type_of", "bundle_id", "previous_3_ans"], axis=1)
    df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df.columns]
    df = df[df["answered_correctly"].notnull()]
    print(df.columns)
    print(df.shape)

    categorical_feature = ["content_id"]
    print(model_id)
    train_lgbm_cv_newuser(df,
                          categorical_feature=categorical_feature,
                          params=params,
                          output_dir=output_dir,
                          model_id=model_id,
                          exp_name=model_id,
                          is_debug=is_debug,
                          drop_user_id=True)
    1/0
    params = {
        'n_estimators': 12000,
        'learning_rate': 0.1,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'random_seed': 0,
        'metric_period': 50,
        'od_wait': 400,
        'task_type': 'GPU',
        'max_depth': 8,
        "verbose": 100
    }
    if is_debug:
        params["n_estimators"] = 100
    train_catboost_cv(df,
                      params=params,
                      output_dir=output_dir,
                      model_id=model_id,
                      exp_name=model_id,
                      is_debug=is_debug,
                      cat_features=None,
                      drop_user_id=True)

    df_oof_lgbm = pd.read_csv(f"{output_dir}/oof_{model_id}_lgbm.csv")
    df_oof_cat = pd.read_csv(f"{output_dir}/oof_{model_id}_catboost.csv")
    df_oof = pd.DataFrame()
    df_oof["target"] = df_oof_lgbm["target"]
    df_oof["lgbm"] = df_oof_lgbm["predict"]
    df_oof["cat"] = df_oof_cat["predict"]

    score, weight = calc_optimized_weight(df_oof)
    if is_debug:
        break
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


for i, fname in enumerate(glob.glob("../input/riiid-test-answer-prediction/split10_base/*")):
    print(fname)
    model_id = os.path.basename(fname).replace(".pickle", "")
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

    feature_factory_manager.model_id = model_id
    for column, dicts in feature_factory_manager.feature_factory_dict.items():
        for factory_name, factory in dicts.items():
            factory.model_id = model_id

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

    if i == 0:
        for k, v in feature_factory_manager.feature_factory_dict.items():
            for kk, vv in v.items():
                try:
                    print(f"{k}-{vv}: len={len(vv.data_dict)} size={round(total_size(vv.data_dict) / 1_000_000, 2)}MB")
                except Exception as e:
                    print(f"{k}-{kk} error")
                    print(e)

for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = None
feature_factory_manager.logger = None
with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
    pickle.dump(feature_factory_manager, f)