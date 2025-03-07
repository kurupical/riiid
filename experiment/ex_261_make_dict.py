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
    PastNFeatureEncoder, \
    PreviousContentAnswerTargetEncoder, \
    DurationPreviousContent,\
    StudyTermEncoder2, \
    UserContentNowRateEncoder,\
    ElapsedTimeMeanByContentIdEncoder, \
    PastNUserAnswerHistory, \
    DurationFeaturePostProcess, \
    CorrectVsIncorrectMeanEncoder

from experiment.common import get_logger, total_size
import pandas as pd
from model.lgbm import train_lgbm_cv_newuser_train95
from model.cboost import train_catboost_cv_train95
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
    feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
    feature_factory_dict["user_id"]["PastNTimestampEncoder"] = PastNFeatureEncoder(column="timestamp",
                                                                                   past_ns=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                                                   agg_funcs=["vslast"],
                                                                                   remove_now=False)
    feature_factory_dict["user_id"]["StudyTermEncoder2"] = StudyTermEncoder2(is_partial_fit=True)
    feature_factory_dict["user_id"]["ElapsedTimeMeanByContentIdEncoder"] = ElapsedTimeMeanByContentIdEncoder()
    feature_factory_dict["user_id"]["CountEncoder"] = CountEncoder(column="user_id", is_partial_fit=True)
    feature_factory_dict[("user_id", "part")] = {
        "UserContentRateEncoder": UserContentRateEncoder(column=["user_id", "part"],
                                                         rate_func="elo")
    }
    for column in ["user_id",
                   "content_id",
                   "part",
                   ("user_id", "part")]:
        if column not in feature_factory_dict:
            feature_factory_dict[column] = {}
        if type(column) == str:
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                               agg_column="duration_previous_content_cap100k",
                                                                                                               remove_now=False)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=column,
                                                                                                agg_column="study_time",
                                                                                                remove_now=False)
        else:
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                               agg_column="duration_previous_content_cap100k",
                                                                                                               remove_now=False)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                agg_column="study_time",
                                                                                                remove_now=False)

    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[2, 5])
    feature_factory_dict["user_id"]["UserContentNowRateEncoder"] = UserContentNowRateEncoder(column="part",
                                                                                             target=[1, 2, 3, 4, 5, 6, 7],
                                                                                             rate_func="elo")
    feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                         column="content_id",
                                                                         is_debug=is_debug,
                                                                         model_id=model_id,
                                                                         n=300)
    feature_factory_dict["user_id"]["PreviousNAnsweredCorrectly"] = PreviousNAnsweredCorrectly(n=5,
                                                                                               is_partial_fit=True)

    feature_factory_dict[f"previous_5_ans"] = {
        "TargetEncoder": TargetEncoder(column="previous_5_ans")
    }
    feature_factory_dict["user_id"]["QuestionLectureTableEncoder2"] = QuestionLectureTableEncoder2(model_id=model_id,
                                                                                                   is_debug=is_debug,
                                                                                                   past_n=100,
                                                                                                   min_size=300)
    feature_factory_dict["user_id"]["QuestionQuestionTableEncoder2"] = QuestionQuestionTableEncoder2(model_id=model_id,
                                                                                                     is_debug=is_debug,
                                                                                                     past_n=100,
                                                                                                     min_size=300)
    feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(column="user_id",
                                                                                       rate_func="elo")
    feature_factory_dict["content_id"]["CorrectVsIncorrectMeanEncoderContent-Duration100k"] = \
        CorrectVsIncorrectMeanEncoder(groupby="content_id",
                                      column="duration_previous_content_cap100k",
                                      min_size=300)
    feature_factory_dict["content_id"]["CorrectVsIncorrectMeanEncoderContent-UserIdTargetEnc"] = \
        CorrectVsIncorrectMeanEncoder(groupby="part",
                                      column="target_enc_user_id",
                                      min_size=300)

    feature_factory_dict["user_id"]["PreviousContentAnswerTargetEncoder"] = PreviousContentAnswerTargetEncoder(min_size=300)
    feature_factory_dict["post"] = {
        "DurationFeaturePostProcess": DurationFeaturePostProcess()
    }
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num,
                                                    model_id=model_id,
                                                    load_feature=not is_debug,
                                                    save_feature=not is_debug)
    return feature_factory_manager

def prepare_df(fit=False):
    fname = "../input/riiid-test-answer-prediction/train_merged.pickle"

    df = pd.read_pickle(fname)

    if is_debug:
        df = pd.concat([df.head(500), df.tail(500)])

    model_id = "all"

    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=1, model_id=model_id)
    print("all_predict")
    df = feature_factory_manager.all_predict(df)

    if not fit:
        df = df.drop(["user_answer", "tags", "type_of", "bundle_id", "previous_5_ans",
                      "tag", "content_type_id"],
                     axis=1, errors="ignore")
        df = df[df["answered_correctly"].notnull()]
        df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in
                      df.columns]
    return df
# df["lec_count"] = df.groupby("user_id")["content_type_id"].cumsum().replace(0, np.nan)
# df["final_lec_idx"] = df.groupby(["user_id", "lec_count"]).cumcount()
# df = df.drop("lec_count", axis=1)

model_id = "all"

feature_factory_manager = make_feature_factory_manager(split_num=1)

model_id = "all"
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
df = prepare_df(fit=True)
model_id = "all"
feature_factory_manager.fit(df, is_first_fit=True)

size = 0
for k, v in feature_factory_manager.feature_factory_dict.items():
    for kk, vv in v.items():
        try:
            w_size = round(total_size(vv.data_dict) / 1_000_000, 2)
            print(f"{k}-{vv}: len={len(vv.data_dict)} size={w_size}MB")
            size += w_size
        except Exception as e:
            print(f"{k}-{kk} error")
            print(e)
print(f"-------------------")
print(f"total_size={size}MB")

for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = None
feature_factory_manager.logger = None
with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
    pickle.dump(feature_factory_manager, f)