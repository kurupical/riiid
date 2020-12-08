import sys
sys.path.append("../")

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
    PreviousContentAnswerTargetEncoder

from experiment.common import get_logger, total_size
import pandas as pd
from model.lgbm import train_lgbm_cv_newuser_alldata
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
                                                                                   past_ns=[5, 20],
                                                                                   agg_funcs=["vslast"],
                                                                                   remove_now=False)
    feature_factory_dict["user_id"]["Past1ContentTypeId"] = PastNFeatureEncoder(column="content_type_id",
                                                                                past_ns=[5, 15],
                                                                                agg_funcs=["mean"],
                                                                                remove_now=False)
    feature_factory_dict["user_id"]["StudyTermEncoder"] = StudyTermEncoder(is_partial_fit=True)
    feature_factory_dict["user_id"]["ElapsedTimeVsShiftDiffEncoder"] = ElapsedTimeVsShiftDiffEncoder()
    feature_factory_dict["user_id"]["CountEncoder"] = CountEncoder(column="user_id", is_partial_fit=True)
    feature_factory_dict["user_id"]["UserAnswerLevelEncoder"] = UserAnswerLevelEncoder(past_n=50)
    feature_factory_dict[("user_id", "part")] = {
        "UserContentRateEncoder": UserContentRateEncoder(column=["user_id", "part"],
                                                         rate_func="elo")
    }

    for column in ["user_id", "content_id", "part", ("user_id", "part")]:
        if column not in feature_factory_dict:
            feature_factory_dict[column] = {}
        if type(column) == str:
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=column,
                                                                                                               agg_column="shiftdiff_timestamp_by_user_id_cap200k",
                                                                                                               remove_now=False)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=column,
                                                                                                agg_column="study_time",
                                                                                                remove_now=False)
        else:
            feature_factory_dict[column][f"MeanAggregatorShiftDiffTimeElapsedTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                               agg_column="shiftdiff_timestamp_by_user_id_cap200k",
                                                                                                               remove_now=False)
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                agg_column="study_time",
                                                                                                remove_now=False)


    feature_factory_dict["user_id"]["CategoryLevelEncoderPart"] = CategoryLevelEncoder(groupby_column="user_id",
                                                                                       agg_column="part",
                                                                                       categories=[2, 5])

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
    feature_factory_dict["user_id"]["QuestionQuestionTableEncoder3"] = QuestionQuestionTableEncoder2(model_id=model_id,
                                                                                                     is_debug=is_debug,
                                                                                                     past_n=100,
                                                                                                     min_size=300)
    feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(column="user_id",
                                                                                       rate_func="elo")
    feature_factory_dict["user_id"]["PreviousContentAnswerTargetEncoder"] = PreviousContentAnswerTargetEncoder(min_size=300)
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


filelist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

df = pd.concat(
    [pd.read_pickle(f"../input/riiid-test-answer-prediction/split10/train_{x}.pickle") for x in filelist])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
if is_debug:
    df = df.head(1000)

model_id = "_".join([str(x) for x in filelist])
df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
feature_factory_manager = make_feature_factory_manager(split_num=2, model_id=model_id)
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
df["user_id_div10"] = df["user_id"] % 10
df = df[df["user_id_div10"].isin(filelist)]
df = df.drop("user_id_div10", axis=1)

df.tail(1000).to_csv("exp028.csv", index=False)

df = df.drop(["user_answer", "tags", "type_of", "bundle_id", "previous_5_ans"], axis=1)
df = df[df["answered_correctly"].notnull()]

target_col = [
'rating_diff_content_user_id',
'content_id',
'rating_diff_content___user_id____part__',
'qq_table2_mean',
'previous_answer_index_content_id',
'te_mean',
'qq_table2_min',
'diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by___user_id____part__',
'__user_id____part___rating',
'previous_answer_content_id',
'diff_shiftdiff_elapsed_time',
'diff_mean_study_time_by___user_id____part__',
'qq_table2_max',
'diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by_part',
'diff_mean_study_time_by_part',
'user_id_rating',
'ql_table2_min',
'user_rate_mean_part_5',
'past5_timestamp_vslast',
'qq_table2_last',
'diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by_content_id',
'shiftdiff_timestamp_by_user_id',
'shiftdiff_timestamp_by_user_id_cap200k',
'ql_table2_max',
'content_rating',
'diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by_user_id',
'user_rate_mean_part_2',
'ql_table2_last',
'target_enc_user_id',
'past20_timestamp_vslast',
'te_min',
'question_id',
'mean_study_time_by_content_id',
'part',
'study_time',
'mean_study_time_by_user_id',
'diff_mean_study_time_by_content_id',
'prior_question_elapsed_time',
'mean_study_time_by_part',
'target_enc_previous_5_ans',
"answered_correctly",
"user_id"
]
df = df[target_col]
print(df.columns)
print(df.shape)
df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df.columns]
print(model_id)
model_name = "_".join([str(x) for x in filelist])

df["answered_correctly"] = df["answered_correctly"].astype("float16")
df["prior_question_elapsed_time"] = df["prior_question_elapsed_time"].astype("float32")

train_lgbm_cv_newuser_alldata(df,
                              params=params,
                              output_dir=output_dir,
                              model_id=model_name,
                              exp_name=model_name,
                              is_debug=is_debug,
                              drop_user_id=True)

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
for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = None
feature_factory_manager.logger = None
with open(f"{output_dir}/feature_factory_manager.pickle", "wb") as f:
    pickle.dump(feature_factory_manager, f)
