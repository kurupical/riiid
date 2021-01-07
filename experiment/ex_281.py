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
    DurationPreviousContent, \
    StudyTermEncoder2, \
    ElapsedTimeMeanByContentIdEncoder

from experiment.common import get_logger, total_size
import pandas as pd
from model.lgbm import train_lgbm_cv, train_lgbm_cv_newuser, train_lgbm_cv_2500kval
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

    feature_factory_dict = {"user_id": {}}

    feature_factory_dict["user_id"]["DurationPreviousContent"] = DurationPreviousContent(is_partial_fit=True)
    feature_factory_dict["user_id"]["UserContentRateEncoder"] = UserContentRateEncoder(rate_func="elo",
                                                                                       column="user_id")
    feature_factory_dict["user_id"]["PreviousAnswer2"] = PreviousAnswer2(groupby="user_id",
                                                                         column="question_id",
                                                                         is_debug=is_debug,
                                                                         model_id=model_id,
                                                                         n=300)
    feature_factory_dict["user_id"]["StudyTermEncoder2"] = StudyTermEncoder2(is_partial_fit=True)
    feature_factory_dict["user_id"][f"MeanAggregatorStudyTimebyUserId"] = MeanAggregator(column="user_id",
                                                                                         agg_column="study_time",
                                                                                         remove_now=False)

    feature_factory_dict["user_id"]["ElapsedTimeMeanByContentIdEncoder"] = ElapsedTimeMeanByContentIdEncoder()

    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=split_num,
                                                    model_id=model_id,
                                                    load_feature=not is_debug,
                                                    save_feature=not is_debug)
    return feature_factory_manager

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
print("all_predict")
print(df.shape)
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

df = df.drop(["user_answer", "tags", "type_of", "bundle_id"], axis=1)
df = df[df["answered_correctly"].notnull()]

# df["lec_count"] = df.groupby("user_id")["content_type_id"].cumsum().replace(0, np.nan)
# df["final_lec_idx"] = df.groupby(["user_id", "lec_count"]).cumcount()
# df = df.drop("lec_count", axis=1)

params = {
    'objective': 'binary',
    'num_leaves': 96,
    'max_depth': -1,
    'learning_rate': 0.3,
    'boosting': 'gbdt',
    'bagging_fraction': 0.5,
    'feature_fraction': 0.7,
    'bagging_seed': 0,
    'reg_alpha': 200,  # 1.728910519108444,
    'reg_lambda': 200,
    'random_state': 0,
    'metric': 'auc',
    'verbosity': -1,
    "n_estimators": 10000,
    "early_stopping_rounds": 50
}
df.tail(1000).to_csv("exp028.csv", index=False)

df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df.columns]
df = df[df["answered_correctly"].notnull()]
print(df.columns)
print(df.shape)

categorical_feature = ["content_id"]
print(model_id)
train_lgbm_cv_2500kval(df,
                      categorical_feature=categorical_feature,
                      params=params,
                      output_dir=output_dir,
                      model_id=model_id,
                      exp_name=model_id,
                      is_debug=is_debug,
                      drop_user_id=True)
