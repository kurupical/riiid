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
    QuestionQuestionTableEncoder, \
    UserAnswerLevelEncoder, \
    WeightDecayTargetEncoder, \
    UserContentRateEncoder, \
    StudyTermEncoder, \
    Word2VecEncoder

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

def make_feature_factory_manager(split_num,
                                 size,
                                 window,
                                 model_id=None):
    logger = get_logger()

    feature_factory_dict = {}
    feature_factory_dict["tags"] = {
        "TagsSeparator": TagsSeparator(is_partial_fit=True)
    }

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
    feature_factory_dict["user_id"]["StudyTermEncoder"] = StudyTermEncoder(is_partial_fit=True)
    # feature_factory_dict["user_id"]["UserLevelEncoder2ContentId"] = UserLevelEncoder2(vs_column="content_id")
    # feature_factory_dict["content_id"]["ContentLevelEncoder2UserId"] = ContentLevelEncoder(vs_column="user_id", is_partial_fit=True)
    # feature_factory_dict["user_id"]["MeanAggregatorContentLevel"] = MeanAggregator(column="user_id",
    #                                                                                agg_column="content_level_user_id",
    #                                                                                remove_now=False)
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
            feature_factory_dict[column][f"MeanAggregatorStudyTimeby{column}"] = MeanAggregator(column=column,
                                                                                                agg_column="study_time",
                                                                                                remove_now=True)
        else:
            feature_factory_dict[column][f"MeanAggregatorPriorQuestionElapsedTimeby{column}"] = MeanAggregator(column=list(column),
                                                                                                               agg_column="prior_question_elapsed_time",
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

    feature_factory_dict[f"previous_3_ans"] = {
        "TargetEncoder": TargetEncoder(column="previous_3_ans")
    }
    feature_factory_dict["user_id"]["QuestionLectureTableEncoder2"] = QuestionLectureTableEncoder2(model_id=model_id,
                                                                                                   is_debug=is_debug,
                                                                                                   past_n=100,
                                                                                                   min_size=100)
    feature_factory_dict["user_id"]["QuestionQuestionTableEncoder"] = QuestionQuestionTableEncoder(model_id=model_id,
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

for window in [20, 50, 100]:
    for size in [3, 5, 10]:
        for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
            print(fname)
            print(f"w2v_window{window}_size{size}")
            if is_debug:
                df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
            else:
                df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            model_id = os.path.basename(fname).replace(".pickle", "")
            df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
            df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
            feature_factory_manager = make_feature_factory_manager(split_num=10,
                                                                   model_id=model_id,
                                                                   window=window,
                                                                   size=size)
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

            categorical_feature = ["tags1", "tags2", "content_id"]
            print(df.shape)
            print(model_id)
            train_lgbm_cv_newuser(df,
                                  categorical_feature=categorical_feature,
                                  params=params,
                                  output_dir=output_dir,
                                  model_id=model_id,
                                  exp_name=f"{model_id}_w2v_window{window}_size{size}",
                                  is_debug=is_debug,
                                  drop_user_id=True)
            break