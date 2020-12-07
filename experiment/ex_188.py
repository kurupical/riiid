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

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"

is_debug = False
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)

def calc_optimized_weight(df):
    best_score = 0
    best_cat_ratio = 0
    for cat_ratio in np.arange(0, 1.05, 0.05):
        pred = df["nn"] * cat_ratio + df["lgbm"] * (1 - cat_ratio)
        score = roc_auc_score(df["target"].values, pred)
        print("[nn_ratio: {:.2f}] AUC: {:.4f}".format(cat_ratio, score))
        if score > best_score:
            best_score = score
            best_cat_ratio = cat_ratio

    return best_score, best_cat_ratio

def get_model(input_len,
              reg,
              hidden1,
              hidden2,
              ):
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(hidden1,
                    kernel_regularizer=regularizers.l2(reg),
                    activity_regularizer=regularizers.l2(reg),
                    input_shape=(input_len,)))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(hidden2,
                    kernel_regularizer=regularizers.l2(reg),
                    activity_regularizer=regularizers.l2(reg)))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(),
                  metrics=tensorflow.keras.metrics.AUC())

    return model

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
    feature_factory_dict["user_id"]["QuestionQuestionTableEncoder2"] = QuestionQuestionTableEncoder2(model_id=model_id,
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

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.concat([pd.read_pickle(fname).head(500), pd.read_pickle(fname).tail(500)])
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"])
    model_id = os.path.basename(fname).replace(".pickle", "")
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    feature_factory_manager = make_feature_factory_manager(split_num=10, model_id=model_id)
    df = feature_factory_manager.all_predict(df)
    os.makedirs(output_dir, exist_ok=True)

    df.tail(1000).to_csv("exp028.csv", index=False)
    df = df[df["answered_correctly"].notnull()]
    df = df.fillna(-1).replace(-99, -1)

    df.columns = [x.replace("[", "_").replace("]", "_").replace("'", "_").replace(" ", "_").replace(",", "_") for x in df.columns]
    """
    df = df[["user_id", "answered_correctly", "rating_diff_content_user_id", "qq_table2_last",
             "rating_diff_content___user_id____part__", "user_id_rating", "te_max", "te_mean",
             "previous_answer_index_content_id", "diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by___user_id____part__",
             "diff_mean_shiftdiff_timestamp_by_user_id_cap200k_by_part", "diff_mean_study_time_by___user_id____part__",
             "past5_timestamp_vslast", ]]
    """
    print(df.columns)
    print(df.shape)
    df = df.drop(["user_answer", "tags", "type_of", "bundle_id", "previous_5_ans"], axis=1)

    params = {
        "input_len": len(df.columns) - 2, # 2: answered_correctly, user_id
        "hidden1": 256,
        "hidden2": 128,
        "reg": 1e-5
    }
    model = get_model(**params)
    print(params)

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
                experiment_id=3)

    df_oof_lgbm = pd.read_csv("../output/ex_172/20201202080625/oof_train_0_lgbm.csv")
    df_oof_nn = pd.read_csv(f"{output_dir}/oof_{model_id}_nn.csv")

    df_oof = pd.DataFrame()
    df_oof["target"] = df_oof_lgbm["target"]
    df_oof["lgbm"] = df_oof_lgbm["predict"]
    df_oof["nn"] = df_oof_nn["predict"]

    score, weight = calc_optimized_weight(df_oof)
