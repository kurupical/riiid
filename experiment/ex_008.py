from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator
from experiment.common import get_logger
import pandas as pd
from model.lgbm import train_lgbm_cv
from sklearn.model_selection import KFold
from datetime import datetime as dt
import os
import glob

output_dir = f"../output/ex_008/{dt.now().strftime('%Y%m%d%H%M%S')}/"

for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    df = pd.read_pickle(fname).head(1000)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    logger = get_logger()
    feature_factory_dict = {}
    for column in ["user_id", "content_id", "content_type_id", "prior_question_had_explanation"]:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=column),
            "TargetEncoder": TargetEncoder(column=column)
        }
    feature_factory_dict["user_id"]["MeanAggregatorTimestamp"] = MeanAggregator(column="user_id",
                                                                                agg_column="timestamp",
                                                                                remove_now=False)
    feature_factory_dict["user_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="user_id",
                                                                                               agg_column="prior_question_elapsed_time",
                                                                                               remove_now=True)
    feature_factory_dict["content_id"]["MeanAggregatorPriorQuestionElapsedTime"] = MeanAggregator(column="content_id",
                                                                                                  agg_column="prior_question_elapsed_time",
                                                                                                  remove_now=True)
    for column in [("user_id", "content_type_id"), ("user_id", "prior_question_had_explanation")]:
        feature_factory_dict[column] = {
            "CountEncoder": CountEncoder(column=list(column)),
            "TargetEncoder": TargetEncoder(column=list(column))
        }
    feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                    logger=logger,
                                                    split_num=10)
    df = feature_factory_manager.all_predict(df)
    os.makedirs(output_dir, exist_ok=True)
    params = {
        'objective': 'binary',
        'num_leaves': 32,
        'min_data_in_leaf': 15,  # 42,
        'max_depth': -1,
        'learning_rate': 0.1,
        'boosting': 'gbdt',
        'bagging_fraction': 0.7,  # 0.5,
        'feature_fraction': 0.5,
        'bagging_seed': 0,
        'reg_alpha': 0.1,  # 1.728910519108444,
        'reg_lambda': 1,
        'random_state': 0,
        'metric': 'auc',
        'verbosity': -1,
        "n_estimators": 10000,
        "early_stopping_rounds": 100
    }

    df = df.drop(["user_answer"], axis=1)
    print(df.columns)

    model_id = os.path.basename(fname).replace(".pickle", "")
    print(model_id)
    train_lgbm_cv(df,
                  params=params,
                  output_dir=output_dir,
                  model_id=model_id,
                  exp_name=f"exp008_{model_id}",
                  drop_user_id=True)
    break
