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
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from model.transformer import train_transformer


warnings.filterwarnings("ignore")
pd.set_option("max_rows", 100)

output_dir = f"../output/{os.path.basename(__file__).replace('.py', '')}/{dt.now().strftime('%Y%m%d%H%M%S')}/"
os.makedirs(output_dir, exist_ok=True)

is_debug = False
wait_time = 0
if not is_debug:
    for _ in tqdm.tqdm(range(wait_time)):
        time.sleep(1)
logger = get_logger()
for fname in glob.glob("../input/riiid-test-answer-prediction/split10/*"):
    print(fname)
    if is_debug:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).head(1000).reset_index(drop=True)
    else:
        df = pd.read_pickle(fname).sort_values(["user_id", "timestamp"]).reset_index(drop=True).head(100000)
    model_id = os.path.basename(fname).replace(".pickle", "")
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")
    df["content_id2"] = df["content_id"].astype("int32") + (16000 * df["content_type_id"]).astype("int32")
    use_cols_config = {
        "content_id2": {"num_embeddings": 50000},
        "part": {"num_embeddings": 8},
        "answered_correctly": {"num_embeddings": 3},
    }
    train_transformer(df=df,
                      use_cols_config=use_cols_config,
                      window=50,
                      criterion=torch.nn.BCEWithLogitsLoss(),
                      optimizer=AdamW,
                      scheduler=get_linear_schedule_with_warmup,
                      scheduler_params={"num_warmup_steps": 30,
                                        "num_training_epochs": 4},
                      optimizer_params={"lr": 1e-4},
                      n_emb=32,
                      n_head=4,
                      n_hidden=64,
                      n_layers=2,
                      batch_size=256,
                      epochs=12,
                      dropout=0.3,
                      logger=logger,
                      output_dir=output_dir,
                      model_id=model_id)
    1/0