from logging import Logger, StreamHandler, Formatter
import pandas as pd
import numpy as np

def get_logger():
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def merge(df: pd.DataFrame,
          df_question: pd.DataFrame,
          df_lecture: pd.DataFrame):

    w_df1 = pd.merge(df[df["content_type_id"] == 0], df_question, how="left", left_on="content_id",
                     right_on="question_id")
    w_df2 = pd.merge(df[df["content_type_id"] == 1], df_lecture, how="left", left_on="content_id",
                     right_on="lecture_id")
    df = pd.concat([w_df1, w_df2]).sort_values(["user_id", "timestamp"])
    df["tag"] = df["tag"].fillna(-1).astype("int16")
    df["correct_answer"] = df["correct_answer"].fillna(-1).astype("int8")
    df["bundle_id"] = df["bundle_id"].fillna(-1).astype("int32")
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")
    df = df.sort_values(["user_id", "timestamp"])
    return df

def read_data(f: str):
    df = pd.read_pickle(f)
    df["answered_correctly"] = df["answered_correctly"].replace(-1, np.nan)
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].fillna(-1).astype("int8")

    return df
