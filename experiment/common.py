from logging import Logger, StreamHandler, Formatter
import pandas as pd
import numpy as np
from sys import getsizeof, stderr
from itertools import chain
from collections import deque


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


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)