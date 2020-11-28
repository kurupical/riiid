from datetime import datetime as dt
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    TargetEncoder, \
    CountEncoder, \
    MeanAggregator, \
    TagsSeparator, \
    UserLevelEncoder2, \
    NUniqueEncoder, \
    ShiftDiffEncoder, \
    PartSeparator, \
    UserCountBinningEncoder, \
    CategoryLevelEncoder, \
    PriorQuestionElapsedTimeBinningEncoder
import pandas as pd
import glob
import os
import tqdm
import lightgbm as lgb
import pickle
import riiideducation
import numpy as np
from logging import Logger, StreamHandler, Formatter
import shutil
import time
import warnings
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import sys
warnings.filterwarnings("ignore")

model_dir = "../output/ex_111/20201123131530"

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
prior_columns = ["prior_group_responses", "prior_group_answers_correct"]

def get_logger():
    formatter = Formatter("%(asctime)s|%(levelname)s| %(message)s")
    logger = Logger(name="log")
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

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


# model loading
models = []
for model_path in glob.glob(f"{model_dir}/*model*.pickle"):
    with open(model_path, "rb") as f:
        models.append(pickle.load(f))

# load feature_factory_manager
logger = get_logger()
ff_manager_path = f"{model_dir}/feature_factory_manager.pickle"
with open(ff_manager_path, "rb") as f:
    feature_factory_manager = pickle.load(f)
for dicts in feature_factory_manager.feature_factory_dict.values():
    for factory in dicts.values():
        factory.logger = logger

for k, v in feature_factory_manager.feature_factory_dict.items():
    for kk, vv in v.items():
        try:
            print(f"{k}-{vv}: len={len(vv.data_dict)} size={round(total_size(vv.data_dict) / 1_000_000, 2)}MB")
        except:
            print(f"{k}-{kk} error")
for i, model in enumerate(models):
    print(f"lgbm_model{i}: {sys.getsizeof(model)}")