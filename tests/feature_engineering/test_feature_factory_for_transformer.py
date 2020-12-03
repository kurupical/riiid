
import unittest
import pandas as pd
import numpy as np
import pickle
import os
from feature_engineering.feature_factory_for_transformer import FeatureFactoryForTransformer
from experiment.common import get_logger

class PartialAggregatorTestCase(unittest.TestCase):

    def test_make_dict_notnull(self):

        agger = FeatureFactoryForTransformer(column_config={"content_id": {"type": "category"}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)
        df = pd.DataFrame({"content_id": [1, 2, 4, 5, 7, 1]})

        pickle_dir = "./test.pickle"

        agger.make_dict(df=df,
                        column="content_id",
                        output_dir=pickle_dir)

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        expect = {1: 1,
                  2: 2,
                  4: 3,
                  5: 4,
                  7: 5}

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)

    def test_make_dict_havenull(self):

        agger = FeatureFactoryForTransformer(column_config={"part": {"type": "category"}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)

        df = pd.DataFrame({"part": [1, 1, 2, 2, 2, np.nan]})

        pickle_dir = "./test.pickle"

        agger.make_dict(df=df,
                        column="part",
                        output_dir=pickle_dir)

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        expect = {-1: 1,
                  1: 2,
                  2: 3}

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)


    def test_make_dict_multikey(self):

        agger = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category"}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)
        df = pd.DataFrame({"content_id": [1, 2, 4, 5, 7, 1, 2],
                           "content_type_id": [0, 0, 0, 0, 0, 1, 0]})

        pickle_dir = "./test.pickle"

        agger.make_dict(df=df,
                        column=("content_id", "content_type_id"),
                        output_dir=pickle_dir)

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        expect = {(1, 0): 1,
                  (1, 1): 2,
                  (2, 0): 3,
                  (4, 0): 4,
                  (5, 0): 5,
                  (7, 0): 6}

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)

    def test_normal(self):
        logger = get_logger()

        embbed_dict = {
            "content_id": {1: 1, 2: 2, 4: 3, 5: 4, 7: 5},
            "part": {1: 1, 3: 2}
        }
        agger = FeatureFactoryForTransformer(column_config={"content_id": {"type": "category"},
                                                            "part": {"type": "category"}},
                                             dict_path=None,
                                             embbed_dict=embbed_dict,
                                             sequence_length=2,
                                             logger=logger)
        df = pd.DataFrame({"user_id": [1, 1, 1, 1, 2, 2, 3],
                           "content_id": [1, 2, 4, 5, 7, 1, 2],
                           "answered_correctly": [1, 1, 1, 0, 1, 1, 1],
                           "part": [1, 1, 3, 1, 1, 3, 1],
                           "is_val": [0, 0, 0, 0, 0, 1, 1]})

        actual = agger.all_predict(df)
        expect = {
            1: {"content_id": [1, 2, 3, 4],
                "part": [1, 1, 2, 1],
                "answered_correctly": [1, 1, 1, 0],
                "is_val": [0, 0, 0, 0]},
            2: {"content_id": [5, 1],
                "part": [1, 2],
                "answered_correctly": [1, 1],
                "is_val": [0, 1]},
            3: {"content_id": [2],
                "part": [1],
                "answered_correctly": [1],
                "is_val": [1]}
        }

        self.assertEqual(expect, actual)
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3, 4],
                           "content_id": [1, 1, 1, 2],
                           "part": [3, 3, 3, 3]})

        actual = agger.partial_predict(df)
        expect = {
            1: {"content_id": [4, 1],
                "part": [1, 2],
                "answered_correctly": [0, -1]},
            2: {"content_id": [1, 1],
                "part": [2, 2],
                "answered_correctly": [1, -1]},
            3: {"content_id": [2, 1],
                "part": [1, 2],
                "answered_correctly": [1, -1]},
            4: {"content_id": [2],
                "part": [2],
                "answered_correctly": [-1]}
        }
        self.assertEqual(expect, actual)

