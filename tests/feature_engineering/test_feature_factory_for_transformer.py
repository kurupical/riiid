
import unittest
import pandas as pd
import numpy as np
import pickle
import os
from feature_engineering.feature_factory_for_transformer import FeatureFactoryForTransformer
from feature_engineering.feature_factory_for_transformer import calc_sec
from experiment.common import get_logger


def assert_equal_numpy_dict(expect, actual):
    for k1, v1 in expect.items():
        for k2, v2 in v1.items():
            assert np.array_equal(v2, actual[k1][k2])

    for k1, v1 in actual.items():
        for k2, v2 in v1.items():
            assert np.array_equal(v2, expect[k1][k2])

class PartialAggregatorTestCase(unittest.TestCase):

    def test_make_dict_notnull(self):

        agger = FeatureFactoryForTransformer(column_config={"content_id": {"type": "category", "dtype": np.int8}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)
        df = pd.DataFrame({"content_id": [1, 2, 4, 5, 7, 1]})

        pickle_dir = "./test.pickle"

        agger._make_dict(df=df,
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

        agger = FeatureFactoryForTransformer(column_config={"part": {"type": "category", "dtype": np.int8}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)

        df = pd.DataFrame({"part": [1, 1, 2, 2, 2, np.nan]})

        pickle_dir = "./test.pickle"

        agger._make_dict(df=df,
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

        agger = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category", "dtype": np.int8}},
                                             dict_path=None,
                                             sequence_length=10,
                                             logger=None)
        df = pd.DataFrame({"content_id": [1, 2, 4, 5, 7, 1, 2],
                           "content_type_id": [0, 0, 0, 0, 0, 1, 0]})

        pickle_dir = "./test.pickle"

        agger._make_dict(df=df,
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
        agger = FeatureFactoryForTransformer(column_config={"content_id": {"type": "category", "dtype": np.int8},
                                                            "part": {"type": "category", "dtype": np.int8},
                                                            "answered_correctly": {"type": "leakage_feature", "dtype": np.int8}},
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
            1: {"content_id": np.array([1, 2, 3, 4]),
                "part": np.array([1, 1, 2, 1]),
                "answered_correctly": np.array([1, 1, 1, 0]),
                "is_val": np.array([0, 0, 0, 0])},
            2: {"content_id": np.array([5, 1]),
                "part": np.array([1, 2]),
                "answered_correctly": np.array([1, 1]),
                "is_val": np.array([0, 1])},
            3: {"content_id": np.array([2]),
                "part": np.array([1]),
                "answered_correctly": np.array([1]),
                "is_val": np.array([1])}
        }

        assert_equal_numpy_dict(expect, actual)

        # fit/partial_predict <1>
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3, 4],
                           "content_id": [1, 1, 1, 2],
                           "part": [3, 3, 3, 3],
                           "answered_correctly": [0, 0, 0, 0]})

        actual = agger.partial_predict(df)
        expect = {
            0: {"content_id": np.array([4, 1]),
                "part": np.array([1, 2]),
                "answered_correctly": np.array([0, -1])},
            1: {"content_id": np.array([1, 1]),
                "part": np.array([2, 2]),
                "answered_correctly": np.array([1, -1])},
            2: {"content_id": np.array([2, 1]),
                "part": np.array([1, 2]),
                "answered_correctly": np.array([1, -1])},
            3: {"content_id": np.array([2]),
                "part": np.array([2]),
                "answered_correctly": np.array([-1])}
        }
        assert_equal_numpy_dict(expect, actual)

        # fit/partial_predict <2>
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [2, 3],
                           "content_id": [2, 4],
                           "part": [1, 1]})

        actual = agger.partial_predict(df)
        expect = {
            0: {"content_id": np.array([1, 2]),
                "part": np.array([2, 1]),
                "answered_correctly": np.array([0, -1])},
            1: {"content_id": np.array([1, 3]),
                "part": np.array([2, 1]),
                "answered_correctly": np.array([0, -1])}
        }
        assert_equal_numpy_dict(expect, actual)


    def test_normal_multikey(self):
        logger = get_logger()

        embbed_dict = {
            ("content_id", "content_type_id"): {(1, 1): 1, (1, 2): 2, (2, 1): 3, (2, 2): 4},
        }
        agger = FeatureFactoryForTransformer(column_config={("content_id", "content_type_id"): {"type": "category", "dtype": np.int8},
                                                            "answered_correctly": {"type": "leakage_feature", "dtype": np.int8}},
                                             dict_path=None,
                                             embbed_dict=embbed_dict,
                                             sequence_length=2,
                                             logger=logger)
        df = pd.DataFrame({"user_id": [1, 1, 2, 2],
                           "content_id": [1, 1, 2, 2],
                           "answered_correctly": [1, 1, 1, 0],
                           "content_type_id": [1, 2, 1, 2],
                           "is_val": [0, 0, 0, 1]})

        actual = agger.all_predict(df)
        expect = {
            1: {("content_id", "content_type_id"): np.array([1, 2]),
                "answered_correctly": np.array([1, 1]),
                "is_val": np.array([0, 0])},
            2: {("content_id", "content_type_id"): np.array([3, 4]),
                "answered_correctly": np.array([1, 0]),
                "is_val": np.array([0, 1])},
        }

        assert_equal_numpy_dict(expect, actual)
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3],
                           "content_type_id": [1, 1, 1],
                           "content_id": [1, 1, 2]})

        actual = agger.partial_predict(df)
        expect = {
            0: {("content_id", "content_type_id"): np.array([2, 1]),
                "answered_correctly": np.array([1, -1])},
            1: {("content_id", "content_type_id"): np.array([4, 1]),
                "answered_correctly": np.array([0, -1])},
            2: {("content_id", "content_type_id"): np.array([3]),
                "answered_correctly": np.array([-1])},
        }
        assert_equal_numpy_dict(expect, actual)

    def test_normal_numeric(self):
        logger = get_logger()

        embbed_dict = {
            "content_id": {1: 1, 2: 2, 4: 3, 5: 4, 7: 5},
            "part": {1: 1, 3: 2}
        }
        agger = FeatureFactoryForTransformer(column_config={"content_id": {"type": "category", "dtype": np.int8},
                                                            "part": {"type": "numeric", "dtype": np.int8},
                                                            "answered_correctly": {"type": "leakage_feature", "dtype": np.int8}},
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
            1: {"content_id": np.array([1, 2, 3, 4]),
                "part": np.array([1, 1, 3, 1]),
                "answered_correctly": np.array([1, 1, 1, 0]),
                "is_val": np.array([0, 0, 0, 0])},
            2: {"content_id": np.array([5, 1]),
                "part": np.array([1, 3]),
                "answered_correctly": np.array([1, 1]),
                "is_val": np.array([0, 1])},
            3: {"content_id": np.array([2]),
                "part": np.array([1]),
                "answered_correctly": np.array([1]),
                "is_val": np.array([1])}
        }

        assert_equal_numpy_dict(expect, actual)

        # fit/partial_predict <1>
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3, 4],
                           "content_id": [1, 1, 1, 2],
                           "part": [3, 3, 3, 3],
                           "answered_correctly": [0, 0, 0, 0]})

        actual = agger.partial_predict(df)
        expect = {
            0: {"content_id": np.array([4, 1]),
                "part": np.array([1, 3]),
                "answered_correctly": np.array([0, -1])},
            1: {"content_id": np.array([1, 1]),
                "part": np.array([3, 3]),
                "answered_correctly": np.array([1, -1])},
            2: {"content_id": np.array([2, 1]),
                "part": np.array([1, 3]),
                "answered_correctly": np.array([1, -1])},
            3: {"content_id": np.array([2]),
                "part": np.array([3]),
                "answered_correctly": np.array([-1])}
        }
        assert_equal_numpy_dict(expect, actual)

        # fit/partial_predict <2>
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [2, 3],
                           "content_id": [2, 4],
                           "part": [1, 1]})

        actual = agger.partial_predict(df)
        expect = {
            0: {"content_id": np.array([1, 2]),
                "part": np.array([3, 1]),
                "answered_correctly": np.array([0, -1])},
            1: {"content_id": np.array([1, 3]),
                "part": np.array([3, 1]),
                "answered_correctly": np.array([0, -1])}
        }
        assert_equal_numpy_dict(expect, actual)

