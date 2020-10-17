
import unittest
import pandas as pd
import numpy as np
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    CountEncoder, \
    TargetEncoder, \
    MeanAggregator
from experiment.common import get_logger

class PartialAggregatorTestCase(unittest.TestCase):

    def test_fit_targetencoding(self):
        """
        countencoding, targetencodingのテスト
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "answered_correctly": [0, 1, 1, 0, 1, 1, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1"),
                "TargetEncoder": TargetEncoder(column="key1")}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "target_enc_key1": [np.nan, np.nan, 1, np.nan, 0, 0.5, 2/3]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        agger.fit(df)

        expect = {"a": 0,
                  "b": 1,
                  "c": 1/2}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["TargetEncoder"].data_dict)

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "target_enc_key1": [0, 1, 1, 1/2, np.nan]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial fit
        df_partial = pd.DataFrame({"key1": ["a", "b", "b", "d", "d"],
                                   "answered_correctly": [0, 0, 1, 1, 0]})

        agger.fit(df_partial)
        expect = {"a": 0,
                  "b": 3/4,
                  "c": 1/2,
                  "d": 1/2}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["TargetEncoder"].data_dict)

    def test_fit_meanaggregator_remove_now(self):
        """
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "data1": [0, 1, 1, 0, 1, 1, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1"),
                "MeanAggregator": MeanAggregator(column="key1", agg_column="data1", remove_now=True)}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "mean_data1_by_key1": [np.nan, np.nan, 1, np.nan, 0, 0.5, 2/3],
                                  "diff_mean_data1_by_key1": [np.nan, np.nan, 0, np.nan, 1, 0.5, -2/3]})
        df_expect["mean_data1_by_key1"] = df_expect["mean_data1_by_key1"].astype("float32")
        df_expect["diff_mean_data1_by_key1"] = df_expect["diff_mean_data1_by_key1"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        agger.fit(df)

        expect = {"a": 0,
                  "b": 1,
                  "c": 1/2}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["MeanAggregator"].data_dict)

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                "data1": [1, 1, 1, 1, 1]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "mean_data1_by_key1": [0, 1, 1, 1/2, np.nan],
                                  "diff_mean_data1_by_key1": [1, 0, 0, 1/2, np.nan]})
        df_expect["mean_data1_by_key1"] = df_expect["mean_data1_by_key1"].astype("float32")
        df_expect["diff_mean_data1_by_key1"] = df_expect["diff_mean_data1_by_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial fit
        df_partial = pd.DataFrame({"key1": ["a", "b", "b", "d", "d"],
                                   "data1": [0, 0, 1, 1, 0]})

        agger.fit(df_partial)
        expect = {"a": 0,
                  "b": 3/4,
                  "c": 1/2,
                  "d": 1/2}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["MeanAggregator"].data_dict)

    def test_fit_countencoding_split2(self):
        """
        countencoding, targetencodingのテスト
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "user_id": ["a", "b", "b", "c", "c", "c", "c"],
                           "answered_correctly": [0, 1, 1, 0, 1, 1, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1")
            },
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger,
                                      split_num=2)
        agger.fit(df)

        # partial_predict
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "user_id": ["a", "b", "b", "c", "c", "c", "c"],
                                  "count_enc_key1": [2, 4, 4, 8, 8, 8, 8],
                                  "count_enc_user_id": [1, 2, 2, 4, 4, 4, 4]})
        df_expect["count_enc_key1"] = df_expect["count_enc_key1"].astype("int32")
        df_expect["count_enc_user_id"] = df_expect["count_enc_user_id"].astype("int32")
        df_actual = agger.partial_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_fit_countencoding_multiple(self):
        """
        countencoding, targetencodingのテスト
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "key2": ["a", "a", "a", "a", "b", "b", "b"],
                           "answered_correctly": [0, 1, 1, 0, 1, 1, 0]})
        logger = get_logger()
        feature_factory_dict = {
            ("key1", "key2"): {
                "CountEncoder": CountEncoder(column=["key1", "key2"]),
                "TargetEncoder": TargetEncoder(column=["key1", "key2"])
            },
        }

        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # all_predict
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "key2": ["a", "a", "a", "a", "b", "b", "b"],
                                  "count_enc_['key1', 'key2']": [0, 0, 1, 0, 0, 1, 2],
                                  "target_enc_['key1', 'key2']": [np.nan, np.nan, 1, np.nan, np.nan, 1, 1]})
        df_expect["count_enc_['key1', 'key2']"] = df_expect["count_enc_['key1', 'key2']"].astype("int32")
        df_expect["target_enc_['key1', 'key2']"] = df_expect["target_enc_['key1', 'key2']"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial_predict
        agger.fit(df)

        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "key2": ["a", "a", "a", "a", "b", "b", "b"],
                                  "count_enc_['key1', 'key2']": [1, 2, 2, 1, 3, 3, 3],
                                  "target_enc_['key1', 'key2']": [0, 1, 1, 0, 2/3, 2/3, 2/3]})
        df_expect["count_enc_['key1', 'key2']"] = df_expect["count_enc_['key1', 'key2']"].astype("int32")
        df_expect["target_enc_['key1', 'key2']"] = df_expect["target_enc_['key1', 'key2']"].astype("float32")
        df_actual = agger.partial_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_fit_targetencoding_with_initialweight(self):
        """
        countencoding, targetencodingのテスト(initial_weight)
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "answered_correctly": [0, 1, 1, 0, 1, 1, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1"),
                "TargetEncoder": TargetEncoder(column="key1",
                                               initial_weight=10,
                                               initial_score=0.5)}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                                  "target_enc_key1": [np.nan, np.nan, 6/11, np.nan, 5/11, 6/12, 7/13]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        agger.fit(df)

        expect = {"a": 5/11,
                  "b": 7/12,
                  "c": 7/14}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["TargetEncoder"].data_dict)

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "target_enc_key1": [5/11, 7/12, 7/12, 7/14, np.nan]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial fit
        df_partial = pd.DataFrame({"key1": ["a", "b", "b", "d", "d"],
                                   "answered_correctly": [0, 0, 1, 1, 0]})

        agger.fit(df_partial)
        expect = {"a": 5/12,
                  "b": 8/14,
                  "c": 7/14,
                  "d": 1/2}

        self.assertEqual(expect, agger.feature_factory_dict["key1"]["TargetEncoder"].data_dict)


if __name__ == "__main__":
    unittest.main()