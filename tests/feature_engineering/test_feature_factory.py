
import unittest
import pandas as pd
import numpy as np
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    CountEncoder, \
    TargetEncoder, \
    MeanAggregator, \
    UserLevelEncoder
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
                                  "target_enc_key1": [0.5, 0.5, 6/11, 0.5, 5/11, 6/12, 7/13]})
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
                                  "target_enc_key1": [5/11, 7/12, 7/12, 7/14, 0.5]})
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


    def test_fit_user_level(self):
        """
        user_level
        :return:
        """
        logger = get_logger()
        feature_factory_dict = {
            "content_id": {
                "CountEncoder": CountEncoder(column="content_id"),
                "TargetEncoder": TargetEncoder(column="content_id",
                                               initial_weight=10,
                                               initial_score=0.5)},
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id"),
                "TargetEncoder": TargetEncoder(column="user_id",
                                               initial_weight=10,
                                               initial_score=0.5),
                "UserLevelEncoder": UserLevelEncoder(initial_weight=10,
                                                     initial_score=0.5)}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": ["a", "b", "b", "c", "c"],
                           "content_id": ["x", "x", "x", "y", "y"],
                           "answered_correctly": [0, 0, 1, 0, 1]})

        # predict_all
        # <default> target_enc_content_id mean([0.5] * 10) = 0.5 なので, user_levelは0.5
        # user_id=a, content_id=x, answered_correctly=1が入ると
        # xのtarget_enc_content_id = mean([0.5] * 10 + [1]) = 6/11.
        # aのuser_level = mean([0.5]*10 + 6/11) に更新される

        df_expect = pd.DataFrame({"user_id": ["a", "b", "b", "c", "c"],
                                  "content_id": ["x", "x", "x", "y", "y"],
                                  "user_level": [0.5,
                                                 np.array([0.5]*10 + [5/11]).astype("float32").mean(),
                                                 np.array([0.5]*10 + [5/11] + [5/12]).astype("float32").mean(),
                                                 0.5,
                                                 np.array([0.5]*10 + [5/10] + [5/11]).astype("float32").mean()]})
        df_expect["user_level"] = df_expect["user_level"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.set_option("max_columns", 100)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        df = pd.DataFrame({"user_id": ["a", "b", "b", "c", "c"],
                           "content_id": ["x", "x", "x", "y", "y"],
                           "answered_correctly": [0, 0, 1, 0, 1]})

        agger.fit(df)
        a = np.array([0.5]*10 + [6/13]).mean()
        b = np.array([0.5]*10 + [6/13] + [6/13]).mean()
        c = np.array([0.5]*10 + [6/12] + [6/12]).mean()
        expect = {"a": a,
                  "b": b,
                  "c": c}
        actual = agger.feature_factory_dict["user_id"]["UserLevelEncoder"].data_dict

        for k, v in expect.items():
            expect[k] = np.float32(np.round(v, 5))
        for k, v in actual.items():
            actual[k] = np.float32(np.round(v, 5))
        self.assertEqual(expect, actual)

        # partial predict
        df_test = pd.DataFrame({"user_id": ["a", "b", "d"],
                                "content_id": ["x", "y", "z"]})
        df_expect = pd.DataFrame({"user_id": ["a", "b", "d"],
                                  "user_level": [a, b, 0.5]})
        df_expect["user_level"] = df_expect["user_level"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial fit
        df_partial = pd.DataFrame({"user_id": ["a", "d"],
                                   "content_id": ["x", "z"],
                                   "answered_correctly": [0, 0]})

        agger.fit(df_partial)
        expect = {"a": np.array([0.5]*10 + [6/13] + [6/14]).mean(),
                  "b": b,
                  "c": c,
                  "d": np.array([0.5]*10 + [5/11]).mean()}
        for k, v in expect.items():
            expect[k] = np.float32(np.round(v, 5))
        for k, v in actual.items():
            actual[k] = np.float32(np.round(v, 5))

        self.assertEqual(expect, agger.feature_factory_dict["user_id"]["UserLevelEncoder"].data_dict)


if __name__ == "__main__":
    unittest.main()