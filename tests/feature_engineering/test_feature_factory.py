
import unittest
import pandas as pd
import numpy as np
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    CountEncoder, \
    TargetEncoder, \
    MeanAggregator, \
    UserLevelEncoder, \
    UserLevelEncoder2, \
    NUniqueEncoder, \
    ShiftDiffEncoder, \
    Counter, \
    PreviousAnswer
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
                "CountEncoder": CountEncoder(column="content_id",
                                             is_partial_fit=True),
                "TargetEncoder": TargetEncoder(column="content_id",
                                               initial_weight=10,
                                               initial_score=0.5,
                                               is_partial_fit=True)},
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

        agger.fit(df, partial_predict_mode=True)
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

        agger.fit(df_partial, partial_predict_mode=True)
        expect = {"a": np.array([0.5]*10 + [6/13] + [6/14]).mean(),
                  "b": b,
                  "c": c,
                  "d": np.array([0.5]*10 + [5/11]).mean()}
        for k, v in expect.items():
            expect[k] = np.float32(np.round(v, 5))
        for k, v in actual.items():
            actual[k] = np.float32(np.round(v, 5))

        self.assertEqual(expect, agger.feature_factory_dict["user_id"]["UserLevelEncoder"].data_dict)

    def test_fit_user_level2(self):
        """
        user_level
        :return:
        """
        logger = get_logger()
        feature_factory_dict = {
            "content_id": {
                "CountEncoder": CountEncoder(column="content_id",
                                             is_partial_fit=True),
                "TargetEncoder": TargetEncoder(column="content_id",
                                               is_partial_fit=True)},
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id"),
                "TargetEncoder": TargetEncoder(column="user_id"),
                "UserLevelEncoder": UserLevelEncoder2(vs_column="content_id")}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"content_id": ["a"]*5 + ["b"]*10,
                           "user_id": ["x"]*10 + ["y"]*5,
                           "answered_correctly": [0, 0, 1, 1, 1,
                                                  1, 1, 1, 1, 0,
                                                  0, 0, 1, 1, 0]})


        # predict_all

        df_expect = pd.DataFrame({"content_id": ["a"]*5 + ["b"]*10,
                                  "user_id": ["x"]*10 + ["y"]*5,
                                  "target_enc_user_id": [np.nan, 0/1, 0/2, 1/3, 2/4,
                                                         3/5, 4/6, 5/7, 6/8, 7/9,
                                                         np.nan, 0/1, 0/2, 1/3, 2/4],
                                  "target_enc_content_id": [np.nan, 0/1, 0/2, 1/3, 2/4,
                                                            np.nan, 1/1, 2/2, 3/3, 4/4,
                                                            4/5, 4/6, 4/7, 5/8, 6/9],
                                  "user_level_content_id": [np.nan] + (np.cumsum([0/1, 0/2, 1/3, 2/4])/np.arange(2, 4+2)).tolist() +
                                                           [np.nan] + (np.cumsum([0/1, 0/2, 1/3, 2/4, 0, # np.nan -> 0に置換のため
                                                                                  1/1, 2/2, 3/3, 4/4])/np.arange(2, 9+2))[5:].tolist() +
                                                           (np.cumsum([4/5, 4/6, 4/7, 5/8, 6/9])/np.arange(1, 5+1)).tolist(),
                                  "user_rate_sum_content_id": [np.nan, np.nan] + np.cumsum([0, 1, 2/3, 2/4]).tolist() +
                                                              [np.nan] + np.cumsum([0, 1, 2/3, 2/4,
                                                                                    0, # np.nan
                                                                                    0, 0, 0])[5:].tolist() +
                                                              [np.nan] + np.cumsum([-4/5, -4/6, 3/7, 3/8]).tolist(),
                                  "user_rate_mean_content_id": [np.nan, np.nan] + (np.cumsum([0, 1, 2/3, 2/4])/np.arange(2, 4+2)).tolist() +
                                                               [np.nan] + (np.cumsum([0, 1, 2/3, 2/4,
                                                                                      0, # np.nan
                                                                                      0, 0, 0])/np.arange(2, 8+2))[5:].tolist() +
                                                               [np.nan] + (np.cumsum([-4/5, -4/6, 3/7, 3/8])/np.arange(1, 4+1)).tolist()})
        for col in df_expect.columns[2:]:
            df_expect[col] = df_expect[col].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial-predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1], partial_predict_mode=True)

        # partial predict
        df_test = pd.DataFrame({"content_id": ["a", "a", "b", "b"],
                                "user_id": ["x", "y", "x", "y"]})

        x_level = np.array([0/1, 0/2, 1/3, 2/4, 3/5, 1/1, 2/2, 3/3, 4/4, 4/5]).sum() / 10
        y_level = np.array([4/6, 4/7, 5/8, 6/9, 6/10]).sum() / 5

        x_rate = np.array([0, 0, 1-1/3, 1-2/4, 1-3/5, 0, 0, 0, 0, -4/5])
        y_rate = np.array([-4/6, -4/7, 1-5/8, 1-6/9, -6/10])
        df_expect = pd.DataFrame({"content_id": ["a", "a", "b", "b"],
                                  "user_id": ["x", "y", "x", "y"],
                                  "target_enc_user_id": [7/10, 2/5, 7/10, 2/5],
                                  "target_enc_content_id": [3/5, 3/5, 6/10, 6/10],
                                  "user_level_content_id": [x_level, y_level, x_level, y_level],
                                  "user_rate_sum_content_id": [x_rate.sum(), y_rate.sum(), x_rate.sum(), y_rate.sum()],
                                  "user_rate_mean_content_id": [x_rate.mean(), y_rate.mean(), x_rate.mean(), y_rate.mean()],
                                  })
        for col in df_expect.columns[2:]:
            df_expect[col] = df_expect[col].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_nunique(self):
        """
        nunique
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "a", "b", "b"],
                           "val": ["x", "x", "x", "y"]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1"),
                "NUniqueEncoder": NUniqueEncoder(groupby="key1", column="val")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "a", "b", "b"],
                                  "val": ["x", "x", "x", "y"],
                                  "nunique_val_by_key1": [1, 1, 1, 2],
                                  "new_ratio_nunique_val_by_key1": [1, 0.5, 1, 1]})
        df_expect["nunique_val_by_key1"] = df_expect["nunique_val_by_key1"].astype("int32")
        df_expect["new_ratio_nunique_val_by_key1"] = df_expect["new_ratio_nunique_val_by_key1"].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        agger.fit(df)

        # partial predict
        df_actual = agger.partial_predict(df)
        df_expect = pd.DataFrame({"key1": ["a", "a", "b", "b"],
                                  "val": ["x", "x", "x", "y"],
                                  "nunique_val_by_key1": [1, 1, 2, 2], # fitは１行ずつ読まないので、[1, 1, 1, 2]ではない
                                  "new_ratio_nunique_val_by_key1": [0.5, 0.5, 1, 1]})
        df_expect["nunique_val_by_key1"] = df_expect["nunique_val_by_key1"].astype("int32")
        df_expect["new_ratio_nunique_val_by_key1"] = df_expect["new_ratio_nunique_val_by_key1"].astype("float32")
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial fit
        df_partial = pd.DataFrame({"key1": ["a", "a", "b", "c"],
                                   "val": ["x", "y", "x", "y"]})
        agger.fit(df_partial)

        df = pd.DataFrame({"key1": ["a", "b", "c", "d"],
                           "val": ["", "", "", ""]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "c", "d"],
                                  "val": ["", "", "", ""],
                                  "nunique_val_by_key1": [2, 2, 1, 0],
                                  "new_ratio_nunique_val_by_key1": [0.5, 2/3, 1, np.nan]})
        df_expect["nunique_val_by_key1"] = df_expect["nunique_val_by_key1"].astype("int32")
        df_expect["new_ratio_nunique_val_by_key1"] = df_expect["new_ratio_nunique_val_by_key1"].astype("float32")
        df_actual = agger.partial_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_shiftdiff(self):
        """
        nunique
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "a", "b", "b"],
                           "val": [1, 2, 4, 8]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "ShiftDiffEncoder": ShiftDiffEncoder(groupby="key1", column="val")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "a", "b", "b"],
                                  "val": [1, 2, 4, 8],
                                  "shiftdiff_val_by_key1": [0, 1, 0, 4]})
        df_expect["shiftdiff_val_by_key1"] = df_expect["shiftdiff_val_by_key1"].astype("int64")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit
        agger.fit(df)

        # partial predict
        df = pd.DataFrame({"key1": ["a", "a", "b", "c"],
                           "val": [4, 8, 16, 1]})
        df_actual = agger.partial_predict(df)
        df_expect = pd.DataFrame({"key1": ["a", "a", "b", "c"],
                                  "val": [4, 8, 16, 1],
                                  "shiftdiff_val_by_key1": [2, 4, 8, 1]})
        df_expect["shiftdiff_val_by_key1"] = df_expect["shiftdiff_val_by_key1"].astype("int64")
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial predict 2 (リアルタイムfitができていることを確認)
        df = pd.DataFrame({"key1": ["a", "a"],
                           "val": [32, 64]})
        df_expect = pd.DataFrame({"key1": ["a", "a"],
                                  "val": [32, 64],
                                  "shiftdiff_val_by_key1": [24, 32]})
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual)

    def test_fit_counter(self):
        """
        user_level
        :return:
        """
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id"),
                "Counter": Counter(groupby_column="user_id",
                                   agg_column="col1",
                                   categories=[1, 2, 3])
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": ["x", "x", "x", "y", "y", "y"],
                           "col1": [1, 2, 2, 3, 1, 3]})


        # predict_all

        df_expect = pd.DataFrame({"user_id": ["x", "x", "x", "y", "y", "y"],
                                  "col1": [1, 2, 2, 3, 1, 3],
                                  "groupby_user_id_col1_1_count": [1, 1, 1, 0, 1, 1],
                                  "groupby_user_id_col1_2_count": [0, 1, 2, 0, 0, 0],
                                  "groupby_user_id_col1_3_count": [0, 0, 0, 1, 1, 2],
                                  "groupby_user_id_col1_1_count_ratio": [1, 1/2, 1/3, 0, 1/2, 1/3],
                                  "groupby_user_id_col1_2_count_ratio": [0, 1/2, 2/3, 0, 0, 0],
                                  "groupby_user_id_col1_3_count_ratio": [0, 0, 0, 1, 1/2, 2/3]})
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial-predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1], partial_predict_mode=True)

        # partial predict
        df_test = pd.DataFrame({"user_id": ["x", "y"],
                                "col1": [1, 2]})
        df_expect = pd.DataFrame({"user_id": ["x", "y"],
                                  "col1": [1, 2],
                                  "groupby_user_id_col1_1_count": [1, 1],
                                  "groupby_user_id_col1_2_count": [2, 0],
                                  "groupby_user_id_col1_3_count": [0, 2],
                                  "groupby_user_id_col1_1_count_ratio": [1/3, 1/3],
                                  "groupby_user_id_col1_2_count_ratio": [2/3, 0],
                                  "groupby_user_id_col1_3_count_ratio": [0, 2/3]})

        df_actual = agger.partial_predict(df_test)
        print(df_actual)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns],
                                      check_dtype=False)

    def test_fit_previous_answered(self):
        logger = get_logger()
        feature_factory_dict = {
            ("user_id", "content_id"): {
                "PreviousAnswer": PreviousAnswer(column=["user_id", "content_id"])
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        # all_predict
        df = pd.DataFrame({"user_id": ["x", "x", "y", "y"],
                           "content_id": ["x", "x", "y", "y"],
                           "answered_correctly": [1, 0, 0, 1]})
        df_expect = pd.DataFrame({"user_id": ["x", "x", "y", "y"],
                                  "content_id": ["x", "x", "y", "y"],
                                  "answered_correctly": [1, 0, 0, 1],
                                  "previous_answer_['user_id', 'content_id']": [np.nan, 1, np.nan, 0]})
        df_expect["previous_answer_['user_id', 'content_id']"] = df_expect["previous_answer_['user_id', 'content_id']"].fillna(-99).astype("int8")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial_predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df_test = pd.DataFrame({"user_id": ["x", "y", "z"],
                                "content_id": ["x", "y", "z"],
                                "answered_correctly": [1, 1, 1]})

        df_expect = pd.DataFrame({"user_id": ["x", "y", "z"],
                                  "content_id": ["x", "y", "z"],
                                  "answered_correctly": [1, 1, 1],
                                  "previous_answer_['user_id', 'content_id']": [0, 1, np.nan]})
        df_expect["previous_answer_['user_id', 'content_id']"] = df_expect["previous_answer_['user_id', 'content_id']"].fillna(-99).astype("int8")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

if __name__ == "__main__":
    unittest.main()