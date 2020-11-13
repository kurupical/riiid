
import unittest
import pandas as pd
import numpy as np
from feature_engineering.feature_factory import \
    FeatureFactoryManager, \
    CountEncoder, \
    TargetEncoder, \
    MeanAggregator, \
    UserLevelEncoder2, \
    NUniqueEncoder, \
    ShiftDiffEncoder, \
    Counter, \
    PreviousAnswer, \
    CategoryLevelEncoder, \
    PreviousAnswer2, \
    QuestionLectureTableEncoder, \
    PreviousLecture, \
    ContentLevelEncoder, \
    FirstColumnEncoder, \
    FirstNAnsweredCorrectly
from experiment.common import get_logger
import pickle
import os

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

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "target_enc_key1": [0, 1, 1, 1/2, np.nan]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_targetencoding_null(self):
        logger = get_logger()

        feature_factory_dict = {
            "key1": {
                "CountEncoder": CountEncoder(column="key1"),
                "TargetEncoder": TargetEncoder(column="key1")}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"key1": ["a", "a", "a", "a", "a"],
                           "answered_correctly": [0, np.nan, np.nan, 1, 0]})

        df_expect = pd.DataFrame({"target_enc_key1": [np.nan, 0, 0, 0, 1/2]})
        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"key1": ["a"]})

        df_expect = pd.DataFrame({"target_enc_key1": [1/3]})
        df_expect = df_expect.fillna(-1).astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_meanaggregator_remove_now(self):
        """
        :return:
        """
        df = pd.DataFrame({"key1": ["a", "b", "b", "c", "c", "c", "c"],
                           "data1": [0, 1, 1, 0, 1, 1, 0],
                           "answered_correctly": [0, 0, 0, 0, 0, 0, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
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

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                "data1": [1, 1, 1, 1, 1],
                                "answered_correctly": [0, 0, 0, 0, 0]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "mean_data1_by_key1": [0, 1, 1, 1/2, np.nan],
                                  "diff_mean_data1_by_key1": [1, 0, 0, 1/2, np.nan]})
        df_expect["mean_data1_by_key1"] = df_expect["mean_data1_by_key1"].astype("float32")
        df_expect["diff_mean_data1_by_key1"] = df_expect["diff_mean_data1_by_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

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

        # partial predict
        df_test = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"]})
        df_expect = pd.DataFrame({"key1": ["a", "b", "b", "c", "d"],
                                  "target_enc_key1": [5/11, 7/12, 7/12, 7/14, 0.5]})
        df_expect["target_enc_key1"] = df_expect["target_enc_key1"].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

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
                                  "user_level_content_id": [np.nan] + (np.cumsum([0/1, 0/2, 1/3, 2/4])/np.arange(1, 4+1)).tolist() +
                                                           [(0/1+0/2+1/3+2/4)/4] +
                                                           (np.cumsum([0/1, 0/2, 1/3, 2/4, 0, # np.nan -> 0に置換のため
                                                                       1/1, 2/2, 3/3, 4/4])/np.arange(9))[5:].tolist() +
                                                           (np.cumsum([4/5, 4/6, 4/7, 5/8, 6/9])/np.arange(1, 5+1)).tolist(),
                                  "user_rate_sum_content_id": np.cumsum([0, 0, 0, 1, 2/3, 2/4]).tolist() +
                                                              [0+1+2/3+2/4] + np.cumsum([0, 1, 2/3, 2/4,
                                                                                         0, # np.nan
                                                                                         0, 0, 0])[5:].tolist() +
                                                              np.cumsum([0, -4/5, -4/6, 3/7, 3/8]).tolist(),
                                  "user_rate_mean_content_id": [np.nan, np.nan] + (np.cumsum([0, 1, 2/3, 2/4])/np.arange(1, 4+1)).tolist() +
                                                               [(0+1+2/3+2/4)/4] + (np.cumsum([0, 1, 2/3, 2/4,
                                                                                               0,
                                                                                               0, 0, 0])/np.arange(8))[5:].tolist() +
                                                               [np.nan] + (np.cumsum([-4/5, -4/6, 3/7, 3/8])/np.arange(1, 4+1)).tolist()})
        for col in df_expect.columns[2:]:
            df_expect[col] = df_expect[col].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial-predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        # partial predict
        df_test = pd.DataFrame({"content_id": ["a", "a", "b", "b"],
                                "user_id": ["x", "y", "x", "y"]})

        x_level = np.array([0, 0/1, 0/2, 1/3, 2/4, 0, 1/1, 2/2, 3/3, 4/4]).sum() / 8
        y_level = np.array([4/5, 4/6, 4/7, 5/8, 6/9]).sum() / 5

        x_rate = np.array([0, 1-0/2, 1-1/3, 1-2/4, 0, 0, 0, -4/4])
        y_rate = np.array([-4/5, -4/6, 1-4/7, 1-5/8, -6/9])
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
                           "val": ["x", "x", "x", "y"],
                           "answered_correctly": [0, 0, 0, 0]})
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
                                   "val": ["x", "y", "x", "y"],
                                   "answered_correctly": [0, 0, 0, 0]})
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
                           "val": [1, 2, 4, 8],
                           "answered_correctly": [0, 0, 0, 0]})
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
                           "col1": [1, 2, 2, 3, 1, 3],
                           "answered_correctly": [0, 0, 0, 0, 0, 0]})


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
            agger.fit(df.iloc[i:i+1])

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
        df = pd.DataFrame({"user_id": [15, 15, 123, 123],
                           "content_id": [15, 15, 123, 123],
                           "answered_correctly": [1, 0, 0, 1]})
        df_expect = pd.DataFrame({"user_id": [15, 15, 123, 123],
                                  "content_id": [15, 15, 123, 123],
                                  "answered_correctly": [1, 0, 0, 1],
                                  "previous_answer_['user_id', 'content_id']": [np.nan, 1, np.nan, 0]})
        df_expect["previous_answer_['user_id', 'content_id']"] = df_expect["previous_answer_['user_id', 'content_id']"].fillna(-99).astype("int8")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial_predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df_test = pd.DataFrame({"user_id": [15, 123, 1222],
                                "content_id": [15, 123, 1222],
                                "answered_correctly": [1, 1, 1]})

        df_expect = pd.DataFrame({"user_id": [15, 123, 1222],
                                  "content_id": [15, 123, 1222],
                                  "answered_correctly": [1, 1, 1],
                                  "previous_answer_['user_id', 'content_id']": [0, 1, np.nan]})
        df_expect["previous_answer_['user_id', 'content_id']"] = df_expect["previous_answer_['user_id', 'content_id']"].fillna(-99).astype("int8")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_levelencoder(self):
        """
        user_level
        :return:
        """
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id"),
                "CategoryLevelEncoder": CategoryLevelEncoder(groupby_column="user_id",
                                                             agg_column="col1",
                                                             categories=["1", "2", "3"])
            },
            "content_id": {
                "CountEncoder": CountEncoder(column="content_id", is_partial_fit=True),
                "TargetEncoder": TargetEncoder(column="content_id", is_partial_fit=True)
            }
        }

        df = pd.DataFrame({"user_id": ["x"]*6 + ["y"]*3,
                           "content_id": ["a"]*3 + ["b"]*6,
                           "col1": ["1", "2", "3"] * 3,
                           "answered_correctly": [0, 0, 1,
                                                  1, 0, 0,
                                                  1, 1, 0]})

        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df_expect = pd.DataFrame(
            {"user_rate_sum_col1_1": [np.nan, 0, 0,
                                      0, 0, 0,
                                      np.nan, 2/3, 2/3],
             "user_rate_mean_col1_1": [np.nan, np.nan, np.nan,
                                       np.nan, np.nan, np.nan,
                                       np.nan, 2/3, 2/3],
             "user_rate_sum_col1_2": [np.nan, 0, 0,
                                      0, 0, -1,
                                      np.nan, 0, 2/4],
             "user_rate_mean_col1_2": [np.nan, np.nan, 0,
                                       0, 0, -1/2,
                                       np.nan, np.nan, 2/4],
             "user_rate_sum_col1_3": [np.nan, 0, 0,
                                      1, 1, 1,
                                      np.nan, 0, 0],
             "user_rate_mean_col1_3": [np.nan, np.nan, np.nan,
                                       1, 1, 1,
                                       np.nan, np.nan, np.nan]
             }
        )
        df_actual = agger.all_predict(df)
        df_expect = df_expect.astype("float32")
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df_test = pd.DataFrame({"user_id": ["x", "y", "z"],
                                "content_id": ["a", "c", "c"]})

        df_expect = pd.DataFrame(
            {"user_rate_sum_col1_1": [0, 2/3, np.nan],
             "user_rate_mean_col1_1": [np.nan, 2/3, np.nan],
             "user_rate_sum_col1_2": [0-1, 2/4, np.nan],
             "user_rate_mean_col1_2": [-1/2, 2/4, np.nan],
             "user_rate_sum_col1_3": [1-1/2, -3/5, np.nan],
             "user_rate_mean_col1_3": [1/2/2, -3/5, np.nan]
             }
        )
        df_expect = df_expect.astype("float32")

        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_fit_previous_answered2(self):
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "PreviousAnswer": PreviousAnswer2(groupby="user_id", column="content_id", repredict=True, is_debug=True)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        # all_predict
        df = pd.DataFrame({"user_id": [15, 15, 123, 123, 15],
                           "content_id": [15, 15, 123, 123, 123],
                           "answered_correctly": [1, 0, 0, 1, 0]})
        df_expect = pd.DataFrame({"previous_answer_content_id": [np.nan, 1, np.nan, 0, np.nan]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial_predict
        agger.partial_predict(df)
        agger.fit(df)

        df_test = pd.DataFrame({"user_id": [15, 123, 1222],
                                "content_id": [15, 123, 1222],
                                "answered_correctly": [1, 1, 1]})

        df_expect = pd.DataFrame({"user_id": [15, 123, 1222],
                                  "content_id": [15, 123, 1222],
                                  "answered_correctly": [1, 1, 1],
                                  "previous_answer_content_id": [0, 1, np.nan]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_previous_answered_index(self):
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "PreviousAnswer": PreviousAnswer2(groupby="user_id", column="content_id", repredict=True, is_debug=True)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        # all_predict
        df = pd.DataFrame({"user_id": [1, 1, 1, 1, 1, 1, 2, 2],
                           "content_id": [1, 2, 2, 1, 1, 1, 1, 1],
                           "answered_correctly": [1, 0, 1, 0, 1, 1, 0, 0]})
        df_expect = pd.DataFrame({"previous_answer_content_id": [np.nan, np.nan, 0, 1, 0, 1, np.nan, 0],
                                  "previous_answer_index_content_id": [np.nan, np.nan, 0, 2, 0, 0, np.nan, 0]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_expect["previous_answer_index_content_id"] = df_expect["previous_answer_index_content_id"].fillna(-99).astype("int16")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.partial_predict(df)
        agger.fit(df)

        df_test = pd.DataFrame({"user_id": [1, 1, 2, 2, 2, 1],
                                "content_id": [1, 2, 1, 3, 2, 1]})

        df_expect = pd.DataFrame({"previous_answer_content_id": [1, 1, 0, np.nan, np.nan, np.nan],
                                  "previous_answer_index_content_id": [0, 4, 0, np.nan, np.nan, 1]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_expect["previous_answer_index_content_id"] = df_expect["previous_answer_index_content_id"].fillna(-99).astype("int16")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        df_test = pd.DataFrame({"user_id": [1, 1, 2, 2, 2, 1],
                                "content_id": [1, 2, 1, 3, 2, 1],
                                "answered_correctly": [1, 1, 1, 0, 0, 0]})
        agger.partial_predict(df_test)
        agger.fit(df_test)

        df_test = pd.DataFrame({"user_id": [1, 1, 2],
                                "content_id": [1, 2, 3]})

        df_expect = pd.DataFrame({"previous_answer_content_id": [0, 1, 0],
                                  "previous_answer_index_content_id": [0, 2, 1]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_expect["previous_answer_index_content_id"] = df_expect["previous_answer_index_content_id"].fillna(-99).astype("int16")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_question_lecture_table_encoder(self):
        logger = get_logger()

        question_lecture_dict = {
            ("q0", "a0"): 0.1,
            ("q1", "a0"): 0.2,
            ("q0", "a1"): 0.4,
            ("q1", "a1"): 0.8,
            ("q0", "a2"): 1.6, # dummy
            ("q1", "a2"): 3.2, # dummy
        }
        feature_factory_dict = {
            "user_id": {
                "QuestionLectureTableEncoder": QuestionLectureTableEncoder(question_lecture_dict=question_lecture_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1] * 9 + [2] * 9,
                           "content_id": ["q0", "q1", "a0", "a2", "q0", "q1", "a1", "q0", "q1"] * 2,
                           "content_type_id": [0, 0, 1, 0, 0, 0, 1, 0, 0] * 2,
                           "answered_correctly": [0] * 18})

        df_expect = pd.DataFrame({"question_lecture_score": [0, 0, 0, 0, 0.1, 0.2, 0, 0.1+0.4, 0.2+0.8] * 2})
        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 1, 2, 3],
                           "content_id": ["a2", "q0", "q1", "q1"],
                           "content_type_id": [1, 0, 0, 0]})
        df_expect = pd.DataFrame({"question_lecture_score": [
            0, 0.1+0.4, 0.2+0.8, 0
        ]})
        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_question_lecture_table_create(self):

        df = pd.DataFrame({"content_id": [1, 2, 3, 4, 1, 2] + [1, 3, 2, 4, 1, 2, 2] + [5, 6],
                           "user_id": [1]*6 + [2]*7 + [3]*2,
                           "content_type_id": [0, 0, 1, 1, 0, 0] + [0, 1, 0, 1, 0, 0, 0] + [1, 1],
                           "answered_correctly": [0, 0, 1, 1, 1, 1]*2 + [0] + [0, 0]})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        expect = {
            (1, 3): 1 - 0,
            (1, 4): 1 - 0,
            (1, 5): 0,
            (1, 6): 0,
            (2, 3): 3/4 - 0,
            (2, 4): 2/3 - 1/2,
            (2, 5): 0,
            (2, 6): 0
        }

        ql_table_encoder = QuestionLectureTableEncoder(question_lecture_dict={})

        ql_table_encoder.make_dict(df, test_mode=True, output_dir=pickle_dir, threshold=0)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)

    def test_previous_lecture(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "PreviousLecture": PreviousLecture(column="content_id")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 2],
                           "content_id": [0, 1, 2, 3, 4, 5],
                           "content_type_id": [0, 1, 1, 0, 1, 0],
                           "answered_correctly": [0, 0, 0, 0, 0, 0]})

        df_expect = pd.DataFrame({"previous_lecture": [np.nan, np.nan, 1, np.nan, np.nan, 4]})
        df_expect = df_expect.fillna(-1).astype("int8")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3],
                           "content_id": [6, 7, 8, 9, 10, 11],
                           "content_type_id": [1, 1, 0, 1, 1, 1]})

        df_expect = pd.DataFrame({"previous_lecture": [2, 6, np.nan, np.nan, np.nan, 10]})
        df_expect = df_expect.fillna(-1).astype("int8")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_content_level2(self):
        """
        content_level
        :return:
        """
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id",
                                             is_partial_fit=True),
                "TargetEncoder": TargetEncoder(column="user_id",
                                               is_partial_fit=True)},
            "content_id": {
                "CountEncoder": CountEncoder(column="content_id"),
                "TargetEncoder": TargetEncoder(column="content_id"),
                "contentLevelEncoder": ContentLevelEncoder(vs_column="user_id")}
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": ["a"]*5 + ["b"]*10,
                           "content_id": ["x"]*10 + ["y"]*5,
                           "answered_correctly": [0, 0, 1, 1, 1,
                                                  1, 1, 1, 1, 0,
                                                  0, 0, 1, 1, 0]})


        # predict_all

        df_expect = pd.DataFrame({"user_id": ["a"]*5 + ["b"]*10,
                                  "content_id": ["x"]*10 + ["y"]*5,
                                  "target_enc_content_id": [np.nan, 0/1, 0/2, 1/3, 2/4,
                                                         3/5, 4/6, 5/7, 6/8, 7/9,
                                                         np.nan, 0/1, 0/2, 1/3, 2/4],
                                  "target_enc_user_id": [np.nan, 0/1, 0/2, 1/3, 2/4,
                                                            np.nan, 1/1, 2/2, 3/3, 4/4,
                                                            4/5, 4/6, 4/7, 5/8, 6/9],
                                  "content_level_user_id": [np.nan] + (np.cumsum([0/1, 0/2, 1/3, 2/4])/np.arange(1, 4+1)).tolist() +
                                                           [(0/1+0/2+1/3+2/4)/4] +
                                                           (np.cumsum([0/1, 0/2, 1/3, 2/4, 0, # np.nan -> 0に置換のため
                                                                       1/1, 2/2, 3/3, 4/4])/np.arange(9))[5:].tolist() +
                                                           (np.cumsum([4/5, 4/6, 4/7, 5/8, 6/9])/np.arange(1, 5+1)).tolist(),
                                  "content_rate_sum_user_id": np.cumsum([0, 0, 0, 1, 2/3, 2/4]).tolist() +
                                                              [0+1+2/3+2/4] + np.cumsum([0, 1, 2/3, 2/4,
                                                                                         0, # np.nan
                                                                                         0, 0, 0])[5:].tolist() +
                                                              np.cumsum([0, -4/5, -4/6, 3/7, 3/8]).tolist(),
                                  "content_rate_mean_user_id": [np.nan, np.nan] + (np.cumsum([0, 1, 2/3, 2/4])/np.arange(1, 4+1)).tolist() +
                                                               [(0+1+2/3+2/4)/4] + (np.cumsum([0, 1, 2/3, 2/4,
                                                                                               0,
                                                                                               0, 0, 0])/np.arange(8))[5:].tolist() +
                                                               [np.nan] + (np.cumsum([-4/5, -4/6, 3/7, 3/8])/np.arange(1, 4+1)).tolist()})
        for col in df_expect.columns[2:]:
            df_expect[col] = df_expect[col].astype("float32")
        df_actual = agger.all_predict(df)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # fit - partial-predict
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        # partial predict
        df_test = pd.DataFrame({"user_id": ["a", "a", "b", "b"],
                                "content_id": ["x", "y", "x", "y"]})

        x_level = np.array([0, 0/1, 0/2, 1/3, 2/4, 0, 1/1, 2/2, 3/3, 4/4]).sum() / 8
        y_level = np.array([4/5, 4/6, 4/7, 5/8, 6/9]).sum() / 5

        x_rate = np.array([0, 1-0/2, 1-1/3, 1-2/4, 0, 0, 0, -4/4])
        y_rate = np.array([-4/5, -4/6, 1-4/7, 1-5/8, -6/9])
        df_expect = pd.DataFrame({"user_id": ["a", "a", "b", "b"],
                                  "content_id": ["x", "y", "x", "y"],
                                  "target_enc_content_id": [7/10, 2/5, 7/10, 2/5],
                                  "target_enc_user_id": [3/5, 3/5, 6/10, 6/10],
                                  "content_level_user_id": [x_level, y_level, x_level, y_level],
                                  "content_rate_sum_user_id": [x_rate.sum(), y_rate.sum(), x_rate.sum(), y_rate.sum()],
                                  "content_rate_mean_user_id": [x_rate.mean(), y_rate.mean(), x_rate.mean(), y_rate.mean()],
                                  })
        for col in df_expect.columns[2:]:
            df_expect[col] = df_expect[col].astype("float32")
        df_actual = agger.partial_predict(df_test)
        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_first_column(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "FirstColumn": FirstColumnEncoder(column="key1",
                                                  astype="int8")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 2],
                           "key1": [0, 1, 2, 3, 4, 5],
                           "answered_correctly": [0, 0, 0, 0, 0, 0]})

        df_expect = pd.DataFrame({"first_column_key1": [0, 0, 0, 3, 3, 3]})
        df_expect = df_expect.fillna(-1).astype("int8")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3, 3],
                           "key1": [6, 7, 8, 9],
                           "answered_correctly": [0, 0, 0, 0]})

        df_expect = pd.DataFrame({"first_column_key1": [0, 3, 8, 8]})
        df_expect = df_expect.fillna(-1).astype("int8")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        df = pd.DataFrame({"user_id": [3],
                           "key1": [10],
                           "answered_correctly": [0]})

        df_expect = pd.DataFrame({"first_column_key1": [8]})
        df_expect = df_expect.fillna(-1).astype("int8")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_first_n_answered_correctly(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "FirstColumn": FirstNAnsweredCorrectly(n=2)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 3],
                           "answered_correctly": [0, 1, np.nan, np.nan, 1, 0]})

        df_expect = pd.DataFrame({"first_2_ans": ["", "0", "01", "", "9", ""]})
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 2, 3, 3, 4]})

        df_expect = pd.DataFrame({"first_2_ans": ["019", "91", "0", "0", ""]})
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


if __name__ == "__main__":
    unittest.main()