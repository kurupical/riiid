
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
    CategoryLevelEncoder, \
    PreviousAnswer2, \
    QuestionLectureTableEncoder, \
    PreviousLecture, \
    ContentLevelEncoder, \
    FirstColumnEncoder, \
    FirstNAnsweredCorrectly, \
    SessionEncoder, \
    PreviousNAnsweredCorrectly, \
    QuestionLectureTableEncoder2, \
    UserAnswerLevelEncoder, \
    QuestionQuestionTableEncoder, \
    WeightDecayTargetEncoder, \
    UserContentRateEncoder, \
    Word2VecEncoder, \
    ElapsedTimeVsShiftDiffEncoder, \
    QuestionQuestionTableEncoder2, \
    TagsTargetEncoder, \
    PastNFeatureEncoder, \
    PreviousContentAnswerTargetEncoder
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
        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

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
        df = pd.DataFrame({"key1": ["a", "a", "b", "b", "b"],
                           "val": [1, 2, 4, 8, 8],
                           "answered_correctly": [0, 0, 0, 0, 0]})
        logger = get_logger()
        feature_factory_dict = {
            "key1": {
                "ShiftDiffEncoder": ShiftDiffEncoder(groupby="key1", column="val")
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        # predict_all
        df_expect = pd.DataFrame({"key1": ["a", "a", "b", "b", "b"],
                                  "val": [1, 2, 4, 8, 8],
                                  "shiftdiff_val_by_key1": [0, 1, 0, 4, 4]})
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
        df = pd.DataFrame({"key1": ["a", "a", "a"],
                           "val": [32, 64, 64]})
        df_expect = pd.DataFrame({"key1": ["a", "a", "a"],
                                  "val": [32, 64, 64],
                                  "shiftdiff_val_by_key1": [24, 32, 32]}) # valが前の値と同じなら、shiftdiffも前の値と同じ
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
                "PreviousAnswer": PreviousAnswer2(groupby="user_id", column="content_id", is_debug=True)
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

    def test_fit_previous_answered_index_limited(self):
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "PreviousAnswer": PreviousAnswer2(groupby="user_id",
                                                  column="content_id",
                                                  n=2,
                                                  is_debug=True)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        # all_predict
        df = pd.DataFrame({"user_id": [1, 1, 1, 1, 1, 1],
                           "content_id": [1, 2, 2, 1, 2, 1],
                           "answered_correctly": [1, 0, 1, 0, 1, 1]})
        df_expect = pd.DataFrame({"previous_answer_content_id": [np.nan, np.nan, 0, np.nan, 1, 0],
                                  "previous_answer_index_content_id": [np.nan, np.nan, 0, np.nan, 1, 1]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_expect["previous_answer_index_content_id"] = df_expect["previous_answer_index_content_id"].fillna(-99).astype("int16")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.partial_predict(df)
        agger.fit(df)

        df_test = pd.DataFrame({"user_id": [1, 1, 2, 2, 2, 1],
                                "content_id": [1, 2, 1, 3, 2, 1]})

        df_expect = pd.DataFrame({"previous_answer_content_id": [1, np.nan, np.nan, np.nan, np.nan, np.nan],
                                  "previous_answer_index_content_id": [0, np.nan, np.nan, np.nan, np.nan, 1]})
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

        df_expect = pd.DataFrame({"previous_answer_content_id": [0, np.nan, 0],
                                  "previous_answer_index_content_id": [0, np.nan, 1]})
        df_expect["previous_answer_content_id"] = df_expect["previous_answer_content_id"].fillna(-99).astype("int8")
        df_expect["previous_answer_index_content_id"] = df_expect["previous_answer_index_content_id"].fillna(-99).astype("int16")
        df_actual = agger.partial_predict(df_test)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_fit_previous_answered_index(self):
        logger = get_logger()
        feature_factory_dict = {
            "user_id": {
                "PreviousAnswer": PreviousAnswer2(groupby="user_id", column="content_id", is_debug=True)
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


    def test_question_lecture_table_encoder2(self):
        logger = get_logger()

        # (lecture, question, is_lectured, past_answered)
        question_lecture_dict = {
            (100, 1, 0, 0): 0.005,
            (100, 1, 0, 1): 0.01,
            (100, 1, 1, 0): 0.02,
            (100, 1, 1, 1): 0.04,
            (101, 1, 0, 0): 0.08,
            (101, 1, 0, 1): 0.16,
            (101, 1, 1, 0): 0.32,
            (101, 1, 1, 1): 0.64
        }
        feature_factory_dict = {
            "user_id": {
                "QuestionLectureTableEncoder2": QuestionLectureTableEncoder2(question_lecture_dict=question_lecture_dict,
                                                                             past_n=2,
                                                                             min_size=0)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)


        u1_c_id = [1, 100, 101, 1]
        u1_p_ans = [-99, -99, -99, 1]
        u2_c_id = [100, 101, 1]
        u2_p_ans = [-99, -99, -99]
        u3_c_id = [1, 100, 1, 101, 101, 1]
        u3_p_ans = [-99, -99, 0, -99, -99, 1]

        user_id = [3]*4 + [2]*3 + [1]*6
        content_id = u1_c_id + u2_c_id + u3_c_id
        content_type_id = [0 if x < 100 else 1 for x in content_id]
        answered_correctly = [0] * len(content_id)
        past_answered = u1_p_ans + u2_p_ans + u3_p_ans
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly,
                           "previous_answer_content_id": past_answered})

        u1_score = [[np.nan], [np.nan], [np.nan], [0.04, 0.64]]
        u2_score = [[np.nan], [np.nan], [0.02, 0.32]]
        u3_score = [[np.nan], [np.nan], [0.04], [np.nan], [np.nan], [0.64, 0.64]]

        score = u1_score + u2_score + u3_score

        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "ql_table2_mean": expect_mean,
            "ql_table2_sum": expect_sum,
            "ql_table2_max": expect_max,
            "ql_table2_min": expect_min,
            "ql_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        content_id = [101, 1, 1]
        content_type_id = [0 if x < 100 else 1 for x in content_id]
        past_answered = [0, 1, 1]
        df = pd.DataFrame({"user_id": [3, 3, 4],
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "previous_answer_content_id": past_answered})
        score = [[np.nan], [0.64, 0.64], [np.nan]]
        expect_mean = [np.array(x).mean() for x in score]
        expect_sum = [np.array(x).sum() for x in score]
        expect_max = [np.array(x).max() for x in score]
        expect_min = [np.array(x).min() for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "ql_table2_mean": expect_mean,
            "ql_table2_sum": expect_sum,
            "ql_table2_max": expect_max,
            "ql_table2_min": expect_min,
            "ql_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_question_lecture2_table_create(self):

        user1_content_id = [100, 1, 2, 100, 1, 2]
        user1_answered_correctly = [None, 1, 1, None, 0, 1]
        user2_content_id = [1, 2, 100, 1, 2]
        user2_answered_correctly = [0, 0, None, 0, 1]
        user3_content_id = [1, 2, 1, 2, 1, 2]
        user3_answered_correctly = [1, 0, 0, 0, 1, 1]
        user4_content_id = [100, 1, 2, 101, 3]
        user4_answered_correctly = [None, 1, 1, None, 1]

        user_id = [1] * len(user1_content_id) + \
                  [2] * len(user2_content_id) + \
                  [3] * len(user3_content_id) + \
                  [4] * len(user4_content_id)
        content_id = user1_content_id + user2_content_id + user3_content_id + user4_content_id
        content_type_id = [0 if x < 100 else 1 for x in content_id]
        answered_correctly = user1_answered_correctly + user2_answered_correctly + \
                             user3_answered_correctly + user4_answered_correctly

        df = pd.DataFrame({"content_id": content_id,
                           "user_id": user_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # key: (lecture_id, question_id, is_lectured, past_answered)
        expect = {
            (100, 1, 1, 0): [1, 1],
            (100, 1, 1, 1): [0, 0],
            (100, 2, 1, 0): [1, 1],
            (100, 2, 1, 1): [1 ,1],
            (100, 3, 1, 0): [1],
            (101, 3, 1, 0): [1],
        }
        for key, value in expect.items():
            expect[key] = (np.array(value).sum() + 30*0.65) / (len(value) + 30)

        ql_table_encoder = QuestionLectureTableEncoder2(question_lecture_dict={},
                                                        past_n=2,
                                                        min_size=0)

        ql_table_encoder.make_dict(df, test_mode=True, output_dir=pickle_dir)
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
                "FirstColumn": FirstColumnEncoder(agg_column="key1",
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

    def test_previous_n_answered_correctly(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "Previous2AnsweredCorrectly": PreviousNAnsweredCorrectly(n=3)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1, 1, 1, 1, 2, 2, 4, 4, 4, 4],
                           "bundle_id": [1, np.nan, 3, 4, 5, 6, 7, 7, 7, 8],
                           "content_type_id": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           "answered_correctly": [0, np.nan, 1, 0, 0, 1, 0, 1, 1, 0]})

        df_expect = pd.DataFrame({"previous_3_ans": ["", "0", "90", "190", "", "0", "", "8", "88", "110"]})
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df)

        # partial_predict1
        df1 = pd.DataFrame({"user_id": [1, 2, 3],
                            "bundle_id": [8, 9, np.nan],
                            "content_type_id": [0, 0, 1],
                            "answered_correctly": [0, 0, np.nan]})

        df_expect = pd.DataFrame({"previous_3_ans": ["019", "10", ""]})
        df_actual = agger.partial_predict(df1)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # partial_predict時点でわからない情報は8にする
        df2 = pd.DataFrame({"user_id": [1, 2, 3, 3],
                            "bundle_id": [11, 12, 13, 13],
                            "content_type_id": [0, 0, 0, 0],
                            "answered_correctly": [0, 0, 1, 1]})

        df_expect = pd.DataFrame({"previous_3_ans": ["801", "810", "9", "89"]})
        df_actual = agger.partial_predict(df2)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        # df1, df2をfitしたら、partial_predict時点で分からなかったところを塗り替える
        agger.fit(pd.concat([df1, df2]))

        df3 = pd.DataFrame({"user_id": [1, 2, 3],
                            "bundle_id": [8, 9, np.nan],
                            "content_type_id": [0, 0, 1],
                            "answered_correctly": [0, 0, np.nan]})

        df_expect = pd.DataFrame({"previous_3_ans": ["000", "001", "119"]})
        df_actual = agger.partial_predict(df3)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_session(self):
        """
        user1: session0 finished & session1
        user2: session0 途中 -> partial_predictでsession0 end & session1 start
        user3: sessionなし -> partial_predictで
        :return:
        """
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "SessionEncoder": SessionEncoder()
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2],
                           "timestamp": [0, 10, 10**7, 0, 10],
                           "answered_correctly": [0, 0, 0, 0, 0]})

        df_expect = pd.DataFrame({"session": [0, 0, 1, 0, 0],
                                  "session_nth": [0, 1, 0, 0, 1],
                                  "first_session_nth": [0, 1, np.nan, 0, 1]}).fillna(-1).astype("int16")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3, 3, 3],
                           "timestamp": [10**8, 10**8+10, 20, 10**7, 0, 10, 10**7]})

        df_expect = pd.DataFrame({"session": [2, 2, 0, 1, 0, 0, 1],
                                  "session_nth": [0, 1, 2, 0, 0, 1, 0],
                                  "first_session_nth": [np.nan, np.nan, 2, np.nan, 0, 1, np.nan]})
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_create_userans_level_dict(self):

        user_id = [0, 0, 0, 0, 0, 1, 0]
        content_id = [0, 0, 0, 1, 1, 1, 0]
        content_type_id = [0, 0, 0, 0, 0, 0, 1]
        user_answer = [0, 1, 1, 3, 1, 3, np.nan]
        answered_correctly = [1, 1, 0, 1, 1, 1, np.nan]

        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "user_answer": user_answer,
                           "answered_correctly": answered_correctly})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # expected target_enc_user_id
        te = [np.nan, 1/1, 2/2, 2/3, 3/4, np.nan, np.nan]

        # key: (content_id, user_answer)
        expect = {
            (0, 1.0): [te[1], te[2]],
            (1, 1.0): [te[4]], # te[5] = np.nan
            (1, 3.0): [te[3]]
        }
        for key, value in expect.items():
            expect[key] = (np.array(value).sum() + 30*0.65) / (len(value) + 30)

        cid_useranswer_dict = UserAnswerLevelEncoder(user_answer_dict={},
                                                     past_n=2,
                                                     min_size=0)

        cid_useranswer_dict.make_dict(df, output_dir=pickle_dir)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        for k in expect.keys():
            self.assertAlmostEqual(expect[k], actual[k])
        os.remove(pickle_dir)

    def test_userans_level(self):
        logger = get_logger()

        # key: (content_id, user_answer)
        user_answer_dict = {
            (101, 1): 0.005,
            (101, 2): 0.01,
            (102, 1): 0.02,
            (102, 2): 0.04
        }
        feature_factory_dict = {
            "user_id": {
                "UserAnswerLevelEncoder": UserAnswerLevelEncoder(user_answer_dict=user_answer_dict,
                                                                 past_n=2,
                                                                 min_size=0)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        u1_c_id = [101, 101, 102, 101, 102]
        u1_u_answer = [1, 2, 1, np.nan, 2]
        u1_c_t_id = [0, 0, 0, 1, 0]
        u2_c_id = [102, 102]
        u2_u_answer = [1, 2]
        u2_c_t_id = [0, 0]

        user_id = [1]*5 + [2]*2
        user_answer = u1_u_answer + u2_u_answer
        content_id = u1_c_id + u2_c_id
        content_type_id = u1_c_t_id + u2_c_t_id
        answered_correctly = [0] * len(content_id)
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "user_answer": user_answer,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})

        u1_score = [[np.nan], [0.005], [0.005, 0.01], [0.01, 0.02], [0.01, 0.02]]
        u2_score = [[np.nan], [0.02]]

        score = u1_score + u2_score

        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "content_ua_table2_mean": expect_mean,
            "content_ua_table2_sum": expect_sum,
            "content_ua_table2_max": expect_max,
            "content_ua_table2_min": expect_min,
            "content_ua_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        user_id = [1, 2, 3]
        content_id = [101, 102, 101]
        content_type_id = [0, 1, 0]
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id})
        score = [[0.02, 0.04], [np.nan], [np.nan]]
        expect_mean = [np.array(x).mean() for x in score]
        expect_sum = [np.array(x).sum() for x in score]
        expect_max = [np.array(x).max() for x in score]
        expect_min = [np.array(x).min() for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "content_ua_table2_mean": expect_mean,
            "content_ua_table2_sum": expect_sum,
            "content_ua_table2_max": expect_max,
            "content_ua_table2_min": expect_min,
            "content_ua_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_question_question_table_create(self):

        user1_content_id = [1, 2, 3, 4, 1, 2, 3, 3]
        user1_answered_correctly = [1, 1, 0, np.nan, 1, 0, 1, 0]
        user1_content_type_id = [0, 0, 0, 1, 0, 0, 0, 0]
        user2_content_id = [2, 3, 4, 5]
        user2_answered_correctly = [1, 1, 1, 1]
        user2_content_type_id = [0, 0, 0, 0]

        user_id = [1] * len(user1_content_id) + [2] * len(user2_content_id)
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        ans_c = user1_answered_correctly + user2_answered_correctly

        df = pd.DataFrame({"content_id": content_id,
                           "user_id": user_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": ans_c})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # key: (lecture_id, question_id, is_lectured, past_answered)
        expect = {
            (1, 2, 1, 0): [ans_c[1]],
            (1, 3, 1, 0): [ans_c[2]],
            (2, 3, 1, 0): [ans_c[2], ans_c[9]],
            (2, 4, 1, 0): [ans_c[10]],
            (2, 5, 1, 0): [ans_c[11]],
            (3, 4, 1, 0): [ans_c[10]],
            (3, 5, 1, 0): [ans_c[11]],
            (4, 5, 1, 0): [ans_c[11]],
            (1, 2, 1, 1): [ans_c[5]],
            (1, 3, 1, 1): [ans_c[6], ans_c[7]],
            (2, 1, 1, 1): [ans_c[4]],
            (2, 3, 1, 1): [ans_c[6], ans_c[7]],
            (3, 1, 1, 1): [ans_c[4]],
            (3, 2, 1, 1): [ans_c[5]]
        }
        for key, value in expect.items():
            expect[key] = np.array(value).sum() / len(value)
        qq_table_encoder = QuestionQuestionTableEncoder(question_lecture_dict={},
                                                        past_n=2,
                                                        min_size=0)

        qq_table_encoder.make_dict(df, test_mode=True, output_dir=pickle_dir)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)


    def test_question_question_table_encoder(self):
        logger = get_logger()

        # (lecture, question, is_lectured, past_answered)
        question_lecture_dict = {
            (1, 2, 1, 0): 0.005,
            (1, 2, 1, 1): 0.02,
            (1, 3, 1, 0): 0.08,
            (2, 1, 1, 0): 0.32,
            (2, 1, 1, 1): 1.28,
            (2, 3, 1, 0): 5.12,
        }
        feature_factory_dict = {
            "user_id": {
                "QuestionQuestionTableEncoder": QuestionQuestionTableEncoder(question_lecture_dict=question_lecture_dict,
                                                                             past_n=3,
                                                                             min_size=0)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)


        u1_c_id = [1, 2, 1, 1, 2, 3]
        u1_c_t_id = [0, 0, 1, 0, 0, 0]
        u1_p_ans = [-99, -99, -99, 1, 1, -99]
        u2_c_id = [1, 1, 2, 2]
        u2_c_t_id = [0, 0, 0, 0]
        u2_p_ans = [-99, 1, -99, 1]

        user_id = [1]*6 + [2]*4
        content_id = u1_c_id + u2_c_id
        content_type_id = u1_c_t_id + u2_c_t_id
        answered_correctly = [0] * len(content_id)
        past_answered = u1_p_ans + u2_p_ans
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly,
                           "previous_answer_content_id": past_answered})

        u1_score = [[np.nan], [0.005], [np.nan], [1.28], [0.02], [0.08, 5.12]]
        u2_score = [[np.nan], [np.nan], [0.005, 0.005], [0.02]]

        score = u1_score + u2_score

        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "qq_table_mean": expect_mean,
            "qq_table_sum": expect_sum,
            "qq_table_max": expect_max,
            "qq_table_min": expect_min,
            "qq_table_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df)

        content_id = [3, 1, 1]
        content_type_id = [0, 1, 0]
        past_answered = [-99, 1, 1]
        df = pd.DataFrame({"user_id": [2, 2, 3],
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "previous_answer_content_id": past_answered})
        score = [[5.12, 5.12], [np.nan], [np.nan]]
        expect_mean = [np.array(x).mean() for x in score]
        expect_sum = [np.array(x).sum() for x in score]
        expect_max = [np.array(x).max() for x in score]
        expect_min = [np.array(x).min() for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "qq_table_mean": expect_mean,
            "qq_table_sum": expect_sum,
            "qq_table_max": expect_max,
            "qq_table_min": expect_min,
            "qq_table_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])



    def test_weighted_target_encoder(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "WeightedTargetEncoder": WeightDecayTargetEncoder(column="user_id",
                                                                  past_n=3,
                                                                  decay=0.1)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)


        user1_ans = [0, 0, 1, 0, np.nan, 1]
        user2_ans = [0, 1]
        user1_ans_2 = [1]
        user_id = [0]*6 + [1]*2 + [0]*1
        answered_correctly = user1_ans + user2_ans + user1_ans_2

        df = pd.DataFrame({"user_id": user_id,
                           "answered_correctly": answered_correctly})

        expect = [
            np.nan,
            (user1_ans[0]*1) / 1,
            (user1_ans[0]*0.9 + user1_ans[1]*1) / 1.9,
            (user1_ans[0]*0.8 + user1_ans[1]*0.9 + user1_ans[2]*1) / 2.7,
            (user1_ans[1]*0.8 + user1_ans[2]*0.9 + user1_ans[3]*1) / 2.7,
            (user1_ans[2]*0.9 + user1_ans[3]*1) / 1.9,
            np.nan,
            (user2_ans[0]*1) / 1,
            (user1_ans[3]*0.9 + user1_ans[5]*1) / 1.9
        ]

        df_expect = pd.DataFrame({"weighted_te_user_id_past3_decay0.1": expect}).astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        # partial_predict1
        ans_partial1 = [1, 1, 1, np.nan]
        df1 = pd.DataFrame({"user_id": [0, 1, 2, 2],
                            "answered_correctly": ans_partial1})

        expect = [
            (user1_ans[3]*0.8 + user1_ans[5]*0.9 + user1_ans_2[0]*1) / 2.7,
            (user2_ans[0]*0.9 + user2_ans[1]*1) / 1.9,
            np.nan,
            np.nan
        ]
        df_expect = pd.DataFrame({"weighted_te_user_id_past3_decay0.1": expect}).astype("float32")
        df_actual = agger.partial_predict(df1)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df1)
        # partial_predict2

        df2 = pd.DataFrame({"user_id": [0, 0, 1, 2],
                            "answered_correctly": [0, np.nan, 0, 0]})

        expect = [
            (user1_ans[5]*0.8 + user1_ans_2[0]*0.9 + ans_partial1[0]*1) / 2.7,
            (user1_ans[5]*0.8 + user1_ans_2[0]*0.9 + ans_partial1[0]*1) / 2.7,
            (user2_ans[0]*0.8 + user2_ans[1]*0.9 + ans_partial1[1]*1) / 2.7,
            (ans_partial1[2]*1) / 1
        ]

        df_expect = pd.DataFrame({"weighted_te_user_id_past3_decay0.1": expect}).astype("float32")
        df_actual = agger.partial_predict(df2)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_user_content_rate(self):
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # prepare test data
        user1_content_id = [1, 2, 2, 3]
        user1_content_type_id = [0, 0, 1, 0]
        user1_answered_correctly = [0, 0, np.nan, 1]

        user2_content_id = [1, 2, 4]
        user2_content_type_id = [0, 0, 0]
        user2_answered_correctly = [1, 1, 1]

        user_id = [0]*4 + [1]*3
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly

        df = pd.DataFrame({"user_id": user_id,
                           "timestamp": [0, 1, 2, 3, 4, 5, 6],
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})
        encoder = UserContentRateEncoder(column="user_id",
                                         rate_func="simple",
                                         initial_rate=1500,
                                         content_rate_dict={})

        encoder.make_dict(df, output_dir=pickle_dir)

        # 25以下は切り捨て
        # 1. content_id=1 (1500 -> 1516) user_id=0 (1500 -> 1484)
        # 2. content_id=2 (1500 -> 1516) user_id=0 (1484 -> 1468)
        # 3. content_id=2 (no change)
        # 4. content_id=3 (1500 -> 1483) user_id=0 (1468 -> 1485)
        # 5. content_id=1 (1516 -> 1500) user_id=1 (1500 -> 1516)
        # 6. content_id=2 (1516 -> 1500) user_id=1 (1516 -> 1532)
        # 7. content_id=4 (1500 -> 1485) user_id=1 (1532 -> 1547)

        expect_content_rate = {
            1: 1500,
            2: 1500,
            3: 1483,
            4: 1485
        }

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect_content_rate, actual)
        os.remove(pickle_dir)


    def test_question_user_content_rate_encoder(self):
        logger = get_logger()

        # (lecture, question, is_lectured, past_answered)
        content_rate_dict = {
            1: 1700,
            2: 1600,
            3: 1500,
            4: 1400
        }
        feature_factory_dict = {
            "user_id": {
                "UserContentRateEncoder": UserContentRateEncoder(column="user_id",
                                                                 rate_func="simple",
                                                                 initial_rate=1500,
                                                                 content_rate_dict=content_rate_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        user1_content_id = [1, 2, 3, 3, 4]
        user1_content_type_id = [0, 0, 1, 0, 0]
        user1_answered_correctly = [1, 1, np.nan, 0, 1]

        user2_content_id = [1, 2, 3]
        user2_content_type_id = [0, 0, 0]
        user2_answered_correctly = [0, 0, 0]

        user_id = [1]*5 + [2]*3
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})

        # [user 1]
        # 1. win 16 + (1700-1500)*0.04 = 24! 1500 -> 1524
        # 2. win 16 + (1600-1524)*0.04 = 19! 1524 -> 1543
        # 3. lose 16 + (1543-1500)*0.04 = 17! 1543 -> 1526
        # 4. win 16 + (1400-1526)*0.04 = 11! 1526 -> 1537
        u1_rate = [1500, 1524, np.nan, 1543, 1526]

        # [user 2]
        # 1. lose 16 + (1500-1700)*0.04 = 8! 1500 -> 1492
        # 2. lose 16 + (1492-1600)*0.04 = 12! 1492 -> 1480
        # 3. lose 16 + (1480-1500)*0.04 = 16! 1480 -> 1464
        u2_rate = [1500, 1492, 1480]
        rate = u1_rate + u2_rate

        df_expect = pd.DataFrame({
            "content_rating": [1700, 1600, np.nan, 1500, 1400, 1700, 1600, 1500],
            "user_id_rating": rate
        })

        df_expect = df_expect.fillna(-1).astype("int16")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])


        df = pd.DataFrame({"user_id": [1, 2, 2, 3, 3],
                           "content_id": [1, 2, 1, 2, 5],
                           "content_type_id": [0, 0, 0, 0, 1]})

        df_expect = pd.DataFrame({
            "content_rating": [1700, 1600, 1700, 1600, np.nan],
            "user_id_rating": [1537, 1464, 1464, 1500, np.nan]
        })

        df_expect = df_expect.fillna(-1).astype("int16")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_user_content_rate_multi_keys(self):
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # prepare test data
        user1_content_id = [1, 2, 2, 3]
        user1_content_type_id = [0, 0, 1, 0]
        user1_answered_correctly = [0, 0, np.nan, 1]

        user2_content_id = [1, 2, 4]
        user2_content_type_id = [0, 0, 0]
        user2_answered_correctly = [1, 1, 1]

        user_id = [0]*7
        part = [0]*4 + [1]*3
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly

        df = pd.DataFrame({"user_id": user_id,
                           "part": part,
                           "timestamp": [0, 1, 2, 3, 4, 5, 6],
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})
        encoder = UserContentRateEncoder(column=["user_id", "part"],
                                         rate_func="simple",
                                         initial_rate=1500,
                                         content_rate_dict={})

        encoder.make_dict(df, output_dir=pickle_dir)

        # 25以下は切り捨て
        # 1. content_id=1 (1500 -> 1516) user_id=0/part=0 (1500 -> 1484)
        # 2. content_id=2 (1500 -> 1516) user_id=0/part=0 (1484 -> 1468)
        # 3. content_id=2 (no change)
        # 4. content_id=3 (1500 -> 1483) user_id=0/part=0 (1468 -> 1485)
        # 5. content_id=1 (1516 -> 1500) user_id=0/part=1 (1500 -> 1516)
        # 6. content_id=2 (1516 -> 1500) user_id=0/part=1 (1516 -> 1532)
        # 7. content_id=4 (1500 -> 1485) user_id=0/part=1 (1532 -> 1547)

        expect_content_rate = {
            1: 1500,
            2: 1500,
            3: 1483,
            4: 1485
        }

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect_content_rate, actual)
        os.remove(pickle_dir)


    def test_question_user_content_rate_encoder_multiple(self):
        logger = get_logger()

        # (lecture, question, is_lectured, past_answered)
        content_rate_dict = {
            1: 1700,
            2: 1600,
            3: 1500,
            4: 1400
        }
        feature_factory_dict = {
            "user_id": {
                "UserContentRateEncoder": UserContentRateEncoder(column=["user_id", "part"],
                                                                 rate_func="simple",
                                                                 initial_rate=1500,
                                                                 content_rate_dict=content_rate_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        user1_content_id = [1, 2, 3, 3, 4]
        user1_content_type_id = [0, 0, 1, 0, 0]
        user1_answered_correctly = [1, 1, np.nan, 0, 1]

        user2_content_id = [1, 2, 3]
        user2_content_type_id = [0, 0, 0]
        user2_answered_correctly = [0, 0, 0]

        user_id = [1]*8
        part = [1]*5 + [2]*3
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly
        df = pd.DataFrame({"user_id": user_id,
                           "part": part,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})

        # [user 1]
        # 1. win 16 + (1700-1500)*0.04 = 24! 1500 -> 1524
        # 2. win 16 + (1600-1524)*0.04 = 19! 1524 -> 1543
        # 3. lose 16 + (1543-1500)*0.04 = 17! 1543 -> 1526
        # 4. win 16 + (1400-1526)*0.04 = 11! 1526 -> 1537
        u1_rate = [1500, 1524, np.nan, 1543, 1526]

        # [user 2]
        # 1. lose 16 + (1500-1700)*0.04 = 8! 1500 -> 1492
        # 2. lose 16 + (1492-1600)*0.04 = 12! 1492 -> 1480
        # 3. lose 16 + (1480-1500)*0.04 = 16! 1480 -> 1464
        u2_rate = [1500, 1492, 1480]
        rate = u1_rate + u2_rate

        df_expect = pd.DataFrame({
            "content_rating": [1700, 1600, np.nan, 1500, 1400, 1700, 1600, 1500],
            "['user_id', 'part']_rating": rate
        })

        df_expect = df_expect.fillna(-1).astype("int16")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])


        df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2],
                           "part": [1, 2, 2, 1, 1],
                           "content_id": [1, 2, 1, 2, 5],
                           "content_type_id": [0, 0, 0, 0, 1]})

        df_expect = pd.DataFrame({
            "content_rating": [1700, 1600, 1700, 1600, np.nan],
            "['user_id', 'part']_rating": [1537, 1464, 1464, 1500, np.nan]
        })

        df_expect = df_expect.fillna(-1).astype("int16")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_w2v(self):
        logger = get_logger()

        # (lecture, question, is_lectured, past_answered)
        w2v_dict = {
            "1_0": [0.01, 0.02],
            "1_1": [0.04, 0.08],
            "2_0": [0.16, 0.32],
            "3_0": [0.64, 1.28]
        }
        window = 3
        size = 2
        columns = ["content_id", "content_type_id"]

        feature_factory_dict = {
            "user_id": {
                "Word2VecEncoder": Word2VecEncoder(columns=columns,
                                                   window=window,
                                                   size=size,
                                                   w2v_dict=w2v_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        user1_content_id = [1, 1, 2, 4, 3]
        user1_content_type_id = [0, 1, 0, 0, 0]
        user1_answered_correctly = [0, 0, 0, 0, 0]

        user2_content_id = [1, 2]
        user2_content_type_id = [0, 0]
        user2_answered_correctly = [0, 0]

        user_id = [1]*5 + [2]*2
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly})

        user1_w2v_dim0 = np.array([0.01, 0.04, 0.16, np.nan, 0.64])
        user1_w2v_dim1 = np.array([0.02, 0.08, 0.32, np.nan, 1.28])

        user2_w2v_dim0 = np.array([0.01, 0.16])
        user2_w2v_dim1 = np.array([0.02, 0.32])

        def get_max_min_mean(x):
            start_idxs = [x-window if x-window > 0 else 0 for x in range(len(x))]
            end_idxs = [x+1 for x in range(len(x))]
            max_ary = [x[start_idx:end_idx].max() for start_idx, end_idx in zip(start_idxs, end_idxs)]
            min_ary = [x[start_idx:end_idx].min() for start_idx, end_idx in zip(start_idxs, end_idxs)]
            mean_ary = [x[start_idx:end_idx].mean() for start_idx, end_idx in zip(start_idxs, end_idxs)]
            return max_ary, min_ary, mean_ary

        user1_max_dim0, user1_min_dim0, user1_mean_dim0 = get_max_min_mean(user1_w2v_dim0)
        user1_max_dim1, user1_min_dim1, user1_mean_dim1 = get_max_min_mean(user1_w2v_dim1)

        user2_max_dim0, user2_min_dim0, user2_mean_dim0 = get_max_min_mean(user2_w2v_dim0)
        user2_max_dim1, user2_min_dim1, user2_mean_dim1 = get_max_min_mean(user2_w2v_dim1)

        col_name = f"w2v_{columns}_window{window}_size{size}"
        df_expect = pd.DataFrame({
            f"{col_name}_dim0": user1_w2v_dim0.tolist() + user2_w2v_dim0.tolist(),
            f"{col_name}_dim1": user1_w2v_dim1.tolist() + user2_w2v_dim1.tolist(),
            f"swem_max_{col_name}_dim0": user1_max_dim0 + user2_max_dim0,
            f"swem_max_{col_name}_dim1": user1_max_dim1 + user2_max_dim1,
            f"swem_min_{col_name}_dim0": user1_min_dim0 + user2_min_dim0,
            f"swem_min_{col_name}_dim1": user1_min_dim1 + user2_min_dim1,
            f"swem_mean_{col_name}_dim0": user1_mean_dim0 + user2_mean_dim0,
            f"swem_mean_{col_name}_dim1": user1_mean_dim1 + user2_mean_dim1,
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        for i in range(len(df)):
            agger.fit(df.iloc[i:i+1])

        df = pd.DataFrame({"user_id": [1, 1, 2, 3],
                           "content_id": [1, 2, 1, 2],
                           "content_type_id": [0, 0, 0, 0]})

        w2v_dim0 = [0.01, 0.16, 0.01, 0.16]
        w2v_dim1 = [0.02, 0.32, 0.02, 0.32]

        w2v_ary_dim0 = [
            [np.nan, 0.64, 0.01],
            [np.nan, 0.64, 0.01],
            [0.01, 0.16, 0.01],
            [0.16]
        ]
        w2v_ary_dim1 = [
            [np.nan, 1.28, 0.02],
            [np.nan, 1.28, 0.02],
            [0.02, 0.32, 0.02],
            [0.32]
        ]

        max_dim0 = [np.array(x).max() for x in w2v_ary_dim0]
        min_dim0 = [np.array(x).min() for x in w2v_ary_dim0]
        mean_dim0 = [np.array(x).mean() for x in w2v_ary_dim0]

        max_dim1 = [np.array(x).max() for x in w2v_ary_dim1]
        min_dim1 = [np.array(x).min() for x in w2v_ary_dim1]
        mean_dim1 = [np.array(x).mean() for x in w2v_ary_dim1]

        df_expect = pd.DataFrame({
            f"{col_name}_dim0": w2v_dim0,
            f"{col_name}_dim1": w2v_dim1,
            f"swem_max_{col_name}_dim0": max_dim0,
            f"swem_max_{col_name}_dim1": max_dim1,
            f"swem_min_{col_name}_dim0": min_dim0,
            f"swem_min_{col_name}_dim1": min_dim1,
            f"swem_mean_{col_name}_dim0": mean_dim0,
            f"swem_mean_{col_name}_dim1": mean_dim1,
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

    def test_elapsed_time_create_dict(self):
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # prepare test data
        user_id = [1, 1, 2, 2, 1, 1]
        content_id = [1, 2, 1, 2, 1, 2]
        elapsed_time = [np.nan, 2, np.nan, 4, 8, 16]

        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "prior_question_elapsed_time": elapsed_time})
        encoder = ElapsedTimeVsShiftDiffEncoder(elapsed_time_dict={})

        encoder.make_dict(df, output_dir=pickle_dir)

        elapsed_time_dict = {
            1: (2+4+16)/3,
            2: 8
        }

        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(elapsed_time_dict, actual)
        os.remove(pickle_dir)


    def test_question_question_table2_create(self):

        user1_content_id = [1, 2, 3, 3, 1, 2, 3, 3]
        user1_answered_correctly = [0, 1, 1, np.nan, 1, 0, 1, 0]
        user1_content_type_id = [0, 0, 0, 1, 0, 0, 0, 0]
        user2_content_id = [1, 2]
        user2_answered_correctly = [1, 1]
        user2_content_type_id = [0, 0]

        user_id = [1] * len(user1_content_id) + [2] * len(user2_content_id)
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        ans_c = user1_answered_correctly + user2_answered_correctly

        df = pd.DataFrame({"content_id": content_id,
                           "user_id": user_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": ans_c})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # key: (lecture_id, question_id, past_answered, answered_correctly)
        expect = {
            (1, 1, 1, 0): [ans_c[4]],
            (1, 2, 0, 0): [ans_c[1]],
            (1, 2, 0, 1): [ans_c[9]],
            (1, 2, 1, 0): [ans_c[5]],
            (1, 2, 1, 1): [ans_c[5]],
            (1, 3, 0, 0): [ans_c[2]],
            (1, 3, 1, 0): [ans_c[6], ans_c[7]],
            (1, 3, 1, 1): [ans_c[6], ans_c[7]],
            (2, 1, 1, 1): [ans_c[4]],
            (2, 2, 1, 1): [ans_c[5]],
            (2, 3, 0, 1): [ans_c[2]],
            (2, 3, 1, 0): [ans_c[6], ans_c[7]],
            (2, 3, 1, 1): [ans_c[6], ans_c[7]],
            (3, 1, 1, 1): [ans_c[4]],
            (3, 2, 1, 1): [ans_c[5]],
            (3, 3, 1, 1): [ans_c[6], ans_c[7]]
        }
        for key, value in expect.items():
            expect[key] = np.array(value).sum() / len(value)
        qq_table_encoder = QuestionQuestionTableEncoder2(question_lecture_dict={},
                                                         past_n=2,
                                                         min_size=0)

        qq_table_encoder.make_dict(df, test_mode=True, output_dir=pickle_dir)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)

    def test_question_question_table2_encoder(self):
        logger = get_logger()

        # key: (lecture_id, question_id, past_answered, answered_correctly)
        values = 0.001 * 2 ** np.arange(16)
        question_question_dict = {}
        i = 0
        for lid in [1, 2]:
            for qid in [1, 2]:
                for pastans in [0, 1]:
                    for anscor in [0, 1]:
                        question_question_dict[(lid, qid, pastans, anscor)] = values[i]
                        i += 1

        feature_factory_dict = {
            "user_id": {
                "QuestionQuestionTableEncoder2": QuestionQuestionTableEncoder2(question_lecture_dict=question_question_dict,
                                                                               past_n=2,
                                                                               min_size=0)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        user1_content_id = [1, 2, 1, 2, 2]
        user1_content_type_id = [0, 0, 0, 1, 0]
        user1_answered_correctly = [0, 0, 1, np.nan, 1]
        user1_past_answered = [-99, -99, 1, np.nan, 1]

        user2_content_id = [2, 2]
        user2_content_type_id = [0, 0]
        user2_answered_correctly = [1, 0]
        user2_past_answered = [0, 1]

        user_id = [1]*5 + [2]*2
        content_id = user1_content_id + user2_content_id
        content_type_id = user1_content_type_id + user2_content_type_id
        answered_correctly = user1_answered_correctly + user2_answered_correctly
        past_answered = user1_past_answered + user2_past_answered
        df = pd.DataFrame({"user_id": user_id,
                           "content_id": content_id,
                           "content_type_id": content_type_id,
                           "answered_correctly": answered_correctly,
                           "previous_answer_content_id": past_answered})

        user1_score = [[np.nan], [values[4]], [values[2], values[10]], [np.nan], [values[14], values[7]]]
        user2_score = [[np.nan], [values[15]]]

        score = user1_score + user2_score

        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_sum = [np.array(x).sum() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "qq_table2_mean": expect_mean,
            "qq_table2_sum": expect_sum,
            "qq_table2_max": expect_max,
            "qq_table2_min": expect_min,
            "qq_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df)

        df = pd.DataFrame({"user_id": [1, 2, 3],
                           "content_id": [1, 1, 1],
                           "content_type_id": [0, 0, 0],
                           "previous_answer_content_id": [1, -99, -99]})
        score = [[values[3], values[11]], [values[9], values[8]], [np.nan]]
        expect_mean = [np.array(x).mean() for x in score]
        expect_sum = [np.array(x).sum() for x in score]
        expect_max = [np.array(x).max() for x in score]
        expect_min = [np.array(x).min() for x in score]
        expect_last = [x[-1] for x in score]

        df_expect = pd.DataFrame({
            "qq_table2_mean": expect_mean,
            "qq_table2_sum": expect_sum,
            "qq_table2_max": expect_max,
            "qq_table2_min": expect_min,
            "qq_table2_last": expect_last
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_tags_target_encoder_create_dict(self):

        tags = ["1 2 3",
                "1 2 4",
                "2 3 4",
                "5",
                np.nan,
                "2 5"]
        ans = [1, 0, 1, 0, np.nan, 0]
        df = pd.DataFrame({"tags": tags,
                           "answered_correctly": ans})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # key: (lecture_id, question_id, past_answered, answered_correctly)
        expect = {
            "1": (ans[0] + ans[1])/2,
            "2": (ans[0] + ans[1] + ans[2] + ans[5])/4,
            "3": ans[2],
            "4": (ans[1] + ans[2])/2,
            "5": (ans[3] + ans[5])/2
        }
        encoder = TagsTargetEncoder(tags_dict={})

        encoder.make_dict(df, output_dir=pickle_dir)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)


    def test_tags_target_encoder(self):
        logger = get_logger()

        # key: (lecture_id, question_id, past_answered, answered_correctly)
        tags_dict = {
            "1": 0.01,
            "2": 0.02,
            "3": 0.04
        }

        feature_factory_dict = {
            "user_id": {
                "TagsTargetEncoder": TagsTargetEncoder(tags_dict=tags_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        tags = ["1",
                "1 2",
                np.nan,
                "1 2 3"]
        ans = [0, 0, np.nan, 0]
        df = pd.DataFrame({"tags": tags,
                           "answered_correctly": ans})

        score = [
            [0.01],
            [0.01, 0.02],
            [],
            [0.01, 0.02, 0.04]
        ]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]

        df_expect = pd.DataFrame({
            "tags_te_mean": expect_mean,
            "tags_te_max": expect_max,
            "tags_te_min": expect_min,
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        print(df_expect)
        print(df_actual)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df)

        tags = ["1",
                "1 2",
                np.nan,
                "1 2 3"]
        ans = [0, 0, np.nan, 0]
        df = pd.DataFrame({"tags": tags,
                           "answered_correctly": ans})
        score = [
            [0.01],
            [0.01, 0.02],
            [],
            [0.01, 0.02, 0.04]
        ]
        expect_mean = [np.array(x).mean() if len(x) > 0 else np.nan for x in score]
        expect_max = [np.array(x).max() if len(x) > 0 else np.nan for x in score]
        expect_min = [np.array(x).min() if len(x) > 0 else np.nan for x in score]

        df_expect = pd.DataFrame({
            "tags_te_mean": expect_mean,
            "tags_te_max": expect_max,
            "tags_te_min": expect_min,
        })

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])



    def test_past_n_feature_encoder(self):
        logger = get_logger()

        feature_factory_dict = {
            "user_id": {
                "PastNFetureEncoder": PastNFeatureEncoder(past_ns=[2, 5],
                                                          column="value",
                                                          remove_now=False,
                                                          agg_funcs=["max", "min", "mean", "last", "vslast"])
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)

        user1_value = [10, 20, 30, 40, 50]
        user2_value = [10, np.nan, 20]
        df = pd.DataFrame({"user_id": [1]*5 + [2]*3,
                           "value": user1_value + user2_value,
                           "answered_correctly": [0]*8})
        user1_ary = [
            [10],
            [10, 20],
            [10, 20, 30],
            [10, 20, 30, 40],
            [10, 20, 30, 40, 50]
        ]
        user2_ary = [
            [10],
            [10],
            [10, 20]
        ]

        data_dict = {}
        score = user1_ary + user2_ary
        for past_n in [2, 5]:
            data_dict[f"past{past_n}_value_mean"] = [np.array(x[-past_n:]).mean() if len(x) > 0 else np.nan for x in score]
            data_dict[f"past{past_n}_value_max"] = [np.array(x[-past_n:]).max() if len(x) > 0 else np.nan for x in score]
            data_dict[f"past{past_n}_value_min"] = [np.array(x[-past_n:]).min() if len(x) > 0 else np.nan for x in score]

        data_dict["past2_value_last"] = [np.nan, 10, 20, 30, 40, np.nan, np.nan, 10]
        data_dict["past5_value_last"] = [np.nan, np.nan, np.nan, np.nan, 10, np.nan, np.nan, np.nan]
        data_dict["past2_value_vslast"] = [np.nan, 10, 10, 10, 10, np.nan, np.nan, 10]
        data_dict["past5_value_vslast"] = [np.nan, np.nan, np.nan, np.nan, 40, np.nan, np.nan, np.nan]
        df_expect = pd.DataFrame(data_dict)

        df_expect = df_expect.astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df)

        df = pd.DataFrame({"user_id": [1, 1, 3],
                           "value": [60, 70, 10]})
        score = [
            [10, 20, 30, 40, 50, 60],
            [10, 20, 30, 40, 50, 60, 70],
            [10]
        ]
        for past_n in [2, 5]:
            data_dict[f"past{past_n}_value_mean"] = [np.array(x[-past_n:]).mean() if len(x) > 0 else np.nan for x in score]
            data_dict[f"past{past_n}_value_max"] = [np.array(x[-past_n:]).max() if len(x) > 0 else np.nan for x in score]
            data_dict[f"past{past_n}_value_min"] = [np.array(x[-past_n:]).min() if len(x) > 0 else np.nan for x in score]

        data_dict["past2_value_last"] = [50, 60, np.nan]
        data_dict["past5_value_last"] = [20, 30, np.nan]
        data_dict["past2_value_vslast"] = [10, 10, np.nan]
        data_dict["past5_value_vslast"] = [40, 40, np.nan]

        df_expect = pd.DataFrame(data_dict)

        df_expect = df_expect.astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


    def test_previous_content_answer_te_make_dict(self):

        u1_c_id = [1, 2, 3]
        u1_c_type = [0, 0, 0]
        u1_u_ans = [1, 1, 2]
        u1_target = [1, 1, 1]

        u2_c_id = [1, 2, 3]
        u2_c_type = [0, 0, 0]
        u2_u_ans = [2, 2, 1]
        u2_target = [0, 0, 0]

        u3_c_id = [1, 2, 3]
        u3_c_type = [0, 0, 0]
        u3_u_ans = [2, 2, 1]
        u3_target = [1, 0, 1]

        u4_c_id = [1, 2]
        u4_c_type = [-1, 0]
        u4_u_ans = [-1, 1]
        u4_target = [-1, 0]

        df = pd.DataFrame({"user_id": [1]*3 + [2]*3 + [3]*3 + [4]*2,
                           "content_id": u1_c_id + u2_c_id + u3_c_id + u4_c_id,
                           "content_type_id": u1_c_type + u2_c_type + u3_c_type + u4_c_type,
                           "user_answer": u1_u_ans + u2_u_ans + u3_u_ans + u4_u_ans,
                           "answered_correctly": u1_target + u2_target + u3_target + u4_target})
        pickle_dir = "./test_dict.pickle"
        if os.path.isdir(pickle_dir):
            os.remove(pickle_dir)

        # key: (now_content_id, past_content_id, user_answer)
        expect = {
            (1, -1, -1): 2/3,
            (2, 1, -1): 0/1,
            (2, 1, 1): 1/1,
            (2, 1, 2): 0/2,
            (3, 2, 1): 1/1,
            (3, 2, 2): 1/2
        }

        encoder = PreviousContentAnswerTargetEncoder(prev_dict={},
                                                     min_size=0)

        encoder.make_dict(df, output_dir=pickle_dir)
        with open(pickle_dir, "rb") as f:
            actual = pickle.load(f)

        self.assertEqual(expect, actual)
        os.remove(pickle_dir)


    def test_previous_content_answer_te(self):
        logger = get_logger()

        prev_dict = {
            (1, -1, -1): 0.005,
            (2, 1, -1): 0.01,
            (2, 1, 1): 0.02,
            (2, 1, 2): 0.04,
            (3, 2, -1): 0.08,
            (3, 2, 1): 0.16,
            (3, 2, 2): 0.32,
            (4, 3, 1): 0.64,
            (4, 3, 2): 1.28
        }

        feature_factory_dict = {
            "user_id": {
                "PreviousContentAnswerTargetEncoder": PreviousContentAnswerTargetEncoder(prev_dict=prev_dict)
            }
        }
        agger = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                      logger=logger)
        u1_c_id = [1, 2, 3]
        u1_c_type = [0, 0, 0]
        u1_u_ans = [1, 2, 1]

        u2_c_id = [1, 2, 3]
        u2_c_type = [0, 0, 0]
        u2_u_ans = [2, 1, 2]

        u3_c_id = [1, 2]
        u3_c_type = [0, 0]
        u3_u_ans = [-1, 1]

        df = pd.DataFrame({"user_id": [1]*3 + [2]*3 + [3]*2,
                           "content_id": u1_c_id + u2_c_id + u3_c_id,
                           "content_type_id": u1_c_type + u2_c_type + u3_c_type,
                           "user_answer": u1_u_ans + u2_u_ans + u3_u_ans,
                           "answered_correctly": [0]*8})

        df_expect = pd.DataFrame({"prev_ans_te": [0.005, 0.02, 0.32, 0.005, 0.04, 0.16, 0.005, 0.01]}).astype("float32")
        df_actual = agger.all_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])

        agger.fit(df.iloc[0:1])
        agger.fit(df.iloc[1:2])
        agger.fit(df.iloc[2:3])
        agger.fit(df.iloc[3:4])
        agger.fit(df.iloc[4:6])
        agger.fit(df.iloc[6:8])

        df = pd.DataFrame({"user_id": [1, 2, 3, 3, 4],
                           "content_id": [4, 4, 3, 3, 1],
                           "content_type_id": [0, 0, 0, 0, 0],
                           "user_answer": [1, 1, 1, 1, 1]})

        df_expect = pd.DataFrame({"prev_ans_te": [0.64, 1.28, 0.16, 0.16, 0.005]}).astype("float32")
        df_actual = agger.partial_predict(df)

        pd.testing.assert_frame_equal(df_expect, df_actual[df_expect.columns])


if __name__ == "__main__":
    unittest.main()