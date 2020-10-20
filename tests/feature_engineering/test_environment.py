from feature_engineering.environment import EnvironmentManager, MyEnvironment
from feature_engineering.feature_factory import FeatureFactoryManager, CountEncoder, TargetEncoder
from experiment.common import get_logger
import pandas as pd
import unittest

class PartialAggregatorTestCase(unittest.TestCase):
    def test_interval1(self):
        logger = get_logger()

        df = pd.DataFrame({"row_id": [0, 1, 2, 3, 4, 5, 6, 7],
                           "user_id": ["a", "a", "a", "b", "a", "b", "a", "b"],
                           "timestamp": [0, 1, 2, 3, 4, 5, 6, 7],
                           "content_id": [0, 0, 1, 1, 0, 0, 1, 1],
                           "content_type_id": [0, 1, 0, 0, 0, 0, 0, 1],
                           "user_answer": [0, 1, 2, 3, 4, 5, 6, 7],
                           "answered_correctly": [0, -1, 1, -1, 0, 0, 1, -1],
                           "prior_question_had_explanation": [0, 0, 0, 0, 0, 0, 0, 0]}).sort_values(["user_id", "timestamp"])
        df_question = pd.DataFrame({"question_id": [0, 1],
                                    "bundle_id": [0, 1],
                                    "correct_answer": [0, 1],
                                    "part": [0, 1],
                                    "tags": ["0", "1"]})
        df_lecture = pd.DataFrame({"lecture_id": [0, 1],
                                   "tag": [0, 1],
                                   "part": [0, 1],
                                   "type_of": ["0", "1"]})
        feature_factory_dict = {
            "user_id": {
                "CountEncoder": CountEncoder(column="user_id"),
                "TargetEncoder": TargetEncoder(column="user_id")}
        }
        print(df.iloc[4:])
        gen = MyEnvironment(df_test=df.iloc[4:],
                            interval=1).iter_test()
        feature_factory_manager = FeatureFactoryManager(feature_factory_dict=feature_factory_dict,
                                                        logger=logger)

        env_manager = EnvironmentManager(feature_factory_manager=feature_factory_manager,
                                         gen=gen,
                                         fit_interval=1,
                                         df_question=df_question,
                                         df_lecture=df_lecture)

        w_df1 = pd.merge(df[df["content_type_id"] == 0], df_question, how="left", left_on="content_id",
                         right_on="question_id")
        w_df2 = pd.merge(df[df["content_type_id"] == 1], df_lecture, how="left", left_on="content_id",
                         right_on="lecture_id")
        df2 = pd.concat([w_df1, w_df2]).sort_values(["user_id", "timestamp"])
        df_expect = feature_factory_manager.all_predict(df2).iloc[4:]
        df_expect["tag"] = df_expect["tag"].fillna(-1)
        df_expect["correct_answer"] = df_expect["correct_answer"].fillna(-1)
        df_expect["bundle_id"] = df_expect["bundle_id"].fillna(-1)
        df_expect["prior_question_had_explanation"] = df_expect["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")
        df_expect.columns = [x.replace(" ", "_") for x in df_expect.columns]

        df_actual = pd.DataFrame()

        feature_factory_manager.fit(df2.iloc[:4])
        while True:
            x = env_manager.step()
            if x is None:
                break
            df_test = x[0]
            df_sub = x[1]
            df_actual = pd.concat([df_actual, df_test], axis=0)

        pd.testing.assert_frame_equal(df_expect.reset_index(drop=True),
                                      df_actual.reset_index(drop=True),
                                      check_dtype=False)

if __name__ == "__main__":
    unittest.main()