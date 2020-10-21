from feature_engineering.feature_factory import FeatureFactoryManager
from experiment.common import merge
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class MyEnvironment:
    def __init__(self,
                 df_test: pd.DataFrame,
                 interval: int):
        self.df_test = df_test
        self.df_test["answered_correctly"] = self.df_test["answered_correctly"].fillna(-1).astype("int8")
        self.interval = interval

    def _to_list_str(self,
                     ary: list):
        ary = [str(x) for x in ary]
        s = ", ".join(ary)
        s = "[" + s + "]"
        return s

    def _set_dtype(self,
                   df: pd.DataFrame):
        dtype_dict = {
            "timestamp": "int64",
            "user_id": "int32",
            "content_id": "int16",
            "content_type_id": "int8",
            "task_container_id": "int16",
            "user_answer": "int8",
        }
        for k, v in dtype_dict.items():
            df[k] = df[k].astype(v)
        return df

    def iter_test(self):
        df_prev = pd.DataFrame()
        for i in range(len(self.df_test)//self.interval):
            df_test = self.df_test.iloc[i*self.interval:(i+1)*self.interval]
            df_test = self._set_dtype(df_test)
            df_test["prior_group_answers_correct"] = np.nan
            df_test["prior_group_responses"] = np.nan
            if len(df_prev) == 0:
                df_test["prior_group_answers_correct"].iloc[0] = "[]"
                df_test["prior_group_responses"].iloc[0] = "[]"
            else:
                df_test["prior_group_answers_correct"].iloc[0] = self._to_list_str(df_prev["answered_correctly"].values.tolist())
                df_test["prior_group_responses"].iloc[0] = self._to_list_str(df_prev["user_answer"].values.tolist())
            df_sub = df_test[["row_id"]]
            df_sub["answered_correctly"] = 0.5
            df_prev = df_test[:]
            df_test = df_test.drop(["answered_correctly", "user_answer"], axis=1, errors="ignore")
            yield df_test, df_sub

class EnvironmentManager:
    def __init__(self,
                 feature_factory_manager: FeatureFactoryManager,
                 gen,
                 fit_interval: int = 1,
                 df_question: pd.DataFrame=None,
                 df_lecture: pd.DataFrame=None):

        self.feature_factory_manager = feature_factory_manager
        self.gen = gen
        self.fit_interval = fit_interval
        self.df_question = df_question
        self.df_lecture = df_lecture

        self.df_test_prev = pd.DataFrame()
        self.answered_correctly = []
        self.user_answer = []

    def _strlist_to_list(self,
                         series: pd.Series):
        series = series[series.notnull()]
        ret = []
        for s in series.values:
            if s != "[]":
                ret.extend([int(x) for x in str(s).replace("[", "").replace("'", "").replace("]", "").replace(" ", "").split(",")])
        return ret

    def _update_previous_list(self, df: pd.DataFrame):
        self.answered_correctly.extend(self._strlist_to_list(df["prior_group_answers_correct"]))
        self.user_answer.extend(self._strlist_to_list(df["prior_group_responses"]))

    def _fit(self):
        self.df_test_prev["answered_correctly"] = self.answered_correctly
        self.df_test_prev["user_answer"] = self.user_answer
        self.df_test_prev["user_answer"] = self.df_test_prev["user_answer"].astype("int8")
        self.df_test_prev["answered_correctly"] = self.df_test_prev["answered_correctly"].replace(-1, np.nan)
        self.feature_factory_manager.fit(self.df_test_prev)
        self.answered_correctly = []
        self.user_answer = []
        self.df_test_prev = pd.DataFrame()

    def step(self):
        try:
            x = self.gen.__next__()

            df_test = x[0]
            df_sub = x[1]

            self._update_previous_list(df_test)
            if len(self.df_test_prev) >= self.fit_interval:
                self._fit()

            df_test = merge(df=df_test,
                            df_question=self.df_question,
                            df_lecture=self.df_lecture)

            # null処理
            df_test["tag"] = df_test["tag"].fillna(-1)
            df_test["correct_answer"] = df_test["correct_answer"].fillna(-1)
            df_test["bundle_id"] = df_test["bundle_id"].fillna(-1)
            df_test["prior_question_had_explanation"] = df_test["prior_question_had_explanation"].astype("float16").fillna(-1).astype("int8")

            df = self.feature_factory_manager.partial_predict(df_test)
            df.columns = [x.replace(" ", "_") for x in df.columns]

            self.df_test_prev = self.df_test_prev.append(df)

            df = df.drop(["prior_group_answers_correct", "prior_group_responses"], axis=1)
            return df, df_sub

        except StopIteration:
            return None