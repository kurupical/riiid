import pandas as pd
from feature_engineering.partial_aggregator import PartialAggregator

class Pipeline:

    def __init__(self, logger):
        self.user_id_pa = PartialAggregator(key="user_id")
        self.content_id_pa = PartialAggregator(key="content_id")
        self.task_container_id = PartialAggregator(key="task_container_id")
        self.logger = logger

    def _common_transform(self,
                          df: pd.DataFrame):
        df = df.drop(["prior_question_had_explanation", "prior_question_elapsed_time"], axis=1)
        return df

    def fit_transform(self,
                      df: pd.DataFrame):
        df = self._common_transform(df)

        self.user_id_pa.fit(df)
        self.content_id_pa.fit(df)
        self.task_container_id.fit(df)

        for pa in [self.user_id_pa, self.content_id_pa, self.task_container_id]:
            df = pa.all_predict(df)

        return df

    def partial_transform(self,
                          df: pd.DataFrame):
        df = self._common_transform(df)
        for pa in [self.user_id_pa, self.content_id_pa, self.task_container_id]:
            df = pa.partial_predict(df)
        return df

    def fit(self,
            df: pd.DataFrame):
        for pa in [self.user_id_pa, self.content_id_pa, self.task_container_id]:
            pa.fit(df)