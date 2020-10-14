import pandas as pd
from feature_engineering.partial_aggregator import PartialAggregator

class Pipeline:

    def __init__(self, logger):
        self.pipelines = [
            PartialAggregator(key="user_id", logger=logger),
            PartialAggregator(key="content_id", logger=logger),
            PartialAggregator(key="content_type_id", logger=logger),
            PartialAggregator(key="task_container_id", logger=logger),
            PartialAggregator(key=["user_id", "content_type_id"], logger=logger),
            PartialAggregator(key=["user_id", "content_id"], logger=logger),
        ]
        self.logger = logger

    def _common_transform(self,
                          df: pd.DataFrame):
        df = df.drop(["prior_question_had_explanation", "prior_question_elapsed_time"], axis=1)
        return df

    def fit_transform(self,
                      df: pd.DataFrame):
        df = self._common_transform(df)

        for pipeline in self.pipelines:
            pipeline.fit(df)

        for pipeline in self.pipelines:
            df = pipeline.all_predict(df)

        return df

    def partial_transform(self,
                          df: pd.DataFrame):
        df = self._common_transform(df)
        for pipeline in self.pipelines:
            self.logger.info(pipeline)
            df = pipeline.partial_predict(df)
        return df

    def fit(self,
            df: pd.DataFrame):
        for pipeline in self.pipelines:
            pipeline.fit(df)