import pandas as pd
from feature_engineering.all_id import one_hot_encoding_count


def transform(df: pd.DataFrame):
    """
    question, lectureがマージされたdfを加工する.
    :param df:
    :return:
    """

    # ---------------
    # 前処理
    # ---------------
    df["not_answered"] = (df["user_answer"] == -1).astype(int)

    # ---------------
    # row_count
    # ---------------
    for id_name in ["user_id"]:
        df[f"{id_name}_count"] = df.groupby(id_name).cumcount()

    df[f"user_content_id_count"] = df.groupby(["user_id", "content_id"]).cumcount()

    # ---------------
    # one_hot_encoding_count
    # ---------------
    for id_name in ["user_id"]:
        for col in ["answered_correctly", "not_answered"]:
            print(col)
            df = one_hot_encoding_count(df=df,
                                        id_name=id_name,
                                        column=col)
