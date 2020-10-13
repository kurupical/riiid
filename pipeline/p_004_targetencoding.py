import pandas as pd
from feature_engineering.all_id import one_hot_encoding_count, target_encoding
from feature_engineering.user_id import user_intelligency

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
            print(f"target_encoding {col}")
            df = one_hot_encoding_count(df=df,
                                        id_name=id_name,
                                        column=col)
    for col in [["user_id"], ["content_id"], ["task_container_id"]]:
        print(f"target_encoding {col}")
        df = target_encoding(df=df, cols=col)

    print("user_intelligency")
    df = user_intelligency(df)
    df = df.drop(["not_answered", "prior_question_had_explanation", "prior_question_elapsed_time"], axis=1)
    return df