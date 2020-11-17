import pandas as pd
import numpy as np



data_types_dict = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
}


df = pd.read_csv("../input/riiid-test-answer-prediction/train.csv",
                 dtype=data_types_dict)

df_question = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv",
                          dtype={"bundle_id": "int32",
                                 "question_id": "int32",
                                 "correct_answer": "int8",
                                 "part": "int8"})
df_lecture = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv",
                         dtype={"lecture_id": "int32",
                                "tag": "int16",
                                "part": "int8"})

div_num = 10
df[f"user_id_div{div_num}"] = df["user_id"]%div_num

for user_id, w_df in df.groupby(f"user_id_div{div_num}"):
    print(len(w_df))
    w_df.drop([f"user_id_div{div_num}", "row_id"], axis=1).to_pickle(f"../input/riiid-test-answer-prediction/split10/train_{user_id}_base.pickle")
    w_df1 = pd.merge(w_df[w_df["content_type_id"]==0], df_question, how="left", left_on="content_id", right_on="question_id")
    w_df2 = pd.merge(w_df[w_df["content_type_id"]==1], df_lecture, how="left", left_on="content_id", right_on="lecture_id")
    w_df = pd.concat([w_df1, w_df2])
    w_df["tag"] = w_df["tag"].fillna(-1).astype("int16")
    w_df["correct_answer"] = w_df["correct_answer"].fillna(-1).astype("int8")
    w_df["bundle_id"] = w_df["bundle_id"].fillna(-1).astype("int32")
    print(len(w_df))
    w_df = w_df.drop(["question_id", "lecture_id"], axis=1)
    w_df.drop([f"user_id_div{div_num}", "row_id"], axis=1).to_pickle(f"../input/riiid-test-answer-prediction/split10/train_{user_id}.pickle")
