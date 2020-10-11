from datetime import datetime as dt
from pipeline.p_001_baseline import transform
import pandas as pd
import glob
import os
import tqdm
import lightgbm as lgb
import pickle
import riiideducation

def run(debug,
        model_dir):

    # environment
    env = riiideducation.make_env()

    # model loading
    models = []
    for model_path in glob.glob(f"{model_dir}/*model*.pickle"):
        with open(model_path, "rb") as f:
            models.append(pickle.load(f))

    # data preprocessing
    data_dir = "../work_csv"
    os.makedirs(data_dir, exist_ok=True)

    """
    for model_id, fname in enumerate(glob.glob("../input/riiid-test-answer-prediction/split10/*")):
        df = pd.read_pickle(fname)
        if debug:
            df = df.head(1000)
        df = transform(df)

        for user_id, w_df in tqdm.tqdm(df.groupby("user_id")):
            w_df.to_pickle(f"{data_dir}/{user_id}.pickle")
    """

    iter_test = env.iter_test()
    for (df_test, df_sample_prediction) in iter_test:
        print("a")

if __name__ == "__main__":
    run(debug=False,
        model_dir="../output/ex_001/20201011201811")