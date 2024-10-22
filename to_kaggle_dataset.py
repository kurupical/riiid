import shutil
import os
import glob

if os.path.isdir("riiid_code"):
    shutil.rmtree("riiid_code")
os.makedirs("riiid_code/pretrained/model302")

for f in glob.glob("output/model302/20210107150104/*"):
    if "csv" in f:
        continue
    if os.path.isdir(f):
        shutil.copytree(f, f"riiid_code/pretrained/model302/{os.path.basename(f)}")
    else:
        shutil.copyfile(f, f"riiid_code/pretrained/model302/{os.path.basename(f)}")


shutil.copytree("experiment", "riiid_code/experiment/")
shutil.rmtree("riiid_code/experiment/mlruns")
os.makedirs("riiid_code/feature_engineering")
for f in glob.glob("feature_engineering/*"):
    if ".pickle" not in f and os.path.isfile(f):
        shutil.copy(f, "riiid_code/feature_engineering/")
shutil.copytree("model", "riiid_code/model/")
# shutil.copytree("pipeline", "riiid_code/pipeline/")