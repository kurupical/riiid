import shutil
import os
import glob

if os.path.isdir("riiid_code"):
    shutil.rmtree("riiid_code")
os.makedirs("riiid_code/pretrained")

for f in glob.glob("output/ex_011/20201016131524/*"):
    if "csv" in f:
        continue
    else:
        shutil.copyfile(f, f"riiid_code/pretrained/{os.path.basename(f)}")
shutil.copytree("experiment", "riiid_code/experiment/")
shutil.rmtree("riiid_code/experiment/mlruns")
shutil.copytree("feature_engineering", "riiid_code/feature_engineering/")
shutil.copytree("model", "riiid_code/model/")
shutil.copytree("pipeline", "riiid_code/pipeline/")

