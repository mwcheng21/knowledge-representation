import os

#change these
newdir = "tiny" #name of new directory
n = 10 #number of lines to copy

os.system(f"mkdir ./data/{newdir}")

for types in ["eval", "test", "train"]:
    os.system(f"mkdir ./data/{newdir}/{types}")
files = ["data.buggy_only",  "data.commit_msg",  "data.fixed_only",  "data.full_code_fullGraph",  "data.full_code_leaveOnly"]
for file in files:
    for types in ["eval", "test", "train"]:
        oldpath = os.path.join("data", "medium", types, file)
        newpath = os.path.join("data", newdir, types, file)
        os.system(f"head -n {n} {oldpath} > {newpath}")