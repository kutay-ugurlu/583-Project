import os
from glob import glob 

def str_checker(a):
    return (True and (a.endswith("loso.py") or False))


all_pys = glob("*py")
all_pys = list(filter(str_checker,all_pys))

os.system("conda activate deepmlp")

for py in all_pys:
    command = "python " + py
    os.system(command)


