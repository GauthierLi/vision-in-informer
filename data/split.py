from dataclasses import replace
import os
import shutil
import random

import numpy as np

lp = os.getcwd()
def mkfolder(fold_name):
    if os.path.isdir(fold_name):
        shutil.rmtree(fold_name)
    os.mkdir(fold_name)
        

def split(path, split_rate = 0.2):
    data_list = list(os.listdir(path))
    mkfolder("train_data")
    mkfolder("valid_data")
    
    lenth = len(data_list)
    name_list = ["acc_" + name.split(".")[0].split("g")[-1] +
                         "_{}".format(lenth) for name in data_list]

    valid_data = np.random.choice(data_list, int(split_rate * lenth), replace=False)
    train_data = [item for item in data_list if item not in valid_data]

    for item in valid_data:
        shutil.copy(os.path.join("cwt_raw_data", item),
             os.path.join("valid_data", "acc_" + item.split(".")[0].split("g")[-1] + "_{}.jpg".format(lenth)))
    for item in train_data:
        shutil.copy(os.path.join("cwt_raw_data", item),
             os.path.join("train_data", "acc_" + item.split(".")[0].split("g")[-1] + "_{}.jpg".format(lenth)))


split(os.path.join(lp, "cwt_raw_data"))