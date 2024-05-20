import os
from os import listdir
import os.path
import numpy as np
import random

img_dir= "/home/ubuntu/abid/dataset/amazon_bin_images/" # read from s3
meta_dir = "/home/ubuntu/abid/dataset/amazon_bin_metadata/" # read from s3

img_list = listdir(img_dir)
N = len(img_list)

json_list = listdir(meta_dir)
list_random = list(range(N)) # bms error TODO remove list
random.shuffle(list_random)

# finding images that metadata exists
meta_avail = np.zeros(N, dtype=bool)
listIds= [0] * N#np.zeros(N, dtype=np.str)
for i in range(N):
    json_path = json_list[i]
    json_path_joined = os.path.join(meta_dir,json_path)
    if os.path.isfile(json_path_joined):
        meta_avail[i] = True
        listIds[i] = json_path.split('.')[0]

# assign validataion set
valset = np.zeros(N, dtype=bool)
n_valset = int(round(N*0.1))
count = 0
random.shuffle(list_random)
for i in range(N):
    idx = list_random[i]
    if meta_avail[idx]:
        valset[idx]=True
        count = count + 1
        if count == n_valset:
            break

# writing out to textfile
train_f = open('random_train.txt','w')
val_f = open('random_val.txt','w')
for i in range(N):
    if meta_avail[i]:
        if valset[i]:
            val_f.write("\n"+listIds[i])
        else:
            train_f.write("\n"+listIds[i])
train_f.close()
val_f.close()

