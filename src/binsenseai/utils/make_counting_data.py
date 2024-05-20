import os
import json
import unicodedata
import numpy as np
import json
import random

# loading metaata
random_train = "random_train.txt"
random_val = "random_val.txt"
metadata_file = "metadata.json"

hard = False

def get_quantity(idx):
    quantity = 0
    #print(metadata,len(metadata),idx)
    if metadata[idx]:
        quantity = metadata[idx]['EXPECTED_QUANTITY']
    return quantity

def get_moderate_list(split_file):
    print("loading random split file")
    train_list = []
    linenum = 0
    with open(split_file) as f:
        for line in f.readlines():
            idx = int(line)-1
            quantity = get_quantity(linenum)
            if quantity > 0 and quantity < 6:#12345
                train_list.append([idx,quantity]) 
            linenum+=1        
    return train_list 

if __name__ == "__main__":
    print("loading metadata!")
    with open(metadata_file) as json_file:
        metadata = json.load(json_file)
    N = len(metadata)
    train_list = get_moderate_list(random_train)
    val_list = get_moderate_list(random_val)
    print("dumping train and val sets into json file")
    out_fname = 'counting_train.json'
    with open(out_fname,'wt') as f:
        json.dump(train_list,f)

    print("counting_val_hard | counting_val.json")
    out_fname = 'counting_val.json'
    with open(out_fname,'wt') as f:
        json.dump(val_list,f)

