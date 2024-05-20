#  data preprocessing pipeline for training and validation data preparation for object verification.


import os
import json
import unicodedata
import numpy as np
import json
import random

# loading metaata and instances
# loading metaata and instances
random_train = "random_train.txt"
random_val = "random_val.txt"
metadata_file = "metadata.json"
instance_file = "instances.json"

hard = False

# get quantity given index
def get_expected_quantity(idx):
    quantity = 0
    if metadata[idx]:
        quantity = metadata[idx]['EXPECTED_QUANTITY']
    return quantity

# True or False array
# True or False array
""" def get_candidates(split_file):
    print("loading random split file")
    candidates = np.zeros(N,bool)
    with open(split_file) as f:
        for line in f.readlines():
            idx = int(line)-1
            quantity = get_expected_quantity(idx)
            if quantity > 0:
                candidates[idx] = True
            
            # if hard:
            #     if quantity > 0:
            #         candidates[idx] = True
            # else:
            #     if  quantity < 10:
            #         candidates[idx] = True
    return candidates 

 """    
def get_candidates(split_file):
    print("loading random split file")
    linenum = 0
    candidates = np.zeros(N,bool)
    with open(split_file) as f:
        for line in f.readlines():
            idx = int(line)-1 # check the content of the line
            quantity = get_expected_quantity(linenum)
            if quantity > 0:
                    candidates[linenum] = True
            linenum = linenum +1
    return candidates

def generate_random_number(exclude_number):
    number = random.randint(1, 10)
    while number == exclude_number:
        number = random.randint(1, 10)
    return number


# This function generates a split train list. It iterates through each image and for each image, 
# iterates through the objects in the bin. For each object, it get the actual quantity present in the bin ,
# and get a  list of metadata_index  of training sets that contain the ASIN 
# making split train list in below format
# [ metadata_index, ASIN,  ground truth quantity  , [ list of metadata_index  of training sets that contain the ASIN ]  ]


def get_train_list(train_candidates):
    train_list = []
    for idx in range(N):
        if train_candidates[idx]:
            bin_info = metadata[idx]['BIN_FCSKU_DATA']
            bin_keys = list(bin_info.keys())
            object_list = []
            # iterate over objects in the bin
            for j in range(0,len(bin_info)):
                asin = bin_info[bin_keys[j]]['asin']
                
                # # Convert dict_keys object to a list
                # bin_keys_list = list(bin_keys)

                # # Access the first key-value pair from bin_info
                # first_key = bin_keys_list[0]
                # asin = bin_info[first_key]['asin']
                
                if asin==None:
                    continue
                actual_quantity = bin_info[bin_keys[j]]['quantity']
                repeat_in = 0
                target_idx_list = []
                # how many this object repeted in the candidate images?
                for target_idx in instances[asin]['bin_list']:
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        repeat_in = repeat_in + 1
                        target_idx_list.append(target_idx)
                # if repeat > 1, then at least twice showed up, we can make a
                # positive pair with this object
                # if repeat_in > 1:
                #     object_list.append([target_idx_list])
                train_list.append([idx,asin, actual_quantity,target_idx_list])
    return train_list


# Tthis function generates a validation list for positive pairs by iterating through each image, 
# extracting information about objects in the bin, and collecting indices of images where each object appears. 
# It then creates positive pairs between the current image and the target images where the object appears and 
# adds them to the validation list.
# positive = means ground truth quantity == input_quantity

# making split val list(for positive pair)
# [   metadata_index, ASIN, positive  (1 ) , ground truth quantity , input_quantity , [ list of metadata_index  of training sets that contain the ASIN ]  ]


def get_val_pos_list(train_candidates, val_candidates):
   
    val_list = []
   
    for idx in range(N):
        
        if val_candidates[idx]:
            
            bin_info = metadata[idx]['BIN_FCSKU_DATA']
            bin_keys = list(bin_info.keys())
            
            # iterate over objects in the bin
            for j in range(0,len(bin_info)):
                
                asin = bin_info[bin_keys[j]]['asin']
                actual_quantity = bin_info[bin_keys[j]]['quantity']
                if asin==None:
                    continue
                
                target_idx_list = []
                
                for target_idx in instances[asin]['bin_list']:
                    
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        target_idx_list.append(target_idx)
                        
                if target_idx_list:
                    val_list.append([idx, asin, 1, actual_quantity, actual_quantity , target_idx_list])
                    
    return val_list 



# this function generates a validation list for negative pairs by randomly selecting an ASIN, 
# retrieving instances of that ASIN from the instances dictionary, and collecting indices 
# of images where the same object appears in the training set. It then creates negative pairs 
# between the current image and the target images where the same object appears and adds them to the validation list.
# negative = means ground truth quantity != input_quantity
# making split val list(for negative pair)
# [  metadata_index, ASIN,  negative ( 0) , ground truth quantity , input_quantity , [ list of metadata_index  of training sets that contain the ASIN ]  ]

def get_val_neg_list(train_candidates, val_candidates):
    val_list = []
   
    
    for idx in range(N):
        if val_candidates[idx]:
            # pick up random object asin from the list
            bin_info = metadata[idx]['BIN_FCSKU_DATA']
            bin_keys = list(bin_info.keys())
            
            # iterate over objects in the bin
            for j in range(0,len(bin_info)):
                
                asin = bin_info[bin_keys[j]]['asin']
                actual_quantity = bin_info[bin_keys[j]]['quantity']
                if asin==None:
                    continue
                
                target_idx_list = []
                
                for target_idx in instances[asin]['bin_list']:
                    
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        target_idx_list.append(target_idx)
                        
                if target_idx_list:
                    val_list.append([idx, asin, 0, actual_quantity, generate_random_number(actual_quantity) , target_idx_list])
                    
                
    return val_list


# The code loads metadata and instance data from JSON files, initializes some variables, gets candidates for training and validation sets, 
# builds training and validation sets, samples positive and negative samples for validation, shuffles the validation list, and 
# finally dumps the training and validation sets into JSON files.



if __name__ == "__main__":
    
    print("loading metadata!")
    with open(metadata_file) as json_file:
        metadata = json.load(json_file)
        
    print("loading instance data!")
    with open(instance_file) as json_file:
        instances = json.load(json_file)
        
    instance_keys = instances.keys()
    N = len(metadata)
    N_inst = len(instance_keys)

    train_candidates = get_candidates(random_train)
    val_candidates = get_candidates(random_val)

    # building training sets
    train_list = get_train_list(train_candidates)

    # building validataion sets
    val_pos_list = get_val_pos_list(train_candidates, val_candidates)
    val_neg_list = get_val_neg_list(train_candidates, val_candidates)
    
    pos_samples = random.sample(val_pos_list,len(val_pos_list))
    neg_samples = random.sample(val_neg_list,len(val_neg_list))
    
    val_list = pos_samples + neg_samples
    random.shuffle(val_list)
    
    print("dumping train and val sets into json file")
    
    out_fname = 'obj_num_verification_train.json'
    with open(out_fname,'w') as f: ## wb to w
        json.dump(train_list,f)

    out_fname = 'obj_num_verification_val.json'
    with open(out_fname,'w') as f: ## wb to w
        json.dump(val_list,f)