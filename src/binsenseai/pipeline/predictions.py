## def predicte() point to verify_object_presence in app.py
import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

import json
import random
import numpy as np
import pandas as pd
import torch
from src.binsenseai.components.object_verification.verify_object_presence import verify_object_presence
from src.binsenseai.components.object_quantity_verification.verify_item_quantity import verify_item_quantity
from flask import Flask, render_template, request
import os 


class PredictionPipeline:
    def __init__(self):
        True
    
    def predict(self, input_list):
        
        output_list = []
        input_image = input_list[0]
        file_instance_json = "./artifacts/data_transformation/instances.json"
        
        with open(file_instance_json, 'r') as f:
            data = json.load(f)
       
        #print(input_list[1])
        for item_list in input_list[1]:
            verified_item_list = []
            asin = item_list[0]
            item_quantity = item_list[1]
            
            item_name = data[asin]['name']
            #print(item_name)
            verified_item_list.append(item_name)
            verified_item_list.append(asin)
            verified_item_list.append(item_quantity)
            
            ## get an image from the dataset for comparasion in the siamese network
            other_images = data[asin]['bin_image_list']
       
            
            if len(other_images)> 1 :
                object_image =  random.choice([other_image for other_image in other_images if other_image != input_image]) 
                
            # select a random image if there is no image in the dataset that contains the item
            else : 
                for asin_, asin_values in data.items():
                    img_list = asin_values['bin_image_list']
                    if isinstance(img_list, list) and len(img_list) > 0:
                        object_image = random.choice(img_list)         

            ## call verify_object_presence API 
           
            item_present_in_image = verify_object_presence(input_image, object_image)
            # returning yes / no
            #print(f"\n check if  item {asin}  present in the image : {item_present_in_image}")
        
            if item_present_in_image =='yes' :
                # call verify_item_quantity  API
                
                if verify_item_quantity(input_image, asin,item_quantity):
                    #print(f'Item quantity verified successfully for ASIN {asin}')
                    result = True
                else:
                    #print(f'Item quantity verification failed for ASIN {asin}')
                    result = False
                
            else:
                result = False
                
            verified_item_list.append(result)
            output_list.append(verified_item_list) 
          
            #print(f"\n Return Output to UI ", output_list)    
            
        return output_list

