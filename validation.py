import json
import random
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request
from src.binsenseai.pipeline.predictions import PredictionPipeline
import os 
import ast



# app = Flask(__name__) # initializing a flask app

# @app.route('/',methods=['GET'])  # route to display the home page
# def homePage():
#     return render_template("index.html")


# @app.route('/train',methods=['GET'])  # route to train the pipeline
# def training():
#     os.system("python main.py")
#     return "Training Successful!" 

obj = PredictionPipeline()

# input_list = request.form['input_list'] # ["08887.jpg", [["B00CRABRSU",1], ["B00DV8AJLS",1], ["B00H95M72S",1]]]
#             # Convert the string to a list of lists
# data = ast.literal_eval(input_list)
# predict = obj.predict(data)

input_list = ["08887.jpg", [["B00CRABRSU",1], ["B00DV8AJLS",1], ["B00H95M72S", 1]]] # request.form['input_list'] #
predict = obj.predict(input_list) 
#data = ast.literal_eval(input_list)
#predict = obj.predict(data)

print(predict)

# @app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
# def index():
#     if request.method == 'POST':
#         try:
#             obj = PredictionPipeline()
#             input_list = request.form['input_list'] #["08887.jpg", [["B00CRABRSU",1], ["B00DV8AJLS",1], ["B00H95M72S", 1]]] 
#             predict = obj.predict(input_list) 
#             return render_template('results.html', prediction = str(predict))

#         except Exception as e:
#             print('The Exception message is: ',e)
#             raise Exception("Sorry, image1 not loaded correctly")
#     else:
#         return render_template('index.html')

""" @app.route('/predict_bin',methods=['POST']) # route to show the predictions in a web UI
def handle_post_request():
    if request.method == 'POST':
        try:
            obj = PredictionPipeline()
            bin_data = request.data
            input_list = json.loads(bin_data)
            predict = obj.predict(input_list)
            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            raise Exception("Sorry, image1 not loaded correctly")
    else:
        return render_template('index.html') """

# if __name__ == "__main__":
# 	# app.run(host="0.0.0.0", port = 8080, debug=True)
# 	app.run(host="0.0.0.0", port = 8080)