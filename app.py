import cv2
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()
#import cv2
#from google.colab.patches import cv2_imshow
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("adult_flask.pkl", "rb"))
    print("start prediction")
    result = loaded_model.predict(to_predict)
    return result[0]
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print("request.form.to_dict(): ", to_predict_list)
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)  
        print("got result: ", result);
        if int(result)== 1:
            prediction ='Income more than 50K'
        else:
            prediction ='Income less that 50K'           
        return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD']=True
    


