import numpy as np
import cv2
from flask import Flask, request, render_template

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import boto3

app = Flask(__name__)

model_root=""
data_root=""

@app.route('/')
def home():
    print("fencing home1")
    url = "http://s3.amazonaws.com/mymodel-heroku/model_final.pth"
    BUCKET="mymodel-heroku"
    model = download_file("model_final.pth", BUCKET)
    print("got model")
    return render_template('index.html')

def download_file(file_name, bucket):
    """
    Function to download a given file from an S3 bucket
    """
    s3 = boto3.resource('s3')
    output = f"downloads/{file_name}"
    s3.Bucket(bucket).download_file(file_name, output)

    return output

def GetKeypointsFromPredictor():
       
    config_file_path = model_root+"config.yml"

    weights_path = model_root+"model_final.pth"

    image_path = data_root+"fencing.jpg"

    model = config_file_path
    im = cv2.imread(image_path)
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    keypoints = outputs["instances"].pred_keypoints
    print("keypoints: ", keypoints)
    return keypoints

# prediction function
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST': 
        result = 1 
        print("got result: ", result);
        if int(result)== 1:
            prediction ='Income more than 50K'
        else:
            prediction ='Income less that 50K'           
        return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD']=True
    


