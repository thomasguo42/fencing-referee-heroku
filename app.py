import numpy as np
import cv2

from flask import Flask, render_template, request, redirect, url_for
import os, json, boto3
from PIL import Image
import base64
from io import BytesIO
from utils import *

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)

model_root=""
data_root=""

@app.route('/')
def home():
    print("fencing home1")
    url = "http://s3.amazonaws.com/mymodel-heroku/model_final.pth"
    BUCKET="mymodel-heroku"
    #model = download_file("model_final.pth", BUCKET)
    print("got model")
    return render_template('index.html')

def download_file(file_name, bucket):
    """
    Function to download a given file from an S3 bucket
    """
    s3 = boto3.resource('s3')
    output = f"./{file_name}"
    s3.Bucket(bucket).download_file(file_name, output)
    #getKeypointsFromPredictor(output)
    return output

def getKeypointsFromPredictor(weights_path, im):
       
    config_file_path = model_root+"config.yml"

    #weights_path = model_root+"model_final.pth"

    image_path = data_root+"fencing.jpg"

    model = config_file_path
    #im = cv2.imread(image_path)
    cfg = get_cfg()
    cfg.merge_from_file(config_file_path)
    cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    keypoints = "keypoints" #outputs["instances"].pred_keypoints
    print("keypoints: ", keypoints)
    return keypoints

# Listen for GET requests to yourdomain.com/account/
@app.route("/account/")
def account():
  # Show the account-edit HTML page:
  return render_template('account.html')

@app.route('/upload')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      print("got the file: ", f)
      video_list = [f]
      image_list = createImagesFromVideos(video_list)
      url = "http://s3.amazonaws.com/mymodel-heroku/model_final.pth"
      BUCKET="mymodel-heroku"
      model = download_file("model_final.pth", BUCKET)
      print("got model")
      getKeypointsFromPredictor(model, image_list[0])

      #f.save(secure_filename(f.filename))
      #img = Image.open(image_list[0].stream)
      #rgb_img = img.convert('RGB')
      with BytesIO() as buf:
        #rgb_img.save(buf, 'jpeg')
        pil_im = Image.fromarray(image_list[0])
        pil_im.save(buf, 'jpeg')
	
        image_bytes = buf.getvalue()
        encoded_string = base64.b64encode(image_bytes).decode()         
      return render_template('upload.html', img_data=encoded_string), 200
   else:
     return render_template('upload.html', img_data=""), 200
     # return 'file uploaded successfully'
    
# Listen for POST requests to yourdomain.com/submit_form/
@app.route("/submit-form/", methods = ["POST"])
def submit_form():
  # Collect the data posted from the HTML form in account.html:
  username = request.form["username"]
  full_name = request.form["full-name"]
  avatar_url = request.form["avatar-url"]

  # Provide some procedure for storing the new details
  #update_account(username, full_name, avatar_url)

  # Redirect to the user's profile page, if appropriate
  print("submit upload")
  return render_template('index.html') #redirect(url_for('profile'))


# Listen for GET requests to yourdomain.com/sign_s3/
#
# Please see https://gist.github.com/RyanBalfanz/f07d827a4818fda0db81 for an example using
# Python 3 for this view.
@app.route('/sign-s3/')
def sign_s3():
  print("run sign_s3()")
  # Load necessary information into the application
  S3_BUCKET = os.environ.get('S3_BUCKET')
  print("S3_BUCKET: ", S3_BUCKET)
  # Load required data from the request
  file_name = request.args.get('file-name')
  file_type = request.args.get('file-type')

  print("start upload_file")
  #upload_file(file_name, S3_BUCKET)
  print("end upload_file")
    
  # Initialise the S3 client
  s3 = boto3.client('s3')

  # Generate and return the presigned URL
  presigned_post = s3.generate_presigned_post(
    Bucket = S3_BUCKET,
    Key = file_name,
    Fields = {"acl": "public-read", "Content-Type": file_type},
    Conditions = [
      {"acl": "public-read"},
      {"Content-Type": file_type}
    ],
    ExpiresIn = 3600
  )
  print("presigned_post: ", presigned_post)
  # Return the data to the client
  return json.dumps({
    'data': presigned_post,
    'url': 'https://%s.s3.amazonaws.com/%s' % (S3_BUCKET, file_name)
  })
    
import logging
from botocore.exceptions import ClientError
import os


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

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
    


