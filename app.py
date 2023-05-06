from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = "C:/Users/aruna/Downloads/deep-learning-pneumonia-detection-web-app-master/deep-learning-pneumonia-detection-web-app-master/models/cnn.h5"

#Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, color_mode='grayscale',target_size=(50, 50)) #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make a prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        if preds > 0.5:
            prediction_result = 'Pneumonia'
        else:
            prediction_result = 'Normal'

        # Return prediction result to the HTML page
        return render_template('index.html', prediction_result=prediction_result, uploaded_file=f.filename)

    return None
        

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run()
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
