from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import math
import os
from glob import glob
from scipy import stats as s

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Load the base model and weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))
#defining the model architecture
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
# loading the trained weights
model.load_weights("models/weight_80_36000.hdf5")
# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Model saved with Keras model.save()
# MODEL_PATH = 'models/weight_80_36000.hdf5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def predict_if_safe(path):
    images = glob("uploads/*")

    try:
        prediction_images = []
        for i in range(len(images)):
            img = image.load_img(images[i], target_size=(80, 80, 3))
            print(images[i])
            img = image.img_to_array(img)
            img = img / 255
            prediction_images.append(img)

        # converting all the frames for a test video into numpy array
        prediction_images = np.array(prediction_images)
        # extracting features using pre-trained model
        prediction_images = base_model.predict(prediction_images)
        # converting features in one dimensional array
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 2 * 2 * 512)
        # predicting tags for each array
        predictionar = model.predict_classes(prediction_images)

        # creating the tags
        train = pd.read_csv('train_new.csv')
        y = train['class']
        y = pd.get_dummies(y)
        res = y.columns.values[predictionar][0]
    except:
        res = "Please upload a valid file"

    files = glob('uploads\*')
    for f in files:
        os.remove(f)

    return str(res)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the uploaded file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Call the prediction function
        result = predict_if_safe(file_path)

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

