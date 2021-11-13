from flask import Flask, flash, request, redirect, url_for, render_template
import os 
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import fileinput
import sys

from os import path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from datetime import datetime, timedelta

UPLOAD_FOLDER = r"C:\Users\Pranav Kukreja\Desktop\Flask\uploads"
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def filecheck():
    five_seconds_ago = datetime.now() - timedelta(seconds=5)
    filetime = datetime.fromtimestamp(path.getmtime(r"uploads\Cell_Image2.png"))

    if filetime > five_seconds_ago:
        return True 

model = tf.keras.models.load_model(r"C:\Users\Pranav Kukreja\Desktop\Flask\My_Model.h5")

class_names = ["0normal" , "1verylittle" , "2moderate" , "3severe"]

img_height = 180 
img_width = 180

Cell_Image2 = r"C:\Users\Pranav Kukreja\Desktop\Flask\uploads\Cell_Image2.png"

@app.route("/home")
def home():
    return render_template('home.html', name='home')

@app.route('/DetectionTool', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = keras.preprocessing.image.load_img(
            Cell_Image2, target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            predicted_cell = (class_names[np.argmax(score)])
            return render_template('detectiontool.html', name='detection' , variable = predicted_cell)
    return render_template('detectiontool.html', name='detection' )

@app.route("/patient-history")
def patienthistory():

    img = keras.preprocessing.image.load_img(
    Cell_Image2, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_cell = (class_names[np.argmax(score)])
    return render_template('PatientHistory.html', variable = predicted_cell)


if __name__ == '__main__':
    app.run(debug=True)

