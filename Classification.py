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

def filecheck():
    five_seconds_ago = datetime.now() - timedelta(seconds=5)
    filetime = datetime.fromtimestamp(path.getmtime(r"Cell_Image.png"))

    if filetime > five_seconds_ago:
        return True 

model = tf.keras.models.load_model(r"C:\Users\bahls\Desktop\Hack The Valley\My_Model.h5")

class_names = ["0normal" , "1verylittle" , "2moderate" , "3severe"]

img_height = 180 
img_width = 180

Cell_Image = r"Cell_Image.png"


img = keras.preprocessing.image.load_img(
Cell_Image, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_cell = (class_names[np.argmax(score)])

print(predicted_cell)
