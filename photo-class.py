import cv2
import os
import calendar
import time
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
from keras.preprocessing import image as image_utils

from custom_util import *
 
# load model
model = tf.keras.models.load_model('./saved-models/cnn1589854703.h5')

# Status counters
total_files = 0
current_file = 0

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# Count total files for status
for filename in os.listdir('photo-class-images'):
    total_files = total_files + 1

print('-----------------STARTED-----------------')

# Main Loop
for filename in os.listdir('photo-class-images'):

    # Update status
    current_file = current_file + 1
    print('Image ' + str(current_file) + ' of ' + str(total_files))

    currentPath = os.path.join('photo-class-images', filename)

    # read current image file
    img_return = cv2.imread(currentPath)

    img_predicted = predict_Luna_Ju(img_return, model, 1000, 64, 50)

    cv2.imwrite(os.path.join('photo-class-images-conv', str(calendar.timegm(time.gmtime())) + '.jpg'), img_predicted)

    time.sleep(1.5)

print('-----------------FINISHED-----------------')