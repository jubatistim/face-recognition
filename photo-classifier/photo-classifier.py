import cv2
import os
import calendar
import time
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as image_utils

# Padding for recognized faces
SCALEY = 60
SCALEX = 60
MINSIZE = 64
 
# load model
model = tf.keras.models.load_model('../neural-network/saved-models/cnn1589854703.h5')

# get running path
base_dir = os.path.dirname(__file__)

# Status counters
total_files = 0
current_file = 0

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('../extractor/cascades/haarcascade_frontalface_default.xml')

# Count total files for status
for filename in os.listdir('images'):
    total_files = total_files + 1

print('-----------------STARTED-----------------')

# Main Loop
for filename in os.listdir('images'):

    # Update status
    current_file = current_file + 1
    print('Image ' + str(current_file) + ' of ' + str(total_files))

    # read current image file
    img = cv2.imread(os.path.join(base_dir, 'images', filename))

    # pre processing
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    img_original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.10, 8)

    # save detected faces
    for (x, y, w, h) in faces:

        if w > MINSIZE and h > MINSIZE:

            try:
                crop_img = img_original[y-SCALEY:y+h+SCALEY, x-SCALEX:x+w+SCALEX]

                crop_img = cv2.resize(crop_img, (64, 64), interpolation = cv2.INTER_AREA)

                test_image = image_utils.img_to_array(crop_img)
                test_image = np.expand_dims(test_image, axis = 0)
                
                # validate
                result = model.predict_on_batch(test_image)

                who = ''

                if result[0][0] == 0:
                    who = 'JULIANO'
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
                    cv2.putText(img, who, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
                else:
                    who = 'LUNA'
                    cv2.rectangle(img, (x, y), (x+w, y+h), (191, 0, 255), 4)
                    cv2.putText(img, who, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (191, 0, 255), 4)

            except:
                print('Error')

    cv2.imwrite(os.path.join(base_dir, 'images_conv', str(calendar.timegm(time.gmtime())) + '.jpg'), img)

print('-----------------FINISHED-----------------')