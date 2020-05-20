import numpy as np
import base64
import io

from PIL import Image
import cv2

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from flask import request
from flask import jsonify
from flask import Flask

from custom_util import *

app = Flask(__name__)

def getModel():
    global model
    model = tf.keras.models.load_model('./saved-models/cnn1589763687.h5')

print("------Loading Keras Model------")
getModel()

@app.route("/predict", methods = ["POST"])
def predict():

    message = request.get_json(force = True)
    encoded = message['image']

    # string base 64 encoded to cv2 image - THIS CHANGES COLOR OF THE IMAGE AND THE RECOGNITION LOOKS BETTER - I DON'T KNOW WHY IT CHANGES IMAGE COLOR
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    img_source = np.array(image)

    # string base 64 encoded to cv2 image - THIS DON'T CHANGE THE IMAGE COLORS, SO FOR NOW I WILL USE THAT TO USE AS BACKGROUND TO DRAW RECOGNITION, BUT NOT WILL BE INPUTED TO THE CNN
    im_bytes_return = base64.b64decode(encoded)
    im_arr_return = np.frombuffer(im_bytes_return, dtype=np.uint8)
    img_return = cv2.imdecode(im_arr_return, flags=cv2.IMREAD_COLOR)

    img = predict_Luna_Ju(img_source, img_return, model)

    _, im_arr = cv2.imencode('.jpg', img)
    im_bytes = im_arr.tobytes()
    image_predicted = base64.b64encode(im_bytes)

    response = {
        'image_predicted': image_predicted.decode()
    }

    return jsonify(response)