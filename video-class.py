import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as image_utils
import io
import base64
from PIL import Image

from custom_util import *
 
# load model
model = tf.keras.models.load_model('./saved-models/cnn1589763687.h5')

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# Define capture device
video=cv2.VideoCapture(0)

while True:
   
    # Capture a frame
    check, frame = video.read()

    # read current image file
    img_return = frame.copy()

    #cv2 image to string base 64 encoded
    _, buffer = cv2.imencode('.jpg', img_return)
    image_read = base64.b64encode(buffer)

    # string base 64 encoded to cv2 image - THIS CHANGES COLOR OF THE IMAGE AND THE RECOGNITION LOOKS BETTER - I DON'T KNOW WHY IT CHANGES IMAGE COLOR
    decoded = base64.b64decode(image_read)
    nimage = Image.open(io.BytesIO(decoded))
    img_source = np.array(nimage)

    img_predicted = predict_Luna_Ju(img_source, img_return, model)

    # Show realtime captures
    cv2.imshow("Color Frame",frame)
    cv2.imshow("Recognized Realtime",img_predicted)

    # Key to end capture session
    key=cv2.waitKey(1)

    # Finish capture session
    if key==ord('q'):
        break

# release videos and windows
video.release()
cv2.destroyAllWindows
