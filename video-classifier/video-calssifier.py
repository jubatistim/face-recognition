import cv2
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as image_utils

# Padding for recognized faces
SCALEY = 60
SCALEX = 60
MINSIZE = 64
 
# load model
model = tf.keras.models.load_model('../neural-network/saved-models/cnn1589763687.h5')

# Face reconition classifier https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('../extractor/cascades/haarcascade_frontalface_default.xml')

# Define capture device
video=cv2.VideoCapture(0)

while True:
   
    # Capture a frame
    check, frame = video.read()

    # read current image file
    img = frame.copy()

    # pre processing
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

    # Show realtime captures
    cv2.imshow("Color Frame",frame)
    cv2.imshow("Recognized Realtime",img)

    # Key to end capture session
    key=cv2.waitKey(1)

    # Finish capture session
    if key==ord('q'):
        break

# release videos and windows
video.release()
cv2.destroyAllWindows
