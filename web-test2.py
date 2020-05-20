import base64
import json
import requests
import cv2
import numpy as np
from PIL import Image
from io import StringIO

image = open("./photo-class-images/a (9).jpg", 'rb') #open binary file in read mode 
# <class '_io.BufferedReader'>
image_read = image.read()
# <class 'bytes'>
image_64_encode = base64.encodebytes(image_read)

data = {
  "image": image_64_encode.decode()
}

json_data = json.dumps(data)

response = requests.post('http://127.0.0.1:5000/predict', json_data)

image_predicted = response.json()['image_predicted']

im_bytes = base64.b64decode(image_predicted)
im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
img = cv2.imdecode(im_arr, flags=cv2.COLOR_RGB2GRAY)

# THIS WORKS TO CHANGE IMAGE COLOR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#OTHER POSSIBILITIES FOR COLORS CONVERSIONS
# COLOR_BGR2GRAY
# COLOR_RGB2BGR
# IMREAD_COLOR
# COLOR_RGB2GRAY

# img = cv2.resize(img, (int(img.shape[1]/7), int(img.shape[0]/7)))

cv2.imshow('Predicted Image', img)
cv2.waitKey()