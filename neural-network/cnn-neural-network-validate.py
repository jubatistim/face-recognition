## Load the model back
import tensorflow as tf
import numpy as np
 
# load model
model = tf.keras.models.load_model('./saved-models/cnn1589763687.h5')

model.summary()

import cv2

img = cv2.imread('./CNN/validation/JU/1589607094.jpg')

y_pred = model.predict(img)

print(y_pred)

# from keras.preprocessing.image import ImageDataGenerator

# val_datagen = ImageDataGenerator(rescale = 1./255)

# val_set = val_datagen.flow_from_directory('CNN/validation', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# y_pred = model.predict(val_set)

# print(np.round(y_pred, 0))