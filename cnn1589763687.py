## Load the model back
import tensorflow as tf
import numpy as np
from keras.preprocessing import image as image_utils
 
# load model
model = tf.keras.models.load_model('./saved-models/cnn1589763687.h5')

# Model cnn1589763687.h5
#
# Model: "sequential" 
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 64, 64, 32)        896
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 32, 32, 32)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0
# _________________________________________________________________
# flatten (Flatten)            (None, 8192)              0
# _________________________________________________________________
# dense (Dense)                (None, 128)               1048704
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 129
# =================================================================
# Total params: 1,058,977
# Trainable params: 1,058,977
# Non-trainable params: 0
# _________________________________________________________________


# get image
# test_image = image_utils.load_img('./CNN/validation/JU/1589607094.jpg', target_size = (64, 64)) #0
# test_image = image_utils.load_img('./CNN/validation/JU/1589656882.jpg', target_size = (64, 64)) #0
# test_image = image_utils.load_img('./CNN/validation/JU/1589658942.jpg', target_size = (64, 64)) #0
# test_image = image_utils.load_img('./CNN/validation/JU/1589758806.jpg', target_size = (64, 64)) #0
# test_image = image_utils.load_img('./CNN/validation/JU/1589759474.jpg', target_size = (64, 64)) #0

# test_image = image_utils.load_img('./CNN/validation/LUNA/1589606808.jpg', target_size = (64, 64)) #1
# test_image = image_utils.load_img('./CNN/validation/LUNA/1589660417.jpg', target_size = (64, 64)) #1
# test_image = image_utils.load_img('./CNN/validation/LUNA/1589660471.jpg', target_size = (64, 64)) #0
# test_image = image_utils.load_img('./CNN/validation/LUNA/1589667384.jpg', target_size = (64, 64)) #1
test_image = image_utils.load_img('./CNN/validation/LUNA/1589667544.jpg', target_size = (64, 64)) #1

test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# validate
result = model.predict_on_batch(test_image)

if result[0][0] == 0:
    print('JU')
else:
    print('LUNA')