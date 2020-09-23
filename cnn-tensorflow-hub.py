import tensorflow as tf

# print('****************************************************')
# print(tf.__version__)
# print('****************************************************')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow_hub as hub

base_model = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x3/1", input_shape=(256,256,3))

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    base_model,
    layers.Dense(3072, activation='relu'),
    layers.Dense(1536, activation='relu'),
    layers.Dense(768, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax'),
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('CNN/train',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('CNN/validation',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = model.fit(x=training_set, validation_data = validation_set, epochs=10, verbose=2)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.legend(['train'], loc='upper left')
plt.show()