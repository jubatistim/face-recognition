# Convolutional Neural Network

### Importing the libraries

from keras.preprocessing.image import ImageDataGenerator
import calendar
import time
import os
import keras

###############################
## Part 1 - Data Preprocessing
###############################

### Generating images for the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

### Generating images for the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

### Creating the Training set
training_set = train_datagen.flow_from_directory('CNN/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

### Creating the Test set
test_set = test_datagen.flow_from_directory('CNN/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

###############################
## Part 2 - Building the CNN
###############################

### Initialising the CNN
cnn = keras.models.Sequential()

### Step 1 - Convolution
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

### Step 2 - Pooling
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

### Adding a second convolutional layer
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# cnn.add(keras.layers.Dropout(0.25))

### Step 3 - Flattening
cnn.add(keras.layers.Flatten())

# cnn.add(keras.layers.Dropout(0.5))

### Step 4 - Full Connection
cnn.add(keras.layers.Dense(units=128, activation='relu'))

# cnn.add(keras.layers.Dropout(0.5))

### Step 5 - Output Layer
cnn.add(keras.layers.Dense(units=2, activation='softmax')) # find probability of each class, outputs between 0 and 1 for each class that represents the probability

### Summarize the network
cnn.summary()

###############################
## Part 3 - Training the CNN
###############################

### Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


### Training the CNN on the Training set and evaluating it on the Test set
# steps_per_epoch: ceil(num_samples / batch_size)
# num samples training = 1873
# num samples validation/test = 445
# cnn.fit(training_set,
#         steps_per_epoch = 50,
#         epochs = 100,
#         validation_data = test_set,
#         validation_steps = 12)

cnn.fit(training_set,
        epochs = 10,
        validation_data = test_set)

### Save the model
cnn.save(os.path.join('./saved-models', 'cnn' + str(calendar.timegm(time.gmtime())) + '.h5'))