# import keras

# # load model
# model = keras.models.load_model('./saved-models/cnn1590263939.h5')

# model_json = model.to_json()
# with open('./saved-models/LunaJuV0.json', 'w') as json_file:
#     json_file.write(model_json)
# model.save_weights('./saved-models/LunaJuV0.h5')

# from keras.models import model_from_json

# json_file = open('./saved-models/LunaJuV0.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights('./saved-models/LunaJuV0.h5')

# loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# loaded_model.summary()

import keras
model = keras.models.load_model('./saved-models/cnn1590263939.h5')

model.summary()