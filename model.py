from keras.models import model_from_json
from PIL import Image
from imageio import imread
import numpy as np
#from scipy.misc import imresize, imsave
IMG_SIZE = 24


def load_eye_model():
    json_file = open('eye_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("eye_model.h5")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    return loaded_model


def predict_eye(img, model):
    img = Image.fromarray(img, 'RGB').convert('L')
    # img = imresize(img, (IMG_SIZE, IMG_SIZE)).astype('float32')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    img /= 255
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    if prediction < 0.1:
        prediction = 'closed'
    elif prediction > 0.9:
        prediction = 'open'
    else:
        prediction = 'idk'
    return prediction
