import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
IMG_SIZE = 24


def load_eye_model():
    json_file = open(os.path.join('saved_models', 'eye_model', 'eye_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join('saved_models', 'eye_model',"eye_model.h5"))
    return loaded_model

def load_gender_model():
    json_file = open(os.path.join('saved_models', 'gender_model', 'age_gender.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join('saved_models', 'gender_model',"age_gender.h5"))
    return loaded_model

def load_emotion_model():
    json_file = open(os.path.join('saved_models', 'emotion_model' , 'emotion_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join('saved_models', 'emotion_model', 'emotion_model.h5'))
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
