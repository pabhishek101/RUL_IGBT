import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from config_loader import load_config

def curr_time():
    config=load_config()
    format_=config['save_model']['time_format']
    now = datetime.now()
    return now.strftime(format_)

from joblib import dump

def save_keras_model(model):
    config=load_config()
    model_name=config['save_model']['model_name']
    path=config['save_model']['path']
    model.save(f'{path}/{model_name}.h5')
    print(f'your model has been saved: {path}/{model_name}.h5')


def load_keras_model():
    config=load_config()
    model_name=config['save_model']['model_name']
    path=config['save_model']['path']
    loaded_model=keras.models.load_model(f'{path}/{model_name}.h5')
    return loaded_model

# Save the scaler to a file
def save_stdscaler(scaler):
    config=load_config()
    path=config['save_model']['std_scaler']
    dump(scaler, path)
