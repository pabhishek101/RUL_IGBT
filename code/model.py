import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D,RepeatVector, Flatten, TimeDistributed,GRU
from tensorflow.keras import backend as K,metrics

def create_lstm_model(input_shape, lstm_units=80):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=False, input_shape=input_shape))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def create_gru_model(input_shape, gru_units=80):
    model = Sequential()
    model.add(GRU(gru_units,activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(GRU(10,return_sequences=True))
    model.add(GRU(1))
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add((Conv1D(filters=64, kernel_size=9, activation='relu', input_shape=input_shape)))
    model.add((MaxPooling1D(pool_size=2)))
    model.add((Flatten()))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(80, activation='relu',return_sequences=False))
    model.add((Dense(10)))
    model.add(Dense(1))
    return model

def create_ensemble_model(input_shape, lstm_units=80,para=(0.1,1,0.1), loss='mean_squared_error'):
    model_lstm = create_lstm_model(input_shape, lstm_units)
    # model_cnn = create_cnn_model(input_shape)
    model_gru= create_gru_model(input_shape)
    
    models = [model_lstm, model_gru]
    model_input = tf.keras.Input(shape=input_shape)
    model_outputs = [model(model_input) for model in models]
    ensemble_output = tf.keras.layers.Average()(model_outputs)
    ensemble_model = Model(inputs=model_input, outputs=ensemble_output)
    
    ensemble_model.compile(optimizer="Adam", loss=loss, metrics=[metrics.MeanAbsoluteError()])
    ensemble_model.summary()
    
    return ensemble_model

def fit_ensemble_model(model, X, y, epochs=100, batch_size=10):
    return model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

def custom_loss_kera(para):
    alpha,beta,gamma=para
    def loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))  # MSE part
        pinn_term1 = K.mean(K.relu(y_pred[1:] - y_pred[:-1]))  # Consider normalization later
        pinn_term2 = K.mean(K.square(K.relu(-y_pred))) + K.mean(K.square(K.relu(y_pred - 1)))
        return (1 - alpha) * mse + alpha * gamma * pinn_term1 + beta * pinn_term2
        # return mse+beta*pinn_term2
    return loss
