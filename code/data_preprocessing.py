import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from save_model import save_stdscaler
from config_loader import load_config



## read .pkl file
def load_pickle_file(file_path):
    """
    Loads data from a pickle file.
    Args:
        file_path (str): Path to the .pkl file.
    Returns:
        object: The deserialized Python object.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
# loaded_data = load_pickle_file('your_file.pkl')





# *****----------------**********Average downsampling*********---------------****


def optimize_df(data):
    # Calculate half length for all rows at once
    half_len = (data.str.len() // 2).values

    # Split the data into two halves using vectorized indexing
    first_half = data.apply(lambda x: x[:half_len[0]])
    second_half = data.apply(lambda x: x[half_len[0]:])

    # Calculate averages for both halves using vectorized operations
    avg_first_half = first_half.apply(np.mean)
    avg_second_half = second_half.apply(np.mean)

    # Create a new column with the average values
    new_data = list(zip(avg_first_half,avg_second_half))
    return new_data

def flattend_data(data):
  array=data[0]
  for i in range(1,len(data)):
    array=np.append(array,data[i][0])
    array=np.append(array,data[i][1])

  return array


def average_downsampling(data):
    return flattend_data(optimize_df(data))




#Standardization
def standard_scaling(data):
    scaler = StandardScaler()
    scaled_data=scaler.fit_transform(data.values.reshape(-1, 1))
    save_stdscaler(scaler)
    return scaled_data




#Exponential moving average (EMA-5, EMA-10, EMA-15,EMA-20)

def ema_smoothing(data,span_):
    return data.ewm(span=span_).mean()
    





# Applying Fourier Transform on Standardized data.
def fourier_transform(data):
    return np.abs(np.fft.fft(data))

def lag(data,lag=2):
    return data.shift(lag)

def filtering(df,x):
    return df.iloc[:x,:]




