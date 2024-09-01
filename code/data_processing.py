import pandas as pd
import numpy as np
from config_loader import load_config

def target_data_creation(data):
    return 1-data/len(data)


def create_merge_df(df_lst):
    """
    Merges a list of dataframes by extracting specified columns and applying a transformation to the 'cycle' column.
    
    Args:
        df_lst (list of pd.DataFrame): List of dataframes to merge.
        columns (list of str): List of columns to extract from each dataframe.
        cycle_transform (function): Function to transform the 'cycle' column. Default is 1 - x/n.
        
    Returns:
        pd.DataFrame: The merged dataframe.
    """
    dfs = []
    columns=df_lst[0].columns
    for df in df_lst:
        n = df.shape[0]
        temp_df = pd.DataFrame({col: df[col] for col in columns})
        # temp_df['cycle'] = cycle_transform(df.index, n)
        dfs.append(temp_df)

    return pd.concat(dfs, ignore_index=True)

# merged_df = create_merge_df(df_lst, columns_to_extract)
# print(merged_df)

# creating dataset for lstm and RNN model

def dataset_insample_tensorflow(X,y):
    config=load_config()
    seq_length=config['train_data']['seq_length']
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(X.shape[0] - seq_length):
        if (i + 1) % 5 == 0:
            X_test.append(X[i:i + seq_length])
            y_test.append(y[i])
        else:
            X_train.append(X[i:i + seq_length])
            y_train.append(y[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train,y_train, X_test,y_test

def dataset_outsample_tensorflow(X_tr,y_tr,X_te,y_te):
    config=load_config()
    seq_length=config['train_data']['seq_length']
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # Training data
    for i in range(len(X_tr) - seq_length):
            X_train.append(X_tr[i:i + seq_length])
            y_train.append(y_tr[i + seq_length])
    # testing data
    for i in range(len(X_te) - seq_length):
            X_test.append(X_te[i:i + seq_length])
            y_test.append(y_te[i + seq_length])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
