import pandas as pd
import numpy as np

from data_preprocessing import  average_downsampling, standard_scaling, ema_smoothing, fourier_transform,lag
from data_processing import target_data_creation,create_merge_df,dataset_insample_tensorflow,dataset_outsample_tensorflow
from model import create_ensemble_model,fit_ensemble_model,custom_loss_kera
from evaluation import evaluation_training_testing,plot_training_testing
from save_model import save_keras_model
from config_loader import load_config



def process_data():
    config=load_config()
    lst=config['train_data']['raw_data']
    spans=config['train_data']['span']
    filter_lst=config['train_data']['filter']
    data_path=config['train_data']['train_pkl_path']
    # step 01: loading pickle data
    df_Device2=pd.read_pickle(f'{data_path}/{lst[0]}')
    df_Device3=pd.read_pickle(f'{data_path}/{lst[1]}')
    df_Device4=pd.read_pickle(f'{data_path}/{lst[2]}')
    df_Device5=pd.read_pickle(f'{data_path}/{lst[3]}')


    # step 02: Average downsampling data
    df_Device2_processed=pd.DataFrame()
    df_Device3_processed=pd.DataFrame()
    df_Device4_processed=pd.DataFrame()
    df_Device5_processed=pd.DataFrame()
    
    df_Device2_processed['VCE_avg'] = average_downsampling(df_Device2['collectorEmitterVoltage'])
    df_Device3_processed['VCE_avg'] = average_downsampling(df_Device3['collectorEmitterVoltage'])
    df_Device4_processed['VCE_avg'] = average_downsampling(df_Device4['collectorEmitterVoltage'])
    df_Device5_processed['VCE_avg'] = average_downsampling(df_Device5['collectorEmitterVoltage'])

    
    
    
        
    # step 03: Standardization
    df_Device2_processed['VCE_avg_std']=standard_scaling(df_Device2_processed['VCE_avg'])
    df_Device3_processed['VCE_avg_std']=standard_scaling(df_Device3_processed['VCE_avg'])
    df_Device4_processed['VCE_avg_std']=standard_scaling(df_Device4_processed['VCE_avg'])
    df_Device5_processed['VCE_avg_std']=standard_scaling(df_Device5_processed['VCE_avg'])
    

    # step 04: Exponential Moving Average (EMA)
    processed_dfs=[df_Device2_processed,df_Device3_processed,df_Device4_processed,df_Device5_processed]
    for df in processed_dfs:
        for span in spans:
            df[f'VCE_EMA_{span}'] = ema_smoothing(df['VCE_avg_std'], span)
            
    # step 05: Fourier Transform
    for df in processed_dfs:
        df['VCE_fft_std'] = fourier_transform(df['VCE_avg_std'])
        
# #    step 06: lag transfrom 
#     lag_value=3
#     for df in processed_dfs:
#         df[f'VCE_lag_{lag_value}_std'] = lag(df['VCE_avg_std'],lag_value).bfill()
        
    
    # step 07: Filtering
    for i, df in enumerate(processed_dfs):
        df = df.iloc[:filter_lst[i], :]
    
    #cycle
    for df in processed_dfs:
        df['cycle']=target_data_creation(df.index)
    
    # merge_dataframe
    
    df_Device234=create_merge_df([df_Device2_processed,df_Device3_processed,df_Device4_processed])
    df_Device345=create_merge_df([df_Device3_processed,df_Device4_processed,df_Device5_processed])
    df_Device235=create_merge_df([df_Device2_processed,df_Device3_processed,df_Device5_processed])
    df_Device245=create_merge_df([df_Device2_processed,df_Device4_processed,df_Device5_processed])
    
    return processed_dfs,[df_Device345,df_Device245,df_Device235,df_Device234]

def run_model(data=None):
    config=load_config()
    datatype=config['train_data']['datatype']
    device=config['train_data']['device']
    para=config['train_data']['para']
    loss=config['train_data']['loss']
    epochs=config['train_data']['epochs']
    batch=config['train_data']['batch']
    if data==None:
        processed_dfs,merged_df=process_data()
    else:
        processed_dfs,merged_df=data
    if device==2:
        final_data=processed_dfs[0]
        final_merge_data=merged_df[0]
    elif device==3:
        final_data=processed_dfs[1]
        final_merge_data=merged_df[1]
    elif device==4:
        final_data=processed_dfs[2]
        final_merge_data=merged_df[2]
    elif device==5:
        final_data=processed_dfs[3]
        final_merge_data=merged_df[3]    
        
    if datatype=='insample':
        X_train,y_train,X_test,y_test=dataset_insample_tensorflow(final_data.iloc[:,1:-1],final_data.iloc[:,-1])
    
    if datatype=='outsample':
        X_train,y_train,X_test,y_test=dataset_outsample_tensorflow(final_merge_data.iloc[:,2:-1],final_merge_data.iloc[:,-1],final_data.iloc[:,2:-1],final_data.iloc[:,-1])
        
    model=create_ensemble_model(X_train.shape[1:], lstm_units=80,para=para, loss=loss)
    model.summary()
    
    
    hist=fit_ensemble_model(model, X_train, y_train, epochs, batch)
    save_keras_model(model)
    output=evaluation_training_testing(model,X_train,y_train,X_test,y_test,device)
    print(X_train.shape)
    
    
## --------------------------------------------running model---------------------------------##

run_model()


        
    





    

    
        
    
            
    
    
    
        
        
    


