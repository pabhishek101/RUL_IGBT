from save_model import load_keras_model,curr_time
import numpy as np
import pandas as pd
import joblib
from config_loader import load_config




def dataset(X_tr):
    config=load_config()
    seq_length=config['test_data']['seq_length']
    X_train = []
    # Training data
    for i in range(len(X_tr) - seq_length):
        X_train.append(X_tr[i:i + seq_length])
    return np.array(X_train)


def process_data(data):
    config=load_config()
    span=config['test_data']['span']
    path=config['save_model']['std_scaler']
    seq_length=config['test_data']['seq_length']
    use_col=config['test_data']['use_col']
    loaded_scaler = joblib.load(path)
    model=load_keras_model()
    
    final_data=[]
    if len(data)<seq_length:
        raise ValueError("Data length must be at least {}".format(seq_length))
    else:
        scaled_data=loaded_scaler.transform(data.reshape(-1,1))
        new_data=pd.DataFrame()
        new_data['avg_std']=scaled_data.squeeze()
        
        for i in range(len(span)):
            new_data[i]=new_data['avg_std'].ewm(span=span[i]).mean()
        new_data['fft_std']=np.fft.fft(new_data['avg_std'])
        # print(new_data.columns)
        final_data=dataset(new_data[use_col])
        
        return model.predict(final_data).flatten()
    
#--------------------main----------------------------#
data=[1.88504574, 1.97993109, 1.98068909, 2.01690959, 2.01695459,
       2.03186709, 2.03125859, 2.03099909, 2.02992709, 2.03038809,
       2.02963059, 2.01864809, 2.01776409, 2.03452809, 2.03503959,
       2.04308059, 2.04310659, 2.05559758, 2.05478908, 2.06174408,
       2.06157458, 2.06504908, 2.06637258, 2.97338399, 2.97478749,
       2.97038349, 2.97283249, 2.97154599, 2.97451249, 2.96482649,
       2.96499249, 2.96003399, 2.96139199, 2.99997199, 2.99992299,
       3.01762149, 3.01659349, 3.02210198, 3.02270398, 3.46294094,
       3.46266094, 3.50092244, 3.50145994, 3.50216994, 3.49932594,
       3.49211494, 3.49420444, 3.51480943, 3.51776543, 3.97016089,
       3.96840539, 3.99957688, 3.99995388, 3.99829588, 3.99826138,
       3.99181889, 3.99288989, 4.38318035, 4.36894835, 4.50025333,
       4.50334833, 4.49556333, 4.49483733, 4.49134983, 4.49457733,
       4.48802983, 4.48497934, 4.50867683, 4.50874833, 4.50060983,
       4.49892033, 4.44801984, 4.44815884, 4.50850633, 4.51084983,
       4.50073383, 4.50137483, 4.95275279, 4.95269179, 4.99408828]
output=pd.Series(process_data(np.array(data)))
output.to_csv(f'./result/test_result_{curr_time()}.csv')
output.head()