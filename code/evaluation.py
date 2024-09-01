import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from save_model import curr_time
## Evaluation of training data
def evaluation_training_testing(lstm_estimator,X_train,y_train,X_test,y_test,device):
  print('Training data...')
  y_pred_train = lstm_estimator.predict(X_train)
  mse_train = mean_squared_error(y_train, y_pred_train)
  print(f'Mean Squared Error (MSE): {mse_train}')
  
  r2_train = 1-((1-r2_score(y_train, y_pred_train))*(len(y_train)-1))/(len(y_train)-X_train.shape[-1]-1)
  print(f'adjusted R-squared (R2) Score: {r2_train}\n')

  print('Test data...')
  y_pred_test = lstm_estimator.predict(X_test)
  mse_test = mean_squared_error(y_test, y_pred_test)
  print(f'Mean Squared Error (MSE): {mse_test}')
  r2_test = 1-((1-r2_score(y_test, y_pred_test))*(len(y_test)-1))/(len(y_test)-X_test.shape[-1]-1)

  # r2_test = r2_score(y_test, y_pred_test)
  print(f'Adjusted R-squared (R2) Score: {r2_test}\n')
  
  var=['mse_train','mse_test','r2_train','r2_test']
  output=[mse_train,mse_test,r2_train,r2_test]
  result=pd.DataFrame({'metrics':var,'result':output})
  
  result.to_csv(f"./result_device{device}_{curr_time()}.csv")
  plot_training_testing(y_train,y_pred_train,y_test,y_pred_test,device)
  return y_pred_train,y_pred_test,mse_train,mse_test,r2_train,r2_test

def plot_training_testing(y_train,y_pred_train,y_test,y_pred_test,device):
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(y_train, label='Actual')
  plt.plot(y_pred_train, label='Predicted')
  plt.xlabel('Cycle')
  plt.ylabel('RUL')
  plt.title(f'Training Data {device}')
  plt.legend()
  plt.subplot(2, 1, 2)
  plt.plot(y_test, label='Actual')
  plt.plot(y_pred_test, label='Predicted')
  plt.xlabel('Cycle')
  plt.ylabel('RUL')
  plt.title(f'Testing Data {device}')
  plt.legend()
  plt.tight_layout()
  plt.savefig(f'./result_device{device}_{curr_time()}.png')
  plt.show()
  
  
