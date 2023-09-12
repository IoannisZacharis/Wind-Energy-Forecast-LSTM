import pandas as pd 
import numpy as np 
import tensorflow as tf
from tensorflow import _keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.optimizers import RMSprop

day = 60*60*24
year = 365.2425*day
#ADDING THE PATH OF THE SAVED DATA 
model_path:str=r"C:\Users\Dell\Desktop\sqlproject"
filepath :str= r"C:\Users\Dell\Desktop\batteryData\GenPrice Data.xlsx"


#CREATING AN INPUT ARRAY WITH SIZE:WINDOW_SIZE
#AND OUTPUT ARRAY WITH SIZE ONE 
def df_to_X_Y(df, window_size=5):
  df_as_np = df.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]] 
    X.append(row) # Input: X = [[,...,], [,...,] ...]
    label = df_as_np[i+window_size][0]
    y.append(label) #Output: Y = [[],[],... ]
  return np.array(X), np.array(y)

#THIS FUNCTION HELP THE MODEL TO CONVERGE FASTER
#AND MAKE THEM LESS SENSITIVE TO THE SCALE OF INPUT FEATURES
def preprocess(X):
  X[:, :, 0] = (X[:, :, 0] - wind_training_mean) / wind_training_std
  return X


#THIS FUNCTION IS USED TO ENCODE CYCLIC PATTERNS IN TIME SERIES DATA
def data_structure(wind_df:pd.DataFrame):
  wind_df.index=wind_df.pop("Datetime")
  wind_df['Seconds'] = wind_df.index.map(pd.Timestamp.timestamp)
  #INTODUCING NEW TIMESERIES RELATIONSHIPS
  wind_df['Day sin'] = np.sin(wind_df['Seconds'] * (2* np.pi / day))
  wind_df['Day cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / day))
  wind_df['Year sin'] = np.sin(wind_df['Seconds'] * (2 * np.pi / year))
  wind_df['Year cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / year))
  return wind_df.drop('Seconds', axis=1)


#CALLING THE DATA FROM THE FILEPATH AND CREATING THE FORMAT WE WANT 
df=pd.read_excel(filepath)
df=df.iloc[:,0:2]
X2, Y2 = df_to_X_Y(data_structure(df)) # X2: ARRAY = [[,...,],...], Y2:ARRAY = [[]]

#SEPARATING THE DATA INTO TRAIN , VALIDATE AND TEST SLICES
X2_train, Y2_train = X2[:39000], Y2[:39000]
X2_val, Y2_val = X2[39000:42000], Y2[39000:42000]
#X2_test, y2_test = X2[40000:], y2[40000:]

wind_training_mean = np.mean(X2_train[:, :, 0])
wind_training_std = np.std(X2_train[:, :, 0])
                           

preprocess(X2_train)
preprocess(X2_val)
#preprocess(X2_test)

#CREATING THE NN-MODEL
#SETTING THE  MODEL PARAMETERS
model = Sequential()
model.add(InputLayer((5, 5)))
model.add(LSTM(64))
#model.add(Dense(64,"relu"))
#model.add(Dense(32,"relu"))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))
model.summary()


cp4 = ModelCheckpoint('model/', save_best_only=True)
model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.01), metrics=['RootMeanSquaredError'])
model.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val),batch_size=64, epochs=40, callbacks=[cp4])
"""
model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.001), metrics=['RootMeanSquaredError'])
model.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val),batch_size=32, epochs=20, callbacks=[cp4])
"""
model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.001), metrics=['RootMeanSquaredError'])
model.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val),batch_size=32, epochs=20, callbacks=[cp4])
model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.0001), metrics=['RootMeanSquaredError'])
model.fit(X2_train, Y2_train, validation_data=(X2_val, Y2_val),batch_size=32, epochs=10, callbacks=[cp4])



#SAVING THE MODEL AFTER BEING TRAINED
model.save(model_path + str(r"\wind_model.h5"))

