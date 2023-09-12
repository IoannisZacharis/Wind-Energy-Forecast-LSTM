import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

number_of_predictions = 20

day = 60*60*24
year = 365.2425*day
#ADDING THE PATH OF THE SAVED DATA
model_path:str=r"C:\Users\Dell\Desktop\sqlproject"
filepath :str= r"C:\Users\Dell\Desktop\batteryData\GenPrice Data.xlsx"

#THIS FUNCTION IS USED TO ENCODE CYCLIC PATTERNS IN TIME SERIES DATA
def data_structure(wind_df:pd.DataFrame,TheLastValue):
  if TheLastValue:#IF TRUE TAKE THE LAST VALUE OF THE DATAFRAME 
    wind_df['Seconds'] = wind_df.index.map(pd.Timestamp.timestamp)
    wind_df['Day sin'].iloc[-1] = np.sin(wind_df['Seconds'].iloc[-1] * (2* np.pi / day))
    wind_df['Day cos'].iloc[-1] = np.cos(wind_df['Seconds'].iloc[-1] * (2 * np.pi / day))
    wind_df['Year sin'].iloc[-1] = np.sin(wind_df['Seconds'].iloc[-1] * (2 * np.pi / year))
    wind_df['Year cos'].iloc[-1] = np.cos(wind_df['Seconds'].iloc[-1] * (2 * np.pi / year))
    #return wind_df.drop('Seconds', axis=1)
  else:#ELSE CREATE THE DATAFRAME
    wind_df.index=wind_df.pop("Datetime")
    wind_df['Seconds'] = wind_df.index.map(pd.Timestamp.timestamp)
    #INTODUCING NEW TIMESERIES RELATIONSHIPS
    wind_df['Day sin'] = np.sin(wind_df['Seconds'] * (2* np.pi / day))
    wind_df['Day cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / day))
    wind_df['Year sin'] = np.sin(wind_df['Seconds'] * (2 * np.pi / year))
    wind_df['Year cos'] = np.cos(wind_df['Seconds'] * (2 * np.pi / year))
  return wind_df.drop('Seconds', axis=1)


#THIS FUNCTION HELP THE MODEL TO CONVERGE FASTER
#AND MAKE THEM LESS SENSITIVE TO THE SCALE OF INPUT FEATURES
def preprocess(X):
  X[:, :, 0] = (X[:, :, 0] - wind_training_mean) / wind_training_std
  return X

#################
#PREPARATION STEP FOR TIME SERIES FORECASTING TASKS
#CREATES OVERLAPPING WINDOWS OF DATA FROM THE TIME SERIES
#EACH WINDOW IS USED AS AN INPUT FEATURE
def df_to_X_Y(df, window_size=5):
  X = []
  y = []
  df_as_np = df.to_numpy()
  for i in range(len(df_as_np)-window_size):
    row = [r for r in df_as_np[i:i+window_size]] 
    X.append(row) # Input: X = [[,...,], [,...,] ...]
    label = df_as_np[i+window_size][0]
    y.append(label) #Output: Y = [[],[],... ]
  return np.array(X), np.array(y)


#THE RESULTING ARRAY IS USED AS INPUT FEATURE FOR THE MODEL
def df_to_x(df, size):
  df_as_np=df.to_numpy()
  x=[]
  row = [r for r in df_as_np[len(df_as_np)-size:len(df_as_np)]]
  x.append(row)
  return (np.array(x)) 

#PREDICTS FUTURE WIND VALUES,  
#THE PREDICTIONS STORED IN A DICTIONARY, 
#ALSO THE MISSING ANGLE VALUES ARE FILLED BASED ON THE LAST AVAILABLE VALUE
def predict_future_wind_values(model, df: pd.DataFrame, number_of_predictions: int):
    predictions = {}
    for _ in range(number_of_predictions):
        x = df_to_x(df, 5)#CREATING THE RIGHT SHAPE OF INPUT (6,5)
        x = preprocess(x)
        y = float(model.predict(x))#PRODUCE THE OUTPUT THROUGH THE TRAINED MODEL
        new_index = df.index[-1] + pd.Timedelta(hours=1)#CREATING THE NEW INDEX OF THE OF THE NEW PREDICTED VALUE   
        new_row = {'Generation in MWh': y}
        df = df._append(pd.DataFrame([new_row], index=[new_index]))#APPENDING THE NEW DATA
        predictions[new_index] = [y]#SAVING THE DATETIME AS KEY AND THE PREDICTED VALUE AS VALUE TO THE PREDICTION DICTIONARY
        df = data_structure(df, True)#FILLING THE NaN VALUES OF THE ANGLE COLUMNS 
    return df, predictions
#PREPARING THE DATA 
loaded_model = tf.keras.models.load_model(model_path + str("\wind_model.h5"))
df=pd.read_excel(filepath)
wind_df=df.iloc[:,0:2]
wind_df = data_structure(wind_df,False)
X2, Y2 = df_to_X_Y(wind_df) # X2: ARRAY = [[,...,],...], Y2:ARRAY = [[]]

#SEPARATING THE DATA INTO TRAIN , VALIDATE
X2_train, Y2_train = X2[:39000], Y2[:39000]
X2_val, Y2_val = X2[39000:42000], Y2[39000:42000]

wind_training_mean = np.mean(X2_train[:, :, 0])
wind_training_std = np.std(X2_train[:, :, 0])

# GENERATE PREDICTIONS FOR THE NEXT 'number_of_predictions' TIME STEPS
# USING THE LOADED MACHINE LEARNING MODEL 'loaded_model'.
# WE EXCLUDE THE LAST 'number_of_predictions' ROWS FROM 'wind_df' AS WE WILL PREDICT THOSE.
new_wind_df,pred=predict_future_wind_values(loaded_model,wind_df.iloc[:-number_of_predictions],number_of_predictions)

# CREATE A DATAFRAME 'prediction_values' TO STORE THE PREDICTED 'Wind Generation in MWh' VALUES.
# THIS DATAFRAME HAS DATETIME INDICES.
prediction_values = pd.DataFrame.from_dict(pred, orient='index', columns=['Wind Generation in MWh']) 


plt.plot(prediction_values["Wind Generation in MWh"])
plt.plot(wind_df["Generation in MWh"].iloc[-number_of_predictions:])
plt.show()
print(prediction_values,"\n\n\n -----------------------")
print(new_wind_df.iloc[-number_of_predictions:])








