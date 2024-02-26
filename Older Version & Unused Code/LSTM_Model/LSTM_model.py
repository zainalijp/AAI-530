#Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from math import sqrt

#Path to data folder storing ExtraSensory data
#Dataset is stored locally in folder one layer above where this project is saved locally
data_folder = '../ExtraSensory.per_uuid_features_labels'

'''
Since dataset is actually 60 separate .csv files with poor naming schema in one folder,
will use glob to find all files matching the .csv.gz pattern in the data folder
'''
all_files = glob.glob(os.path.join(data_folder, "*.csv.gz"))

#List to store the 60 users individual data (each a separate .csv) as a collective, singular dataframe
data_frames_list = []

#For Loop to iterate over list of files
for file in all_files:
    #Extract user ID from files (Based on assumption it's present, it is not)
    user_id = os.path.basename(file).split('.')[0]
    #Read csv file into dataframe
    df = pd.read_csv(file, compression='gzip')
    #Adding a user_id column to dataframe since it starts with 'timestamp' feature
    df['user_id'] = user_id
    data_frames_list.append(df)

#Concatenate all the dataframes from the empty list into a grand, singular dataframe
df_all = pd.concat(data_frames_list, ignore_index=True)

#Reference check for concatenation, message in terminal
print("Concatenation complete, the dataframe shape is: ", df_all.shape)

#Percentange of null values in newly created dataframe
null_percentage = (df_all.isnull().sum() / len(df_all)) * 100
#Sorting percentages in order from highest to lowest
null_percentage_sorted = null_percentage.sort_values(ascending=False)
#Print via loop function to see each feature and associated percentage
for feature, percentage in null_percentage.items():
    print(f'{feature}: {percentage:.2f}%')

'''
IMPUTATION SECTION
Since we will focus our LSTM on over 40+ features, with prediction on the Lying_Down, Sitting, 
Walking, and Standing labels, which all have varying data missing (between ~ 0.8% and ~75% missing data, 
we will apply linear interpolation according to timestamp to maintain underlying relationships of time series data. 
'''
#Line Break for easier terminal view
print("\n\nImputation Starts Here. \n") 


#Sensor data of focus from 279 total features, determined via correlation and inference from how data was recorded and produced
#We will apply linear interpolation on the following data
columns_to_interpolate = ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:OR_standing',
    'lf_measurements:screen_brightness', 'discrete:time_of_day:between0and6', 'label:LOC_home', 
    'watch_acceleration:3d:mean_y', 'audio_naive:mfcc12:mean', 'label:WITH_FRIENDS', 'location:min_altitude',
    'location_quick_features:std_lat',
    'audio_naive:mfcc1:mean','discrete:battery_state:is_full', 
    'audio_naive:mfcc1:std', 'location:log_diameter', 'audio_naive:mfcc10:mean', 'watch_acceleration:magnitude_stats:time_entropy', 
    'audio_naive:mfcc6:mean', 'audio_naive:mfcc3:std', 'audio_naive:mfcc2:std', 'audio_naive:mfcc8:mean', 
    'audio_naive:mfcc9:mean', 'raw_magnet:3d:ro_yz', 
    'audio_naive:mfcc6:std', 'audio_naive:mfcc4:std', 'audio_naive:mfcc11:mean', 
    'audio_naive:mfcc8:std', 'audio_naive:mfcc9:std', 'location:max_altitude',
    'watch_acceleration:3d:std_z', 'audio_naive:mfcc7:std', 'discrete:time_of_day:between3and9', 'label:OR_indoors', 
    'label:OR_outside', 'raw_acc:magnitude_spectrum:log_energy_band2', 'raw_acc:magnitude_stats:percentile25', 
    'raw_acc:magnitude_stats:std', 'raw_acc:magnitude_stats:percentile75', 'proc_gyro:magnitude_stats:percentile50', 
    'proc_gyro:magnitude_stats:percentile75', 'proc_gyro:magnitude_stats:mean', 'raw_acc:magnitude_spectrum:log_energy_band3', 
    'raw_acc:magnitude_stats:time_entropy', 'proc_gyro:magnitude_stats:percentile25', 'raw_acc:magnitude_spectrum:log_energy_band4', 
    'label:BATHING_-_SHOWER']

#Empty DataFrame for transformed data, preserve old dataframe just in case
df_ip = pd.DataFrame(index=df_all.index)

#Apply innterpolation to each column with 'timestamp' as reference 
for column in columns_to_interpolate:
    if column != 'timestamp':
        #Drop instances where timestamp/columns are NaN
        valid_data = df_all.dropna(subset=[column, 'timestamp'])
        #Get timestamp and instance values for interpolation
        x = valid_data['timestamp']
        y = valid_data[column]
        #Interpolation function (can be changed to different types)
        func = interp1d(x, y, kind='linear', fill_value='extrapolate', bounds_error=False)
        df_ip[column] = func(df_all['timestamp'])

#Double check that 'timestamp' is in the interpolated dataframe
df_ip['timestamp'] = df_all['timestamp']

#Check Interpolation via NaN values present (Re-Use of Above Code)
#Percentange of null values per summary
null_percentage_ip = (df_ip.isnull().sum() / len(df_ip)) * 100
#Sorting percentages in order from highest to lowest
null_percentage_sorted = null_percentage_ip.sort_values(ascending=False)
#Print Individually in loop so each feature is printed with percentage
for feature, percentage in null_percentage_ip.items():
    print(f'{feature}: {percentage:.2f}%')

'''
Things to consider: df_ip is now (377346, 44) due to reduced features which are all correlated to our labels
'''

#DATA PREPROCESSING
#Removing the label data from the overall features to prevent any form of data leakage
df_ip_copy = df_ip.copy()
df_clean = df_ip_copy.drop(columns=['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:OR_standing'])

#Selected features (sensor and timeseries data) and labels (features to predict) for model training
features = df_clean.iloc[:, :44]
labels = df_ip[['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:OR_standing']]

#Re-organizing dataframe by 'timestamp' feature so it is in correct chronological order (preserve timeseries relationship)
df_ip = df_ip.sort_values(by='timestamp', ascending=True)

#Split sorted data into training and testing sets with 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Data heavily varying, reduce feature range to values between 0 and 1 while preserving relationship
scaler = MinMaxScaler(feature_range=(0,1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Keras and Tensorflow requires 3D data for timestamp dimension as input to LSTM
#Reshape scaled data to fit dimensionality
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

#Model Architecture, Bidirectional LSTM
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001), activation='relu'), input_shape=(1, 44)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001), activation='relu')),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='sigmoid')
])

#Compile model with Adam optimization, binary crossentropy loss function
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#Callback to log training progress
class lossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        if logs is None:
            logs = {}
        with open('training_log_v7.txt', 'a') as log_file:
            log_file.write(f"Epoch: {epoch + 1}, Loss: {logs['loss']}, Val_Loss: {logs['val_loss']}, Accuracy: {logs['accuracy']}, Val_Accuracy: {logs['val_accuracy']} \n")

loss_history = lossHistory()

#Early stopping conditions for callbacks based on training progression
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_v7.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
callbacks_list = [early_stopping, model_checkpoint, loss_history]

#Model Training
history = model.fit(X_train_scaled, y_train, epochs=25, batch_size=256, validation_split=0.2, callbacks=callbacks_list, verbose=1)

#Visualization of Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model losses in Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()