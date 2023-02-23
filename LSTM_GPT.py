# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load OHLC data into a Pandas DataFrame
df = pd.read_csv("Group_data_1D.csv")

# Preprocess the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Split the data into training and testing sets
train_data = df_scaled[:int(df_scaled.shape[0]*0.8),:]
test_data = df_scaled[int(df_scaled.shape[0]*0.8):,:]

# Create sliding windows of data for training and testing
timesteps = 60
X_train = []
y_train = []
for i in range(timesteps, train_data.shape[0]):
    X_train.append(train_data[i-timesteps:i,:])
    y_train.append(train_data[i,3])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(timesteps, test_data.shape[0]):
    X_test.append(test_data[i-timesteps:i,:])
    y_test.append(test_data[i,3])
X_test, y_test = np.array(X_test), np.array(y_test)

# Build the LSTM network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the network
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Inverse transform the predictions to the original scale
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
