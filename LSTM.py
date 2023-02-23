import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

# Load the stock data into a pandas DataFrame
data = pd.read_csv('df_30min.csv').filter(['Date/Time','Close'])
data['Date/Time'] = pd.to_datetime(data['Date/Time'])
data['Date/Time'] = (data['Date/Time'] - data['Date/Time'].min()) / np.timedelta64(1,'D')
# Normalize the stock data using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Convert the normalized data into a time series format
timesteps = 60
X = []
y = []
for i in range(timesteps, data_scaled.shape[0]):
    X.append(data_scaled[i-timesteps:i, 0])
    if data_scaled[i, 0] > data_scaled[i-1, 0]:
        y.append(1)
    else:
        y.append(0)
X, y = np.array(X), np.array(y)

# Reshape the input data into a 3D format
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# One-hot encode the target data
y = to_categorical(y)

# Split the data into training and testing sets
train_size = int(0.8 * X.shape[0])
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
print(X_train, y_train)
'''
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model on the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Plot the training loss and validation loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# Plot the training accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
# Plot the true values and validation loss
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
'''