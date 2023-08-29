import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

total_data = pd.read_csv('Clean/Clean-MSN.csv')
data = total_data.filter(['Close'])
dataset = data.values
train_len = math.ceil(len(dataset) * 0.7)

scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(dataset)

data_train = data_scaled[0:train_len, :]
x_train = []
y_train = []

for i in range(60, len(data_train)):
    x_train.append(data_train[i-60:i])
    y_train.append(data_train[i])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(30, return_sequences=False))
model.add(Dense(15))
model.add(Dense(1))

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=0.1)
)
model.fit(x_train, y_train, batch_size=1, epochs=3)

data_test = data_scaled[train_len - 60:, :]
x_test = []
y_test = dataset[train_len:, :]

for i in range(60, len(data_test)):
    x_test.append(data_test[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:]
valid = data[train_len:]
valid['Predictions'] = predictions

plt.title('Model Prediction for MSN')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training', 'Valid','Predictions'])
plt.show()

print(predictions[-1])






