import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

apple = pd.read_csv('../data/transactions/Tech/AAPL.csv')
google = pd.read_csv('../data/transactions/Tech/GOOG.csv')

train_size = len(apple)*3//4
test_size = len(apple) - train_size
time_steps = 15
cell_unit = 16
epochNum = 100
# open_train = np.array(sp500['Open'][0:train_size])
# close_train = np.array(sp500['Close'][0:train_size])

# open_test = np.array(sp500['Open'][train_size:])
# cloase_test = np.array(sp500['Close'][train_size:])
apple_t =  np.array(apple['Close'])
apple_t2 = np.array(apple['Close'])
# apple_t = (apple_t-np.mean(apple_t))/(np.max(apple_t)-np.min(apple_t))
# apple_t = apple_t/100

google_t =  np.array(google['Close'])
# google_t = (google_t-np.mean(google_t))/(np.max(google_t)-np.min(google_t))
# google_t = google_t/100


def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb)])
        Y.append(data[(i+lb)])
    return np.array(X),np.array(Y)
X,y = processData(apple_t,time_steps)
apple_t2_i,apple_t2_o = processData(apple_t2,time_steps)
apple_t2_i = apple_t2_i.reshape((apple_t2_i.shape[0],apple_t2_i.shape[1],1))

X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

google_input,google_out = processData(google_t,time_steps)
google_input = google_input.reshape((google_input.shape[0],google_input.shape[1],1))

print(google_input.shape)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential() # most common
model.add(LSTM(cell_unit, input_shape = (time_steps,1), activation = 'relu'))

model.add(Dense(time_steps, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

apt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=apt, loss='mse')

# print(X_train.shape[0],X_test.shape[1])

callback = EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode='auto')
history = model.fit(X_train,y_train, epochs = epochNum, validation_data=(X_test,y_test),shuffle=False,callbacks =[callback])


print(apple_t2_i)
print(apple_t2_o)
predictions = model.predict([apple_t2_i])
print(predictions[0])

predictions = np.array(predictions)
real = apple_t2_o
plt.plot(predictions,color='green')
plt.plot(real,color='red')
plt.show()