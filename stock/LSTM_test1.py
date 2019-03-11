import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

apple = pd.read_csv('../data/transactions/Tech/AAPL.csv')

train_size = len(apple)*3//4
test_size = len(apple) - train_size
input_size = 1
output_size = 1
time_steps = 5
cell_unit = 16

print(len(apple))
# open_train = np.array(sp500['Open'][0:train_size])
# close_train = np.array(sp500['Close'][0:train_size])

# open_test = np.array(sp500['Open'][train_size:])
# cloase_test = np.array(sp500['Close'][train_size:])
cl =  np.array(apple['Close'])

cl = (cl-np.mean(cl))/(np.max(cl)-np.min(cl))
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb)])
        Y.append(data[(i+lb)])
    return np.array(X),np.array(Y)
X,y = processData(cl,10)
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential() # most common
model.add(LSTM(cell_unit, input_shape = (10,1), activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(LSTM(cell_unit, activation = 'relu'))
# model.add(Dropout(0.2))

# model.add(Dense(time_steps, activation = 'relu'))
# model.add(Dropout(0.2))

model.add(Dense(1))

apt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=apt, loss='mse')

# print(X_train.shape[0],X_test.shape[1])

callback = EarlyStopping(monitor='val_loss', patience=10,verbose=1,mode='auto')
history = model.fit(X_train,y_train, epochs = 10, validation_data=(X_test,y_test),shuffle=False,callbacks =[callback])


predictions = model.predict([X_test])
print(predictions)
print(y_test[0])
# plt.plot(np.array(predictions))
plt.plot(y_test)
plt.show