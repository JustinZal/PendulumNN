import pendulum
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

time = np.arange(0, 100, 0.01)
angle = pendulum.theta(time)

# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 64, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mse', optimizer="adam")

# Display the model
model.summary()
