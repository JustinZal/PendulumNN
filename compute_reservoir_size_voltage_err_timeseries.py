import numpy as np
import matplotlib.pyplot as plt
from time import time
import echo_state_network

import os

voltage1d = np.load('Voltage 1D.npy')

node1_time = np.linspace(0, 35, voltage1d.shape[0])
node1_voltage = voltage1d[:,50]

total_time_dataset = node1_time[:int(len(node1_time) * 0.4)]
total_voltage_dataset = node1_voltage[:int(len(node1_time) * 0.4)]

cutoff = int(len(total_time_dataset) * 0.8)

training_data_input = total_time_dataset[:cutoff]
training_data_output = node1_voltage[:cutoff]

validation_data_input = total_time_dataset[cutoff:]
validation_data_output = total_voltage_dataset[cutoff:]

BASE = './results/automated_real_grid_timeseries/voltage'

future_steps = 500
reservoir_sizes = np.arange(100, 3100, 100)

for r_size in reservoir_sizes:
    m = []

    print('Processing reservoir size:', r_size)

    for _ in range(10):
        esn = echo_state_network.ESN(
            1, 1, n_reservoir=r_size, sparsity=0.1, noise=0
        )
        weights = esn.fit(training_data_input, training_data_output)
        prediction = esn.predict(validation_data_input[:future_steps])[:, 0]

        loss = np.sqrt(((prediction - validation_data_output[:future_steps]) ** 2).mean())
        m.append(loss)

    errors.append(np.array(m).mean())

err_file = open('real_grid_reservoir_size_loss.npy', 'wb')
np.save(err_file, errors)


errors = np.load('real_grid_reservoir_size_loss.npy')

plt.plot(reservoir_sizes[2:], errors[2:])
plt.xlabel('Reservoir Size (N)')
plt.ylabel('Average RMSE')
plt.title('Average RMSE vs Reservoir Size')

plt.savefig('Voltage_Average_RMSE.png')

plt.show()