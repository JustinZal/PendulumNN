import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import echo_state_network

import os

phase1 = np.load('Phase 1D.npy')

node1_time = np.linspace(0, 35, phase1.shape[0])
node1_phase = phase1[:,50]


total_time_dataset = node1_time[:int(len(node1_time) * 0.3)]
total_phase_dataset = node1_phase[:int(len(node1_time) * 0.3)]

cutoff = int(len(total_time_dataset) * 0.8)

training_data_input = total_time_dataset[:cutoff]
training_data_output = node1_phase[:cutoff]

validation_data_input = total_time_dataset[cutoff:]
validation_data_output = total_phase_dataset[cutoff:]

BASE = './results/automated_real_grid_timeseries/phase'

future_steps = int(len(validation_data_input) / 2)

for _ in range(30):
    esn = echo_state_network.ESN(
        1, 1, n_reservoir=130, sparsity=0.1, noise=0
    )
    weights = esn.fit(training_data_input, training_data_output)
    prediction = esn.predict(validation_data_input[:future_steps])[:,0]

    plt.plot(
        validation_data_input[:future_steps],
        validation_data_output[:future_steps],
        validation_data_input[:future_steps],
        prediction
    )

    print('Result:', ((prediction - validation_data_output[:future_steps]) ** 2).mean())

    sleep(1)

    timestamp = int(time())
    directory = f'{BASE}/{timestamp}'

    os.mkdir(directory)

    plt.savefig(f'{directory}/graph.png')
    plt.clf()

    prediction_file = open(f'{directory}/prediction.npy', 'wb')
    input_file = open(f'{directory}/input.npy', 'wb')
    validation_output_file = open(f'{directory}/validation.npy', 'wb')

    np.save(prediction_file, prediction)
    np.save(input_file, validation_data_input)
    np.save(validation_output_file, validation_data_output)

print('Done')

