import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import echo_state_network

import os

time_set = np.load('time.npy')
phase1_set = np.load('phase1.npy')

cutoff = int(len(time_set) * 0.9)

training_set_inputs = time_set[:cutoff]
training_set_outputs = phase1_set[:cutoff]

validation_set_inputs = time_set[cutoff:]
validation_set_outputs = phase1_set[cutoff:]

future_steps = 300

BASE = './results/automated_2_pendulas'

print(len(training_set_inputs))

# for _ in range(30):
#     esn = echo_state_network.ESN(1, 1, 500, sparsity=0.3, spectral_radius=0.95)
#     esn.fit(training_set_inputs, training_set_outputs)
#     prediction = esn.predict(validation_set_inputs[:future_steps])[:, 0]
#
#     plt.clf()
#
#     plt.plot(
#         validation_set_inputs[:future_steps],
#         validation_set_outputs[:future_steps],
#         validation_set_inputs[:future_steps],
#         prediction
#     )
#
#     print('Result:', np.sqrt(((prediction - validation_set_outputs[:future_steps]) ** 2).mean()))
#
#     timestamp = int(time())
#     directory = f'{BASE}/{timestamp}'
#
#     os.mkdir(directory)
#
#     plt.savefig(f'{directory}/graph.png')
#
#     prediction_file = open(f'{directory}/prediction.npy', 'wb')
#     input_file = open(f'{directory}/input.npy', 'wb')
#     validation_output_file = open(f'{directory}/validation.npy', 'wb')
#
#     np.save(prediction_file, prediction)
#     np.save(input_file, validation_set_inputs)
#     np.save(validation_output_file, validation_set_outputs)
#
# print('Done')

# plt.plot(validation_set_inputs[:N_samples], validation_set_outputs[:N_samples], validation_set_inputs[:N_samples], prediction)
# plt.show()


