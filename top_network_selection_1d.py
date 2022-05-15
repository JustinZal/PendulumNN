import numpy as np
import echo_state_network
from time import time
import matplotlib.pyplot as plt

ESN = echo_state_network.ESN

number_of_reservoirs = 2000
output_channels = 1
input_channels = 1
training_percentage = 0.97
spectral_radius = 0.95
sparsity = 0.01

t = np.load('time.npy')
phase1 = np.load('phase1.npy')
phase2 = np.load('phase2.npy')

cutoff = int(len(t) * training_percentage)

training_input = t[:cutoff]
validation_input = t[cutoff:]

training_output = phase1[:cutoff]
validation_output = phase1[cutoff:]

def get_network():
    return ESN(
        input_channels,
        output_channels,
        n_reservoir=number_of_reservoirs,
        spectral_radius=spectral_radius,
        sparsity=sparsity
    )

best_result = np.Infinity
best_prediction = None

for _ in range(30):
    t1 = time()

    esn = get_network()
    esn.fit(training_input, training_output)

    prediction = esn.predict(validation_input)[:, 0]

    result = ((prediction - validation_output) ** 2).sum()

    if result < best_result:
        best_result = result
        best_prediction = prediction
        # print(best_prediction.shape, 'here')

    t2 = time()

    print('Result:', result, 'Time:', t2 - t1)

plt.plot(validation_input, validation_output, validation_input, best_prediction)
plt.show()

print('Here is the best result:', best_result)