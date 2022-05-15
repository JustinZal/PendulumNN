import numpy as np
import matplotlib.pyplot as plt
from time import time

import echo_state_network

ESN = echo_state_network.ESN

number_of_reservoirs = 1200
output_channels = 1
input_channels = 1
training_percentage = 0.9
ensemble_size = 10

def get_network_with_sparsity_and_sr(input_amount, sparsity, sr):
    return ESN(
        input_amount,
        output_channels,
        n_reservoir=number_of_reservoirs,
        spectral_radius=sr,
        sparsity=sparsity
    )


def compute_rms(d1, d2):
    return np.sqrt(((d1 - d2) ** 2).mean())


t = np.load('time.npy')
phase1 = np.load('phase1.npy')
phase2 = np.load('phase2.npy')

cutoff = int(len(t) * training_percentage)

training_input = t[:cutoff]
validation_input = t[cutoff:]

training_output = phase1[:cutoff]
validation_output = phase1[cutoff:]

sparsities = np.linspace(0.05, 0.95, 20)
spectral_radii = np.linspace(0.5, 2, 20)

N = 1

total_computations = ensemble_size * len(sparsities) * len(spectral_radii)
progress = 1

for s in sparsities:
    for sr in spectral_radii:

        for _ in range(ensemble_size):
            print('Total progress:', progress / total_computations * 100)

            esn = get_network_with_sparsity_and_sr(input_channels, s, sr)
            esn.fit(training_input, training_output)

            prediction = esn.predict(validation_input)[:, 0]

            progress += 1

# plt.plot(t, phase1)
# plt.show()
