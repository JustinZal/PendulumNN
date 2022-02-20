import numpy as np
import matplotlib.pyplot as plt

import pendulum
from echo_state_network import ESN

t = np.arange(0, 100, 0.1)
angle = pendulum.theta(t)

esn = ESN(1, 1, n_reservoir=1000)

cutoff = int(len(t) * 0.8)

inputs = t[0:cutoff]
outputs = angle[0:cutoff]

esn.fit(inputs, outputs)

validation_input = t[cutoff:]
validation_output = angle[cutoff:]

results = esn.predict(validation_input)

error = (results[0] - validation_output) ** 2

plt.plot(validation_input, error)
plt.show()

# validate_inputs = []

# def __init__(self, n_inputs, n_outputs, n_reservoir=200,
#              spectral_radius=0.95, sparsity=0, noise=0.001, input_shift=None,
#              input_scaling=None, teacher_forcing=True, feedback_scaling=None,
#              teacher_scaling=None, teacher_shift=None,
#              out_activation=identity, inverse_out_activation=identity,
#              random_state=None, silent=True):
