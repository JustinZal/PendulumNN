import numpy as np
import matplotlib.pyplot as plt
import echo_state_network

import os

voltage = np.load('Voltage 1D.npy')

node1_time = np.linspace(0, 35, voltage.shape[0])
node1_voltage = voltage[:,50]

total_time_dataset = node1_time[:int(len(node1_time) * 0.3)]
total_voltage_dataset = node1_voltage[:int(len(node1_time) * 0.3)]

cutoff = int(len(total_time_dataset) * 0.8)

training_data_input = total_time_dataset[:cutoff]
training_data_output = node1_voltage[:cutoff]

validation_data_input = total_time_dataset[cutoff:]
validation_data_output = total_voltage_dataset[cutoff:]

# future_steps = int(len(validation_data_input) / 2)

esn = echo_state_network.ESN(
        1, 1, n_reservoir=3000, sparsity=0.35, noise=0
    )
weights = esn.fit(training_data_input, training_data_output)
prediction = esn.predict(validation_data_input)[:,0]

errors = []
for i in range(1, len(prediction) + 1):
    loss = np.sqrt(((prediction[:i] - validation_data_output[:i]) ** 2).mean())
    errors.append(loss)

plt.plot(validation_data_input, errors)
plt.xlabel('Prediction distance [s]')
plt.ylabel('RMSE [rad]')
plt.title('RMSE vs Forecasting distance')

plt.show()

# errors = []
# prediction = esn.predict(validation_data_input)[:,0]
#
# for i in range(1, len(validation_data_input) + 1):
#     loss = np.sqrt(((validation_data_output[:i] - prediction[:i]) ** 2).mean())
#     errors.append(loss)
#
# plt.plot(validation_data_input, errors)
# plt.xlabel('Prediction Distance [s]')
# plt.ylabel('RMSE [rad]')
# plt.title('RMSE vs Forecasting distance')
#
# plt.show()
#
# plt.savefig('real_grid_phase_prediction_based_on_forecast_distance.png', dpi=1200)