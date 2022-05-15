import numpy as np
import matplotlib.pyplot as plt

import echo_state_network

t = np.load('time.npy')
# phase1 = np.load('phase1.npy')
# phase2 = np.load('phase2.npy')
#
# cutoff = int(len(t) * 0.9)
#
# training_data_input = np.column_stack((t[:cutoff], phase2[:cutoff]))
# training_data_output = phase1[:cutoff]
#
# validation_data_input = np.column_stack((t[cutoff:], phase2[cutoff:]))
# validation_data_output = phase1[cutoff:]
#
# future_steps = 500
reservoir_sizes = np.arange(100, 3100, 100)

# errors = []
#
# for r_size in reservoir_sizes:
#     m = []
#
#     print('Processing reservoir size:', r_size)
#
#     for i in range(10):
#         esn = echo_state_network.ESN(
#             2, 1, n_reservoir=r_size, sparsity=0.35, noise=0
#         )
#         weights = esn.fit(training_data_input, training_data_output)
#         prediction = esn.predict(validation_data_input[:future_steps])[:, 0]
#
#         loss = np.sqrt(((prediction - validation_data_output[:future_steps]) ** 2).mean())
#         m.append(loss)
#
#         print('Processing internal:', i + 1)
#
#     errors.append(np.array(m).mean())
#
# # plt.plot(reservoir_sizes, errors)
# # plt.show()

plt.plot(reservoir_sizes, np.load('second_angle_predicion_error_on_reservoir_size.npy'))
plt.xlabel('Reservoir Size (N)')
plt.ylabel('Average RMSE')
plt.title('Average RMSE vs Reservoir Size')
plt.tight_layout()

plt.show()

# f = open('second_angle_predicion_error_on_reservoir_size.npy', 'wb')
# np.save(f, np.array(errors))

