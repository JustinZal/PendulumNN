import numpy as np
import matplotlib.pyplot as plt

h = np.arange(0, np.sqrt(1000), 0.01)
p = 1/10 * (h ** 2)

plt.plot(h, p)
plt.xlabel('Hours into writing thesis (h)')
plt.ylabel('Suicide probability P(suicide)')
plt.title('Hours of Thesis vs Suicide Probability')

plt.show()