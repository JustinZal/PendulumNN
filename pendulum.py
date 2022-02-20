import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.special

g = 9.81
L = 100
theta_0 = np.pi * 0.1

omega = np.sqrt(g / L)

def K(k): #TODO fix?
    func = lambda x: 1 / np.sqrt((1 - x ** 2) * (1 - k * (x ** 2)))

    return scipy.integrate.quad(func, 0, 1)

# t = array of times
def theta(t):
    s = np.sin(theta_0 / 2)
    k = K(s ** 2)[0]

    param1 = k - omega * t
    eliptical = scipy.special.ellipj(param1, [s ** 2])[0]

    return 2 * np.arcsin(s * eliptical)
