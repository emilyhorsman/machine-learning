import numpy as np
import matplotlib.pyplot as plt


def random_linear(m, b):
    def f(x):
        return x * m + b + np.random.uniform(-m * 4, m * 4)
    return f


def calculate_model_coefficients(x, y):
    # Based on the example here:
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y)[0]


f = np.vectorize(random_linear(10, 20))
x = np.arange(100)
y = f(x)

b1, b0 = calculate_model_coefficients(x, y)

plt.plot(x, y)
plt.plot(x, b0 + b1 * x)
plt.show()
