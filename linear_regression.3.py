import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def random_linear(m, b):
    def f(x):
        return x * m + b + np.random.uniform(-m * 4, m * 4)
    return f


f = np.vectorize(random_linear(10, 20))
x = np.arange(100).reshape(-1, 1)
y = f(x)

regression = linear_model.LinearRegression()
regression.fit(x, y)
y_prediction = regression.predict(x)

plt.plot(x, y)
plt.plot(x, y_prediction)
plt.show()
