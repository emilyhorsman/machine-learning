import numpy as np
import matplotlib.pyplot as plt


def random_linear(b, m):
    def f(x):
        return x * m + b + np.random.uniform(-m * 4, m * 4)
    return f


def dcost_dm(b, m, f):
    def d(x):
        return x * (m * x + b - f(x))
    return np.vectorize(d)


def dcost_db(b, m, f):
    def d(x):
        return m * x + b - f(x)
    return np.vectorize(d)


def gradient_descent(b, m, f, x, learning_rate):
    db = dcost_db(b, m, f)
    dm = dcost_dm(b, m, f)
    model_coefficients = [
        np.sum(db(x)) / len(x),
        np.sum(dm(x)) / len(x)
    ]
    return (
        b - (learning_rate * model_coefficients[0]),
        m - (learning_rate * model_coefficients[1]),
    )


b_train = np.random.uniform(-40, 40)
m_train = np.random.uniform(-10, 10)
print("b =", b_train, "m =", m_train, "(Training)")
f = np.vectorize(random_linear(b_train, m_train))
x = np.arange(10)
y = f(x)

b, m = -1, 0
for i in range(0, 200):
    b, m = gradient_descent(b, m, f, x, 0.0005)
    print("b =", b, "m =", m, "(Prediction)")

plt.plot(x, y)
plt.plot(x, b + m * x)
plt.show()
