import numpy as np
import matplotlib.pyplot as plt


def random_linear(b, m):
    def f(x):
        return x * m + b + np.random.uniform(-m * 4, m * 4)
    return f


def cost(b, m, f, xs):
    numerator = sum([
        (m * x + b - f(x)) ** 2
        for x in xs
    ])
    return numerator / (2 * len(xs))


def dcost_dm(b, m, f, xs):
    return sum([
        x * (m * x + b - f(x))
        for x in xs
    ]) / float(len(xs))


def dcost_db(b, m, f, xs):
    return sum([
        m * x + b - f(x)
        for x in xs
    ]) / float(len(xs))


def gradient_descent(b, m, f, xs, learning_rate):
    model_coefficients = [
        dcost_db(b, m, f, xs),
        dcost_dm(b, m, f, xs)
    ]
    return (
        b - (learning_rate * model_coefficients[0]),
        m - (learning_rate * model_coefficients[1]),
    )


b_train = np.random.uniform(-40, 40)
m_train = np.random.uniform(-10, 10)
print("b =", b_train, "m =", m_train)
f = random_linear(b_train, m_train)
xs = list(range(0, 100))
ys = list(map(f, xs))

b, m = -1, 0
costs = []
for i in range(0, 20000):
    b, m = gradient_descent(b, m, f, xs, 0.00001)
    costs.append(cost(b, m, f, xs))


print("b =", b, "m =", m)

plt.subplot(211)
plt.plot(xs, ys)
plt.plot(xs, [b + m * x for x in xs])

plt.subplot(212)
plt.plot(range(0, len(costs)), costs)
plt.show()
