import numpy as np
import matplotlib.pyplot as plt


def random_linear(m, b):
    def f(x):
        return x * m + b + np.random.uniform(-m * 4, m * 4)
    return f


def mean(ns):
    return sum(ns) / len(ns)


def residual_sum_of_squares(xs, ys):
    y_mean = mean(ys)
    x_mean = mean(xs)
    b1_num = sum([
        (xs[i] - x_mean) * (ys[i] - y_mean)
        for i in range(0, len(xs))
    ])
    b1_dem = sum([
        (x - x_mean) ** 2
        for x in xs
    ])
    b1 = b1_num / b1_dem
    b0 = y_mean - b1 * x_mean

    return b0, b1


def cost(b0, b1, f, xs):
    numerator = sum([
        (b1 * x + b0 - f(x)) ** 2
        for x in xs
    ])
    return numerator / (2 * len(xs))


f = random_linear(10, 20)
xs = list(range(0, 100))
ys = list(map(f, xs))

b0, b1 = residual_sum_of_squares(xs, ys)
ys_fit = [b1 * x + b0 for x in xs]

print("cost:", cost(b0, b1, f, xs))

plt.plot(xs, ys)
plt.plot(xs, ys_fit)
plt.show()
