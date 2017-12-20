import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def random_linear(theta):
    def f(x):
        return theta.dot(x) + np.random.uniform(-20, 20)
    return f


def cost(theta, x, y):
    return np.sum(np.square(theta.dot(x) - y)) / (2 * x.shape[1])


def gradient_descent(theta, x, y, learning_rate):
    next = learning_rate / x.shape[1] * np.matmul(theta.dot(x) - y, x.T)
    return theta - next


theta = np.array([[10, 100, 10]])
f = random_linear(theta)

xs = (
    np.ones(100),
    np.repeat(np.arange(0, 10), 10) / 9,
    np.tile(np.arange(0, 10), 10) / 9,
)
input = np.array(xs)
y = f(input)

candidate = np.array([np.random.uniform(-1, 1, 3)])
for i in range(0, 5000):
    candidate = gradient_descent(candidate, input, y, 0.01)
print(cost(candidate, input, y), candidate)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs[1], xs[2], y)
z = np.array([
    [
        candidate.dot([1, x / 9, y / 9])[0]
        for y in range(0, 10)
    ]
    for x in range(0, 10)
])
ax.plot_surface(
    np.arange(0, 10) / 9,
    np.arange(0, 10) / 9,
    z
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
