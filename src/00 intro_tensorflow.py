from turtle import shape
import numpy as np
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


a = tf.constant(2)
print(tf.rank(a))
print(a)

a = tf.constant([2, 3])
print(tf.rank(a))
print(a)

a = tf.constant([[2, 3], [6, 7]])
print(tf.rank(a))
print(a)

a = tf.constant(["string"])
print(tf.rank(a))
print(a)

a = tf.random.uniform(shape=(2, 3), minval=0, maxval=1)
print(a.shape)
print(a)

a = tf.random.normal(shape=(2, 3), mean=0, stddev=1)
print(tf.rank(a))
print(a)

# Eager Mode (즉시 실행 모드)
a = tf.constant(2)
b = tf.constant(3)
print(tf.add(a, b))
print(a + b)
print(tf.subtract(a, b))
print(a - b)
print(tf.multiply(a, b))
print(a * b)
print(tf.divide(a, b))
print(a / b)

# tensorflow ↔ numpy
c = tf.add(a, b).numpy()
print(type(c))

c2 = np.square(c, dtype=np.float32)
c_tensor = tf.convert_to_tensor(c2)
print(c2)
print(type(c_tensor))
print(c_tensor)

a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(a.shape)
print(a.dtype)
print(a[:, 1:])
print(a[..., 2, tf.newaxis])
print(a[..., 2])
print(a + 10)

x = np.arange(10)
print(x)
print(x[..., 0])

x = np.array([[[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6], [6, 7]]])
print(x)
print(x.shape)
print(x[..., 1])
print(x[:, :, 0])
print(x[:, np.newaxis, :, :].shape)

t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print(t @ tf.transpose(t))
print(tf.matmul(t, tf.transpose(t)))

a = tf.constant(2)
b = tf.constant(3.5)
c = tf.cast(a, tf.float32) + b
print(c)

# import timeit


# @tf.function
# def myfunc(x):
#     return x**2 - 10 * x + 3


# print(myfunc(2))
# print(myfunc(tf.constant(2)))


# def myfunc_(x):
#     return x**2 - 10 * x + 3


# print(myfunc_(2))
# print(myfunc_(tf.constant(2)))

# tf_myfunc = tf.function(myfunc_)
# print(tf_myfunc(2))
# print(tf_myfunc(tf.constant(2)))
# print(tf_myfunc.python_function(2))


# def func_get_fast(x, y, b):
#     x = tf.matmul(x, y)
#     return x + b


# tf_myfunc = tf.function(func_get_fast)

# x1 = tf.constant([[1.0, 2.0]])
# y1 = tf.constant([[2.0], [3.0]])
# b1 = tf.constant(4.0)
# print(x1.shape)
# print(y1.shape)
# print(tf_myfunc(x1, y1, b1).numpy())


# from tensorflow.keras.layers import Dense, Dropout, Flatten


# class SequentialModel(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.flatten = Flatten(input_shape=(28, 28))
#         self.dense_1 = Dense(128, activation="relu")
#         self.dropout = Dropout(0.2)
#         self.dense_2 = Dense(10)

#     def call(self, x):
#         x = self.flatten(x)
#         x = self.dense_1(x)
#         x = self.dropout(x)
#         x = self.dense_2(x)
#         return x


# inputs = tf.random.uniform(shape=(60, 28, 28))

# eager_model = SequentialModel()
# graph_model = tf.function(SequentialModel())

# print("Eager time: ", timeit.timeit(lambda: eager_model(inputs), number=100))
# print("Graph time: ", timeit.timeit(lambda: graph_model(inputs), number=100))

# x = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#     y = x**2
# grad = tape.gradient(y, x)
# print(grad.numpy())

# # only to compute one set of gradients
# # x2 = tf.Variable(4)
# # grad = tape.gradient(y, x2)
# # print(grad.numpy())

# x = tf.Variable(2.0)
# y = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#     y2 = y**2
#     z = x**2 + tf.stop_gradient(y2)
# grad = tape.gradient(z, {"x": x, "y": y2})
# print("dz/dx", grad["x"])
# print("dz/dy", grad["y"])

# weights = tf.Variable(tf.random.normal(shape=(3, 2), mean=0, stddev=1))
# biases = tf.Variable(tf.zeros(shape=(2,), dtype=tf.float32))

# x = [[1.0, 2.0, 3.0]]
# with tf.GradientTape(persistent=True) as tape:
#     y = x @ weights + biases
#     loss = tf.reduce_mean(y**2)

# (dw, db) = tape.gradient(loss, [weights, biases])
# print(weights.shape)
# print(dw.shape)

# weights2 = tf.Variable(tf.random.normal(shape=(3, 2), mean=0, stddev=1), name="weights")
# biases2 = tf.Variable(tf.zeros(shape=(2,), dtype=tf.float32), name="biases")

# x2 = [[4.0, 5.0, 6.0]]
# dw, db = tape.gradient(loss, [weights2, biases2])
# print(weights2.shape)

# del tape


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def neuron(x, W, bias=0):
    z = x * W + bias
    return sigmoid(z)


# x = tf.random.normal(shape=(2, 3), mean=0, stddev=1)
# W = tf.random.normal(shape=(2, 3), mean=0, stddev=1)
# print(x.shape)
# print(W.shape)
# print(neuron(x, W).shape)

# x = tf.random.normal(shape=(1,), mean=0, stddev=1)
# W = tf.random.normal(shape=(2, 7), mean=0, stddev=1)
# print(x.shape)
# print(W.shape)
# print(neuron(x, W).shape)

# x = 1
# y = 0
# W = tf.random.normal(shape=(1,), mean=1, stddev=0)
# print(neuron(x, W))
# print(y)

# for i in range(100):
#     output = neuron(x, W)
#     loss = y - output
#     W += 0.1 * x * loss
#     if i % 100 == 99:
#         print(f"{i+1}\t{loss}\t{output}\t")


def neuron2(x, W, bias=0):
    y = tf.matmul(a=x, b=W, transpose_a=True)
    return sigmoid(y)


# x = tf.random.normal(shape=(3, 1), mean=0, stddev=1)
# W = tf.random.normal(shape=(3, 1), mean=0, stddev=1)
# b = tf.zeros(shape=(1,), dtype=tf.float32)
# y = tf.ones(shape=(1,), dtype=tf.float32)
# print("weights: ", W)
# print("biases: ", b)
# print("y_true: ", y)

# print(neuron2(x, W, b))
# print("y: ", y)

# for i in range(1000):
#     output = neuron2(x, W, bias=b)
#     loss = y - output
#     W += 0.1 * x * loss
#     b += 0.1 * 1.0 * loss

#     if i % 100 == 99:
#         print(f"{i+1}\t{loss}\t{output}\t")

# print("weights: ", W)
# print("biases: ", b)


# # AND gate
# X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# y = np.array([[1], [0], [0], [0]])

# # W = tf.random.uniform(shape=(2,), minval=0, maxval=1)
# # b = tf.random.uniform(shape=(1,), minval=0, maxval=1)
# W = tf.ones(shape=(2,))
# b = tf.ones(shape=(1,))

# print((X[0] * W))
# print(X * W)
# print(np.sum(X * W) + b + 1.0)

# for epoch in range(2000):
#     loss_sum = 0
#     for i in range(len(X)):
#         output = sigmoid(np.sum(X[i] * W) + b + 1)
#         loss = y[i][0] - output
#         W += 0.1 * X[i] * loss
#         b += 0.1 * 1.0 * loss
#         loss_sum += loss

#     if epoch % 400 == 0:
#         print(f"Epoch:{epoch:5}\tLoss sum:{loss_sum[0]:10.5f}")

# for i in range(4):
#     print(f"x[{i}]:{X[i]}, y[{i}]:{y[i]}, output: {sigmoid(np.sum(X[i]*W) + b)}")


# XOR gate
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[0], [1], [1], [0]])

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(
    [Dense(2, input_shape=(2,), activation="sigmoid"), Dense(1, activation="sigmoid")]
)
model.compile(optimizer="rmsprop", loss="mse")
model.summary()

history = model.fit(X, y, batch_size=1, epochs=1000, verbose=1)
print(model.predict(X))

import matplotlib.pyplot as plt

plt.style.use("seaborn-white")

x = range(100)
y = tf.random.uniform(shape=(100,), minval=0, maxval=1)
plt.plot(x, y, "ro")
plt.show()
