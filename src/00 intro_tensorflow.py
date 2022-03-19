import numpy as np
import tensorflow as tf
from icecream import ic
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# a = tf.constant(2)
# print(tf.rank(a))
# print(a)

# a = tf.constant([2, 3])
# print(tf.rank(a))
# print(a)

# a = tf.constant([[2, 3], [6, 7]])
# print(tf.rank(a))
# print(a)

# a = tf.constant(["string"])
# print(tf.rank(a))
# print(a)

# a = tf.random.uniform(shape=(2, 3), minval=0, maxval=1)
# print(a.shape)
# print(a)

# a = tf.random.normal(shape=(2, 3), mean=0, stddev=1)
# print(tf.rank(a))
# print(a)

# # Eager Mode (즉시 실행 모드)
# a = tf.constant(2)
# b = tf.constant(3)
# print(tf.add(a, b))
# print(a + b)
# print(tf.subtract(a, b))
# print(a - b)
# print(tf.multiply(a, b))
# print(a * b)
# print(tf.divide(a, b))
# print(a / b)

# # tensorflow ↔ numpy
# c = tf.add(a, b).numpy()
# print(type(c))

# c2 = np.square(c, dtype=np.float32)
# c_tensor = tf.convert_to_tensor(c2)
# print(c2)
# print(type(c_tensor))
# print(c_tensor)

# a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
# print(a.shape)
# print(a.dtype)
# print(a[:, 1:])
# print(a[..., 2, tf.newaxis])
# print(a[..., 2])
# print(a + 10)

# x = np.arange(10)
# print(x)
# print(x[..., 0])

# x = np.array([[[1, 2], [2, 3], [3, 4]], [[4, 5], [5, 6], [6, 7]]])
# print(x)
# print(x.shape)
# print(x[..., 1])
# print(x[:, :, 0])
# print(x[:, np.newaxis, :, :].shape)

# t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
# print(t @ tf.transpose(t))
# print(tf.matmul(t, tf.transpose(t)))

# a = tf.constant(2)
# b = tf.constant(3.5)
# c = tf.cast(a, tf.float32) + b
# print(c)

import timeit


@tf.function
def myfunc(x):
    return x**2 - 10 * x + 3


print(myfunc(2))
print(myfunc(tf.constant(2)))


def myfunc_(x):
    return x**2 - 10 * x + 3


print(myfunc_(2))
print(myfunc_(tf.constant(2)))

tf_myfunc = tf.function(myfunc_)
print(tf_myfunc(2))
print(tf_myfunc(tf.constant(2)))
print(tf_myfunc.python_function(2))


def func_get_fast(x, y, b):
    x = tf.matmul(x, y)
    return x + b


tf_myfunc = tf.function(func_get_fast)

x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)
print(x1.shape)
print(y1.shape)
print(tf_myfunc(x1, y1, b1).numpy())


from tensorflow.keras.layers import Dense, Dropout, Flatten


class SequentialModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten(input_shape=(28, 28))
        self.dense_1 = Dense(128, activation="relu")
        self.dropout = Dropout(0.2)
        self.dense_2 = Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


inputs = tf.random.uniform(shape=(60, 28, 28))

eager_model = SequentialModel()
graph_model = tf.function(SequentialModel())

print("Eager time: ", timeit.timeit(lambda: eager_model(inputs), number=1000))
print("Graph time: ", timeit.timeit(lambda: graph_model(inputs), number=1000))


