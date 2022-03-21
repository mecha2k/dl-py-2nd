import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.optimizers import Adam


np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)


x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

epochs = 1000
learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


def linear_regression(x):
    return x * W + b


def mean_square(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def run_optimization():
    with tf.GradientTape() as tape:
        pred = linear_regression(x)
        loss = mean_square(y, pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


# for step in range(epochs):
#     run_optimization()
#     if step % 100 == 0:
#         pred = linear_regression(x)
#         loss = mean_square(y, pred)
#         print(f"Step: {step:5}\tLoss: {loss:.3f}\tW: {W.numpy():.3f}\tb: {b.numpy():.3f}")

# plt.figure(figsize=(6, 4))
# plt.plot(x, y, "bo", label="데이터")
# plt.plot(x, linear_regression(x), "r", label="Fitted 데이터")
# plt.grid()
# plt.legend()
# plt.show()

a = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
c = tf.Variable(np.random.randn())

x = tf.random.normal(shape=(100,), mean=0, stddev=1)
y = x ** 2 + x * np.random.randn(100)

line_x = np.arange(min(x), max(x), step=0.001)
line_y = a * line_x ** 2 + b * line_x + c

x_ = np.arange(-5, 5, step=0.001)
y_ = a * x_ ** 2 + b * x_ + c

# plt.figure(figsize=(6, 4))
# plt.scatter(x, y, label="Data")
# plt.plot(x_, y_, "g--", label="original line")
# plt.plot(line_x, line_y, "r--", label="before training")
# plt.grid()
# plt.xlim(-4, 4)
# plt.legend()
# plt.show()


def compute_loss():
    X = x.numpy()
    pred = a * X ** 2 + b * X + c
    loss = tf.reduce_mean(tf.square(y - pred))
    return loss


optimizer = Adam(learning_rate=learning_rate)
print(compute_loss())

# for epoch in range(epochs):
#     optimizer.minimize(compute_loss, var_list=[a, b, c])
#     if epoch % 100 == 0:
#         print(f"epoch: {epoch}\t a: {a.numpy()}\t b: {b.numpy()}\t c: {c.numpy()}")

# line_x = np.arange(min(x), max(x), step=0.001)
# line_y = a * line_x ** 2 + b * line_x + c

# plt.figure(figsize=(6, 4))
# plt.scatter(x, y, label="Data")
# plt.plot(x_, y_, "g--", label="original line")
# plt.plot(line_x, line_y, "r--", label="after training")
# plt.grid()
# plt.xlim(-4, 4)
# plt.legend()
# plt.show()

from tensorflow.keras.datasets import mnist

num_classes = 10
num_features = 784
batch_size = 256
learning_rate = 5e-3

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, num_features))
x_train = x_train.astype("float32") / 255

x_test = x_test.reshape((-1, num_features))
x_test = x_test.astype("float32") / 255

print(x_train.shape)
print(x_test.shape)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.repeat().shuffle(5000).batch(batch_size=batch_size).prefetch(1)

W = tf.Variable(tf.random.normal(shape=(num_features, num_classes)))
b = tf.Variable(tf.random.normal(shape=(num_classes,)))


def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def cross_entropy(y_true, y_pred):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-9, clip_value_max=1.0)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))


def accuracy(y_true, y_pred):
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


optimizer = Adam(learning_rate=learning_rate)


def run_optimization(x, y):
    with tf.GradientTape() as tape:
        pred = logistic_regression(x)
        loss = cross_entropy(y, pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))


for epoch, (x_batch, y_batch) in enumerate(train_ds.take(1000)):
    run_optimization(x_batch, y_batch)
    if epoch % 100 == 0:
        pred = logistic_regression(x_batch)
        loss = cross_entropy(y_batch, pred)
        acc = accuracy(y_batch, pred)
        print(f"epoch: {epoch}\t loss: {loss}\t accuracy: {acc}")

pred = logistic_regression(x_test)
print(f"Test accuracy: {accuracy(y_test, pred)}")

num_images = 5
np.random.shuffle(x_test)
test_images = x_test[:num_images]
predictions = logistic_regression(test_images)

plt.figure(figsize=(6, 4))
plt.rcParams["font.size"] = 8

for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(np.reshape(test_images[i], newshape=(28, 28)), cmap="gray")
    plt.title(label=f"pred: {np.argmax(predictions.numpy()[i])}")
plt.show()
