from cProfile import label
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, x_test.shape)
print(y_train.shape)
print(y_test[5])

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

sample_size = 9
rand_idx = np.random.randint(low=0, high=len(x_train), size=sample_size)
print(rand_idx)

plt.figure(figsize=(5, 5))
for i, idx in enumerate(rand_idx):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[idx])
    plt.xlabel(class_names[int(y_train[idx])])

x_mean = np.mean(x_train, axis=(0, 1, 2))
x_std = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape)

inputs = Input(shape=(32, 32, 3))
x = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    x_train, y_train, batch_size=256, epochs=2, validation_data=(x_valid, y_valid), verbose=1
)

hist_items = ["loss", "accuracy"]
plt.figure(figsize=(10, 4))
for i, item in enumerate(hist_items):
    plt.subplot(1, 2, i + 1)
    plt.plot(history.history[item], "b--", label=item)
    plt.plot(history.history[f"val_{item}"], "r:", label=f"val_{item}")
    plt.xlabel("epochs")
    plt.grid()
    plt.legend()
plt.show()
