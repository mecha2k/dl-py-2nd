import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools
import os
from icecream import ic


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

np.random.seed(42)
tf.random.set_seed(42)


# Data API
# tf.data: https://www.tensorflow.org/api_docs/python/tf/data
# tf.data.datasets
# tf.data.datasets

# from_tensor_slices(): 개별 또는 다중 넘파이를 받고, 배치를 지원
# from_tensors(): 배치를 지원하지 않음
# from_generator(): 생성자 함수에서 입력을 취함

# 변환
# batch(): 순차적으로 지정한 배치사이즈로 데이터셋을 분할
# repeat(): 데이터를 복제
# shuffle(): 데이터를 무작위로 섞음
# map(): 데이터에 함수를 적용
# filter(): 데이터를 거르고자 할 때 사용

# 반복
# next_batch = iterator.get_next() 사용


builders = tfds.list_builders()
print(builders)

data, info = tfds.load("fashion_mnist", with_info=True)
print(info)

data, info = tfds.load("mnist", with_info=True)
print(info)

train, test = data["train"], data["test"]

# num_list = np.arange(20)
# num_list_data = tf.data.Dataset.from_tensor_slices(num_list)
# for item in num_list_data:
#     print(item)


# def generator(stop):
#     for i in itertools.count(1):
#         if i < stop:
#             yield (i, [1] * i)


# datasets = tf.data.Dataset.from_generator(
#     generator=generator,
#     args=[10],
#     output_types=(tf.int64, tf.int64),
#     output_shapes=(tf.TensorShape([]), tf.TensorShape([None])),
# )
# print(list(datasets.take(3).as_numpy_iterator()))
# for item in datasets:
#     print(item)


# datasets = num_list_data.batch(8)
# for item in datasets:
#     print(item)

data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
print(info)

# train_datasets = data["train"]
# print(type(train_datasets))
# print(train_datasets.take(1))
# print("=" * 200)

# train_datasets = train_datasets.batch(3).shuffle(3).take(2)
# for item in train_datasets:
#     print(item)
#     print("=" * 200)


# datasets = tf.data.Dataset.range(5)
# for item in datasets:
#     print(item)

# iterator = iter(datasets)
# ic(datasets)
# ic(iterator.get_next())


from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape((60000, 28, 28))
x_train = x_train.astype("float32") / 255

x_test = x_test.reshape((10000, 28, 28))
x_test = x_test.astype("float32") / 255

print(x_train.shape)
print(x_test.shape)

batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(10000).batch(batch_size=batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(batch_size=batch_size)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


import matplotlib.pyplot as plt

for image, label in train_ds.take(1):
    plt.title(f"{class_names[label[0]]}")
    plt.imshow(image[0, :, :], cmap="gray")
    plt.savefig("images/fashion_mnist", dpi=300)


from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation

inputs = Input(shape=(28, 28), dtype=tf.float32)

x = Flatten()(inputs)
x = Dense(256, kernel_initializer="he_normal")(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
x = Dropout(rate=0.5)(x)

x = Dense(128, kernel_initializer="he_normal")(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
x = Dropout(rate=0.5)(x)

x = Dense(64, kernel_initializer="he_normal")(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
x = Dropout(rate=0.5)(x)

outputs = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy

loss_func = SparseCategoricalCrossentropy()
optimizer = Adam(learning_rate=5e-5)

train_loss = Mean(name="train_loss")
train_accuracy = SparseCategoricalAccuracy(name="train_accuracy")
test_loss = Mean(name="test_loss")
test_accuracy = SparseCategoricalAccuracy(name="test_accuracy")


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_func(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


epochs = 20

for epoch in range(epochs):
    for images, labels in train_ds:
        train_step(images, labels)

    for images, labels in test_ds:
        test_step(images, labels)

    template = (
        "Epochs: {:3d}\tLoss: {:.4f}\tAccuracy: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4f}\t"
    )
    print(
        template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            test_loss.result(),
            test_accuracy.result() * 100,
        )
    )
