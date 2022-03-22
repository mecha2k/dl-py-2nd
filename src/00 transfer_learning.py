import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)


from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPool2D,
    Conv2D,
    BatchNormalization,
    Activation,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    img_to_array,
    load_img,
    ImageDataGenerator,
)

vgg16 = VGG16(weights="imagenet", input_shape=(32, 32, 3), include_top=False)

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation(activation="relu"))
model.add(Dense(10, activation="softmax"))
model.summary()


import zipfile
import matplotlib.image as mpimg


data_dir = "../data/cats_and_dogs_filtered"

if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
zip_file = zipfile.ZipFile("../data/cats_and_dogs_filtered.zip", mode="r")
zip_file.extractall(path=data_dir)
zip_file.close()

train_cat_fnames = os.listdir(os.path.join(data_dir, "train/cats"))
train_dog_fnames = os.listdir(os.path.join(data_dir, "train/dogs"))
valid_cat_fnames = os.listdir(os.path.join(data_dir, "validation/cats"))
valid_dog_fnames = os.listdir(os.path.join(data_dir, "validation/dogs"))
print(train_dog_fnames[:10])

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

img_file = os.path.join(data_dir, "train/cats/" + train_cat_fnames[2])
image = load_img(img_file, target_size=(150, 150))
x = img_to_array(image)
x = x.reshape((1,) + x.shape)
# x = x[np.newaxis, :, :, :]
print(x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 5 == 0:
        break

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    data_dir + "/train", target_size=(150, 150), batch_size=20, class_mode="binary"
)
valid_generator = valid_datagen.flow_from_directory(
    data_dir + "/validation", target_size=(150, 150), batch_size=20, class_mode="binary"
)
print(type(train_generator))

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["acc"])
model.summary()

# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=1,
#     batch_size=256,
#     validation_data=valid_generator,
#     validation_steps=50,
#     verbose=2,
# )
