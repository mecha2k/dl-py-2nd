import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)


from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
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

vgg16_model = VGG16(weights="imagenet", input_shape=(32, 32, 3), include_top=False)

inputs = Input(shape=(32, 32, 3))
x = vgg16_model(inputs, training=False)
x = Flatten()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()


data_dir = "../data/cats_and_dogs_filtered"

# import zipfile
# import matplotlib.image as mpimg

# if not os.path.exists(data_dir):
#     os.makedirs(data_dir, exist_ok=True)
# zip_file = zipfile.ZipFile("../data/cats_and_dogs_filtered.zip", mode="r")
# zip_file.extractall(path=data_dir)
# zip_file.close()

train_cat_fnames = os.listdir(os.path.join(data_dir, "train/cats"))
train_dog_fnames = os.listdir(os.path.join(data_dir, "train/dogs"))
valid_cat_fnames = os.listdir(os.path.join(data_dir, "validation/cats"))
valid_dog_fnames = os.listdir(os.path.join(data_dir, "validation/dogs"))
print(train_dog_fnames[:5])

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

batch = datagen.flow(x, batch_size=1).next()
plt.figure(figsize=(5, 5))
plt.imshow(array_to_img(batch[0]))
plt.savefig("images/transfer_keras_00")

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
    data_dir + "/train", target_size=(150, 150), batch_size=16, class_mode="binary"
)
valid_generator = valid_datagen.flow_from_directory(
    data_dir + "/validation", target_size=(150, 150), batch_size=16, class_mode="binary"
)
print(type(train_generator))

inputs = Input(shape=(150, 150, 3))
x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation="relu")(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

print(type(train_generator))
batch = train_generator.next()
print(type(batch))
print(len(batch))
print(type(batch[0]))
print(type(batch[1]))
print(batch[0].shape)
print(batch[1].shape)

# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=1,
#     batch_size=256,
#     validation_data=valid_generator,
#     validation_steps=50,
#     verbose=2,
# )
# model.save("../data/cats_and_dogs_raw.h5")


vgg16_model = VGG16(weights="imagenet", input_shape=(150, 150, 3), include_top=False)

inputs = Input(shape=(150, 150, 3))
x = vgg16_model(inputs, training=False)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.summary()

print(len(model.trainable_weights))
vgg16_model.trainable = False
print(len(model.trainable_weights))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# history = model.fit(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=1,
#     batch_size=256,
#     validation_data=valid_generator,
#     validation_steps=50,
#     verbose=2,
# )
# model.save("../data/cats_and_dogs_pretrained.h5")

model = load_model("../data/cats_and_dogs_raw.h5")
model.summary()

image_file = os.path.join(data_dir, "validation/cats/" + valid_cat_fnames[5])
image = load_img(image_file, target_size=(150, 150))
img_tensor = img_to_array(image)
img_tensor = img_tensor[np.newaxis, ...] / 255.0
print(img_tensor.shape)

# plt.imshow(img_tensor[0])
# plt.show()

layer_output = [layer.output for layer in model.layers[:7]]
print(layer_output[6])

model = Model(inputs=[model.input], outputs=layer_output)
prediction = model.predict(img_tensor)
print(len(prediction))
print(prediction[0].shape)
plt.matshow(prediction[0][0, :, :, 0], cmap="viridis")
plt.show()
