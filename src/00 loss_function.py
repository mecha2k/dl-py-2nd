# 딥러닝에서 쓰이는 logit은 매우 간단합니다. 모델의 출력값이 문제에 맞게 normalize 되었느냐의 여부입니다.
# 예를 들어, 10개의 이미지를 분류하는 문제에서는 주로 softmax 함수를 사용하는데요.
# 이때, 모델이 출력값으로 해당 클래스의 범위에서의 확률을 출력한다면, 이를 logit=False라고 표현할 수 있습니다.
# logit이 아니라 확률값이니까요. (이건 저만의 표현인 점을 참고해서 읽어주세요).
# 반대로 모델의 출력값이 sigmoid 또는 linear를 거쳐서 확률이 아닌 값이 나오게 된다면, logit=True라고 표현할 수 있습니다.
# 말 그대로 확률이 아니라 logit이니까요. 다시 코드로 돌아가보죠. 먼저 코드를 해석하려면 두 가지 가정이 필요합니다.
# (1) Loss Function이 CategoricalCrossEntropy이기 때문에 클래스 분류인 것을 알 수 있다.
# (2) output 배열은 모델의 출력값을 나타내며, softmax 함수를 거쳐서 나온 확률값이다.
# 이제 우리는 왜 2번 코드에서 from_logits=False를 사용했는지 알 수 있습니다. 문제에 알맞게 normalize된 상태이기 때문입니다
# (값을 전부 더해보면 1입니다, 확률을 예로 든거에요). 반대로 from_logits=True일 때는 output 배열의 값이 logit 상태가 아니기 때문에
# 우리가 생각한 값과 다른 값이 나오게 된 것입니다.

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import Input, models, Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import logging
from icecream import ic

np.random.seed(42)
tf.random.set_seed(42)

logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

cce = tf.keras.losses.CategoricalCrossentropy()
ic(cce(y_true, y_pred).numpy())

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
ic(scce(y_true, y_pred).numpy())

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0, 0, 0.5], [0.1, 0.8, 0.1, 0, 0.5]]

scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ic(scce(y_true, y_pred).numpy())


inputs = tf.random.uniform(shape=(10,), minval=0, maxval=1)
model = models.Sequential([Dense(32, input_shape=(10,), activation="relu"), Dense(3)])
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
outputs = model(inputs[np.newaxis, :])
ic(outputs)

inputs = tf.random.uniform(shape=(20, 10), minval=0, maxval=1)
model = models.Sequential([Dense(32, input_shape=(10,), activation="relu"), Dense(3)])
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
plot_model(model, "images/loss_function_01.png", show_shapes=True)
outputs = model(inputs)
y_pred = outputs.numpy()
y_true = tf.random.uniform(shape=(20, 1), minval=0, maxval=1)
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ic(y_true.shape)
ic(outputs.shape)
ic(scce(y_true, outputs).numpy())

# outputs = tf.random.uniform(shape=(20, 1), minval=0, maxval=1)
# model.fit(inputs, outputs, batch_size=16, epochs=1)


inputs = tf.random.uniform(shape=(20, 10), minval=0, maxval=1)
model = models.Sequential(
    [
        Input(shape=(10,)),
        Dense(32, activation="relu"),
        Dense(32, activation="sigmoid"),
        Dense(32, activation="tanh"),
        Dense(3, activation="softmax"),
    ]
)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
plot_model(model, "images/loss_function_02.png", show_shapes=True)

# outputs = tf.random.uniform(shape=(20, 3), minval=0, maxval=1)
# model.fit(inputs, outputs, batch_size=16, epochs=1)


inputs = Input(shape=(10,))
x = Dense(32, activation="relu")(inputs)
outputs = Dense(5, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

inputs = tf.random.uniform(shape=(20, 10), minval=0, maxval=1)
outputs = tf.random.uniform(shape=(20, 5), minval=0, maxval=1)
model.fit(inputs, outputs, batch_size=16, epochs=1)

model.save("../data/loss_function.h5")
model = models.load_model("../data/loss_function.h5")
model.summary()

predictions = model.predict(inputs, batch_size=16)
np.set_printoptions(precision=5)
ic(predictions)
ic(predictions.shape)
ic(np.argmax(predictions, axis=0))
ic(np.argmax(predictions, axis=1))


plt.figure(figsize=(10, 6))
cm = confusion_matrix(np.argmax(outputs, axis=1), np.argmax(predictions, axis=1))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("true")
plt.savefig("images/confusion_matrix", dpi=300)
# print(classification_report(np.argmax(outputs, axis=1), np.argmax(predictions, axis=1)))


inputs = Input(shape=(10,))
x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Flatten()(x)
x = Concatenate()([inputs, x])
outputs = Dense(2, activation="tanh")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

# inputs = tf.random.uniform(shape=(20, 10), minval=0, maxval=1)
# outputs = tf.random.uniform(shape=(20, 2), minval=0, maxval=1)
# model.fit(inputs, outputs, batch_size=16, epochs=1)

# validation = model.evaluate(inputs, outputs)
# ic(validation)


from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    TensorBoard,
)

inputs = Input(shape=(10,))
x = Dense(32, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Flatten()(x)
x = Concatenate()([inputs, x])
outputs = Dense(2, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])

inputs = tf.random.uniform(shape=(20, 10), minval=0, maxval=1)
outputs = tf.random.uniform(shape=(20, 2), minval=0, maxval=1)

callbacks = [
    EarlyStopping(monitor="accuracy", min_delta=0.0001, patience=2),
    ModelCheckpoint(
        "../data/loss_func_weights.keras",
        monitor="accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ),
]
model.fit(inputs, outputs, batch_size=16, epochs=1, callbacks=callbacks)
