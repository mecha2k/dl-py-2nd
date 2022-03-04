import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import text_dataset_from_directory
from tensorflow.keras.layers import TextVectorization, LSTM, Bidirectional, Dropout, Dense

batch_size = 32

train_ds = text_dataset_from_directory("../data/aclImdb/train", batch_size=batch_size)
val_ds = text_dataset_from_directory("../data/aclImdb/val", batch_size=batch_size)
test_ds = text_dataset_from_directory("../data/aclImdb/test", batch_size=batch_size)

max_length = 200
max_tokens = 20000

text_vectorization = TextVectorization(
    max_tokens=max_tokens, output_mode="int", output_sequence_length=max_length
)
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

inputs = Input(shape=(None,), dtype="int64")
embedded = tf.one_hot(inputs, depth=max_tokens)
x = Bidirectional(LSTM(32))(embedded)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("../data/one_hot_bidir_lstm.keras", save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = keras.models.load_model("../data/one_hot_bidir_lstm.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
