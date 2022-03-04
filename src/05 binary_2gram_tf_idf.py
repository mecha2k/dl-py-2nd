import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization


def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    return model


batch_size = 32

train_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/test", batch_size=batch_size)

text_vectorization = TextVectorization(ngrams=2, max_tokens=20000, output_mode="count")
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
tfidf_2gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
tfidf_2gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

# model = get_model()
# model.summary()
# callbacks = [keras.callbacks.ModelCheckpoint("../data/tfidf_2gram.keras", save_best_only=True)]
# model.fit(
#     tfidf_2gram_train_ds.cache(),
#     validation_data=tfidf_2gram_val_ds.cache(),
#     epochs=10,
#     callbacks=callbacks,
# )

model = keras.models.load_model("../data/tfidf_2gram.keras")
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

inputs = keras.Input(shape=(1,), dtype="string")
processed_inputs = text_vectorization(inputs)
outputs = model(processed_inputs)
inference_model = keras.Model(inputs, outputs)

raw_text_data = tf.convert_to_tensor([["That was an excellent movie, I loved it."]])
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")
