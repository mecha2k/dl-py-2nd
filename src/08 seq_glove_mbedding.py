from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import text_dataset_from_directory
from tensorflow.keras.layers import (
    TextVectorization,
    LSTM,
    Bidirectional,
    Dropout,
    Dense,
    Embedding,
)
import numpy as np


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

path_to_glove_file = "../data/glove.6B.100d.txt"

embeddings_index = {}
with open(path_to_glove_file, encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print(f"Found {len(embeddings_index)} word vectors.")

embedding_dim = 100

vocabulary = text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

embedding_vector = None
embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True,
)

inputs = Input(shape=(None,), dtype="int64")
embedded = embedding_layer(inputs)
x = Bidirectional(LSTM(32))(embedded)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "../data/glove_embeddings_sequence_model.keras", save_best_only=True
    )
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = keras.models.load_model("../data/glove_embeddings_sequence_model.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
