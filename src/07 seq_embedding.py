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

for inputs, targets in int_train_ds:
    print(inputs.shape)
    print(targets.shape)
    print(inputs[0])
    print(targets[0])
    break

inputs = Input(shape=(max_length,), dtype="int64")
embedded_layer = Embedding(
    input_dim=max_tokens, output_dim=256, input_length=max_length, mask_zero=True
)
embedded = embedded_layer(inputs)
x = Bidirectional(LSTM(32))(embedded)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print(embedded_layer.get_weights()[0].shape)
keras.utils.plot_model(model, "images/seq_embedding.png", show_shapes=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "../data/embeddings_bidir_gru_with_mask.keras", save_best_only=True
    )
]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("../data/embeddings_bidir_gru_with_mask.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
