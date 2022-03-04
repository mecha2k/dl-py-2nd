import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import text_dataset_from_directory
from tensorflow.keras.layers import (
    Layer,
    TextVectorization,
    Dropout,
    Dense,
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    GlobalMaxPooling1D,
)


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [
                Dense(dense_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    @staticmethod
    def compute_mask(inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


batch_size = 32

train_ds = text_dataset_from_directory("../data/aclImdb/train", batch_size=batch_size)
val_ds = text_dataset_from_directory("../data/aclImdb/val", batch_size=batch_size)
test_ds = text_dataset_from_directory("../data/aclImdb/test", batch_size=batch_size)
text_only_train_ds = train_ds.map(lambda x, y: x)

max_length = 200
max_tokens = 20000

text_vectorization = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
int_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)


vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32
sequence_length = 600


inputs = Input(shape=(None,), dtype="int64")
# x = Embedding(vocab_size, embed_dim)(inputs)
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("../data/full_transformer_encoder.keras", save_best_only=True)
]
# model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)

model = keras.models.load_model(
    "../data/full_transformer_encoder.keras",
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "PositionalEmbedding": PositionalEmbedding,
    },
)
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
