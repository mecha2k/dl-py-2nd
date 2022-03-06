import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    TextVectorization,
    Dropout,
    Dense,
    Embedding,
)
from tensorflow.keras.utils import plot_model
import string, re, random
import numpy as np


text_file = "../data/spa-eng/spa.txt"
with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))
print("Total pairs of samples : ", len(text_pairs))
print(random.choice(text_pairs))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]
print(num_train_samples, num_train_samples + num_val_samples + len(test_pairs))

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
print(strip_chars)


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


vocab_size = 15000
sequence_length = 20

source_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

batch_size = 64


def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return (
        {
            "english": eng,
            "spanish": spa[:, :-1],
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    # for eng, spa in dataset:
    #     print(spa)
    #     print(format_dataset(eng, spa))
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f"targets.shape: {targets.shape}")


embed_dim = 256
latent_dim = 1024

source = Input(shape=(None,), dtype="int64", name="english")
x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(source)
encoded_source = Bidirectional(GRU(latent_dim), merge_mode="sum")(x)

past_target = Input(shape=(None,), dtype="int64", name="spanish")
x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(past_target)
x = GRU(latent_dim, return_sequences=True)(x, initial_state=encoded_source)
x = Dropout(0.5)(x)
target_next_step = Dense(vocab_size, activation="softmax")(x)

seq2seq_rnn = Model(inputs=[source, past_target], outputs=target_next_step)
seq2seq_rnn.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
plot_model(seq2seq_rnn, "images/seq2seq_rnn.png", show_shapes=True)
seq2seq_rnn.summary()

callbacks = [keras.callbacks.ModelCheckpoint("../data/seq2seq_rnn.keras", save_best_only=True)]
# seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks)


model = keras.models.load_model("../data/seq2seq_rnn.keras")

# **Translating new sentences with our RNN encoder and decoder**
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict(
            [tokenized_input_sentence, tokenized_target_sentence]
        )
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(5):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))
