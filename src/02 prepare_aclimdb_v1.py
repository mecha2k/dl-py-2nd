import os, pathlib, shutil, random

# base_dir = pathlib.Path("../data/aclImdb")
# val_dir = base_dir / "val"
# train_dir = base_dir / "train"
# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname, val_dir / category / fname)

import tensorflow as tf
import string, re

from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
from icecream import ic


batch_size = 32

train_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("../data/aclImdb/test", batch_size=batch_size)

# **Displaying the shapes and dtypes of the first batch**
ic(type(train_ds))
for inputs, targets in train_ds:
    ic("inputs.shape:", inputs.shape)
    ic("inputs.dtype:", inputs.dtype)
    ic("targets.shape:", targets.shape)
    ic("targets.dtype:", targets.dtype)
    ic("inputs[0]:", inputs[0])
    ic("targets[0]:", targets[0])
    break

for input_, target in train_ds.take(1):
    ic(input_.numpy()[0])
    ic(target.numpy()[0])

ic(train_ds.class_names[0])
ic(train_ds.class_names[1])

strip_chars = string.punctuation
# strip_chars = strip_chars.replace("[", "")
# strip_chars = strip_chars.replace("]", "")
print(strip_chars)

# ### Processing words as a set: The bag-of-words approach
# #### Single words (unigrams) with binary encoding
# **Preprocessing our datasets with a `TextVectorization` layer**
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    lowercase = tf.strings.regex_replace(lowercase, "<br />", "")
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")


text_vectorization = TextVectorization(
    max_tokens=20000, output_mode="multi_hot", standardize=custom_standardization
)
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)
vocabulary = text_vectorization.get_vocabulary()
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)

for i, text in enumerate(text_only_train_ds):
    ic(text.shape, text.dtype)
    if i > 3:
        break

for text in text_only_train_ds.take(1):
    ic(text.numpy()[0])
ic(vocabulary[:10])
ic(encoded_sentence.shape)

binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)

for inputs, targets in binary_1gram_train_ds:
    ic("inputs.shape:", inputs.shape)
    ic("inputs.dtype:", inputs.dtype)
    ic("targets.shape:", targets.shape)
    ic("targets.dtype:", targets.dtype)
    ic("inputs[0]:", inputs[0])
    ic("targets[0]:", targets[0])
    break
