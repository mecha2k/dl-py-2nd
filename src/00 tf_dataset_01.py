import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import os


np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
print(data)
print(type(data))

df = pd.read_csv("../data/train.tsv", sep="\t")
df = df.drop(columns=["Phrase"])
print(df.head())

datasets = tf.data.Dataset.from_tensor_slices(df)
for item in datasets.take(1):
    print(item)

datasets = datasets.map(lambda x: ({"input_ids": [x[0]], "mask": [x[1]]}, [2 * x[2]]))
for item in datasets.take(1):
    print(item)

datasets = datasets.shuffle(10000).batch(batch_size=3).prefetch(1)
for item in datasets.take(1):
    print(item[0]["input_ids"])
    print(item[1])
