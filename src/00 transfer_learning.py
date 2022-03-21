import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

np.set_printoptions(precision=3)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
tf.random.set_seed(42)
