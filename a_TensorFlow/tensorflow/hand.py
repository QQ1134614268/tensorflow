import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(xs, ys), _ = datasets.mnist.load_data()
print('datasets', xs.shape, ys.shape)
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset .from_tensor_slices  ((xs, ys))
for step, (x, y) in enumerate(db):
    print(step, x.shape, y.shape)
