import numpy as np
import tensorflow as tf

np_a = np.arange(5)
print(np_a, np_a.dtype)  # [0 1 2 3 4] int32
a = tf.convert_to_tensor(np_a)
print(tf.cast(a, dtype=tf.float32))  # tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)

a = tf.range(5)
b = tf.Variable(a)
print(b, b.dtype)  # <tf.Variable 'Variable:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])> <dtype: 'int32'>
print(b.name)  # Variable:0

a = tf.ones([3])
print("[3]:", a.numpy())

a = tf.ones([2, 3])
print("[2, 3]:", a.numpy())

a = tf.ones([2, 2, 3])
print("[2, 2, 3]:", a.numpy())
a = tf.ones([])
print(int(a))
