import tensorflow as tf

a = tf.constant(1)
print(a)  # tf.Tensor(1, shape=(), dtype=int32)

a = tf.constant(1.)
print(a)
a = tf.constant(1., dtype=tf.double)
print(a)
a = tf.constant([True, False])
print(a)
a = tf.constant("hello world")
print(a)
print("tf.string: ", tf.string)  # tf.Tensor(b'hello world', shape=(), dtype=string)

with tf.device("cpu"):
    a = tf.constant(1)
    print(a.device)  # /job:localhost/replica:0/task:0/device:CPU:0
    # cpu变量转gpu
    a = a.gpu()
    print(a.device)

with tf.device("gpu"):
    a = tf.constant(1)
    # print(a.device)

a = tf.range(4)
print(a)
print(a.numpy())
print(a.ndim)
a = tf.rank(a)
print(a)
a = tf.is_tensor(a)
print(a)

