import tensorflow as tf
import timeit

with tf.device('/cpu:0'):
    a = tf.random.normal([10000, 1000])
    b = tf.random.normal([1000, 2000])
    print(a, b)


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(a, b)
    return c


cpu_time = timeit.timeit(cpu_run, number=10)
print(cpu_time)

cpu_time = timeit.timeit(cpu_run, number=10)
print(cpu_time)
