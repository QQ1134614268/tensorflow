import tensorflow as tf

# 切片
# a[1][2][3]=a[1,2,3]
#
# a[start:end:step,start:end:step,]
# a[start:end:step,...,start:end:step]
a = tf.random.normal([2, 4, 28, 28, 3])
# print(a)
print(a[0].shape)
a[1, 2, ..., 0]
