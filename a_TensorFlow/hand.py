import tensorflow as tf
from tensorflow.python.keras  import datasets,layers ,optimizers
(xs,ys),_=datasets.mnist.load_data ()
print('',xs.shape,ys.shape)
