import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import datasets, layers, optimizers
print(optimizers.optimizer_v2)
import os

# opt = tf.keras.optimizers.SGD(learning_rate=0.1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print('000  ', x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

model = keras.Sequential([layers.Dense(512, activation='relu'), layers.Dense(256, activation='relu'), layers.Dense(10)])
# learning_rate  与 lr
optimizer = optimizers.SGD(lr=0.001)
print(optimizers.SGD.__dict__)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            print(x.shape)
            x = tf.reshape(x, (-1, 28 * 28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables))  # AttributeError: 'SGD' object has no attribute 'apply_gradients'
        if step % 100 == 0:
            print(epoch, step, 'loss', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
