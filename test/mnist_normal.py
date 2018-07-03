import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tensornet.utils import load_params_conv2d_tensor, load_params_dense_tensor

from network import Model

# global settings
folder_name = "/Users/sjiahao/Documents/Models/mnist/"
model_name = "/model.ckpt"

# data_format = "channels_first" # for GPU
data_format = "channels_last" # for CPU

# load the MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
input_size, reshape_size, input_channel, output_class = 28, 32, 1, 10

# input and output
x = tf.placeholder(tf.float32, shape = [None, input_size * input_size])
x_image = tf.reshape(x, [-1, input_size, input_size, input_channel])
x_image = tf.image.resize_images(x_image, [reshape_size, reshape_size])
if data_format == "channels_first":
  x_image = tf.transpose(x_image, [0, 3, 1, 2])
y = tf.placeholder(tf.float32, shape = [None, output_class])

# convolutinonal neural network
scope = "normal"
results, params, hypers = Model(inputs = x_image, scope = scope, data_format = data_format)

# saver for checkpoint
saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope))
file = folder_name + scope + model_name

# lost function and other measures
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = results["y"]))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(results["y"], 1)), tf.float32))

# optimizer and training strategy
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

epoch_num, batch_size = 25, 100
batch_num_train = int(mnist.train.num_examples / batch_size)
batch_num_test = int(mnist.test.num_examples / batch_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(1, epoch_num + 1):
    print("Epoch %d: " % (epoch))

    for i in range(1, batch_num_train + 1):
      batch = mnist.train.next_batch(batch_size)
      train_step.run(feed_dict = {x: batch[0], y: batch[1]})

      if i % 50 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
        train_cross_entropy = cross_entropy.eval(feed_dict = {x: batch[0], y: batch[1]})
        print("samples %d, cross entropy: %.5f, training accuracy: %.2f" % (i * batch_size, train_cross_entropy, train_accuracy))

    acc = 0
    for i in range(1, batch_num_test + 1):
      batch = mnist.test.next_batch(batch_size)
      acc += accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
    acc /= batch_num_test
    print("Epoch %d, test accuracy: %.4f" % (epoch, acc))

  print("Model saved to file: %s" % (saver.save(sess, file)))
