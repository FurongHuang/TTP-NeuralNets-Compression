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

# convolutional neural network
scope = "normal"
results, params, hypers = Model(inputs = x_image, scope = scope, data_format = data_format)

scope_t, dense_method, dense_rate = "dtt", "tt", 0.01
results_t, params_t, hypers_t = Model(inputs = x_image, scope = scope_t, data_format = data_format,
  dense_method = dense_method, dense_rate = dense_rate)

# saver for checkpoints
saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope))
file = folder_name + scope + model_name

saver_t = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope_t))
file_t = folder_name + scope_t + str(dense_rate) + model_name

# lost function and other measures
mean_squared_error = tf.losses.mean_squared_error(results["fc1"], results_t["fc1"])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = results_t["y"]))

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(results_t["y"], 1)), tf.float32))

# optimizers and training strategy
train_step_1 = tf.train.AdamOptimizer(1e-3).minimize(mean_squared_error, 
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope_t))

train_step_2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, 
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope_t))

epoch_num_1, epoch_num_2, batch_size = 1, 1, 100
batch_num_train = int(mnist.train.num_examples / batch_size)
batch_num_test = int(mnist.test.num_examples / batch_size)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, file)

  def eval():
    acc = 0
    for i in range(1, batch_num_test + 1):
      batch = mnist.test.next_batch(batch_size)
      acc += accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
    return acc / batch_num_test

  print("Phase 0: Intialize the weights in the tensorized network using tensor decomposition.")
  load_params_conv2d_tensor(sess, hypers["conv1"], hypers_t["conv1"])
  load_params_conv2d_tensor(sess, hypers["conv2"], hypers_t["conv2"])
  load_params_conv2d_tensor(sess, hypers["conv3"], hypers_t["conv3"])
  load_params_dense_tensor(sess, hypers["fc1"], hypers_t["fc1"], method = dense_method, rate = dense_rate)
  load_params_dense_tensor(sess, hypers["fc2"], hypers_t["fc2"])
  print("Phase 0, test accuracy: %.3f" % (eval()))

  print("Phase 1: Tune the tensorized network using mean squared error.")
  for epoch in range(1, epoch_num_1 + 1):
    print("Phase 1, epoch %d: " % (epoch))

    for i in range(1, batch_num_train + 1):
      batch = mnist.train.next_batch(batch_size)
      train_step_1.run(feed_dict = {x: batch[0], y: batch[1]})

      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
        train_mean_squared_error = mean_squared_error.eval(feed_dict = {x: batch[0], y:batch[1]})
        print("samples %d, mean squared error: %.5f, training accuracy: %.3f" % (i * batch_size, train_mean_squared_error, train_accuracy))

    print("Phase 1, epoch %d, test accuracy: %.3f" % (epoch, eval()))

  print("Phase 2: Tune the tensorized network using cross entropy.")
  for epoch in range(1, epoch_num_2 + 1):
    print("Phase 2, epoch %d: " % (epoch))

    for i in range(1, batch_num_train + 1):
      batch = mnist.train.next_batch(batch_size)
      train_step_2.run(feed_dict = {x: batch[0], y: batch[1]})

      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y: batch[1]})
        train_cross_entropy = cross_entropy.eval(feed_dict = {x: batch[0], y: batch[1]})
        print("samples %d, cross entropy: %5f, training accuracy: %.3f" % (i * batch_size, train_cross_entropy, train_accuracy))

    print("Phase 2, epoch %d, test accuracy: %.4f" % (epoch, eval()))

  print("Model saved to file: %s" % (saver_t.save(sess, file_t)))