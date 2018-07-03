import tensorflow as tf
from tensornet.layers import conv2d_tensor, dense_tensor

def Model(inputs, data_format = "channels_last", scope = "normal", return_info = True,
  conv2d_method = "normal", conv2d_rate = 1, dense_method = "normal", dense_rate = 1):

  results, params, hypers = {}, {}, {}

  with tf.variable_scope(scope):

    with tf.variable_scope("conv1"):
      results["h1c"], params["conv1"], hypers["conv1"] = conv2d_tensor(inputs, filters = 64, kernel_size = 3, strides = 1, use_bias = False, 
        data_format = data_format, return_info = True)
      results["h1c"] = tf.nn.relu(results["h1c"])

      if data_format == "channels_first":
        results["h1p"] = tf.nn.max_pool(results["h1c"], ksize = [1, 1, 2, 2], strides = [1, 1, 2, 2], padding = "SAME", data_format = "NCHW")
      else:
        results["h1p"] = tf.nn.max_pool(results["h1c"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", data_format = "NHWC") 

    with tf.variable_scope("conv2"):
      results["h2c"], params["conv2"], hypers["conv2"] = conv2d_tensor(results["h1p"], filters = 64, kernel_size = 3, strides = 1, use_bias = False, 
        data_format = data_format, return_info = True, method = conv2d_method, rate = conv2d_rate)
      results["h2c"] = tf.nn.relu(results["h2c"])

      if data_format == "channels_first":
        results["h2p"] = tf.nn.max_pool(results["h2c"], ksize = [1, 1, 2, 2], strides = [1, 1, 2, 2], padding = "SAME", data_format = "NCHW")
      else:
        results["h2p"] = tf.nn.max_pool(results["h2c"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", data_format = "NHWC") 

    with tf.variable_scope("conv3"):
      results["h3c"], params["conv3"], hypers["conv3"] = conv2d_tensor(results["h2p"], filters = 64, kernel_size = 3, strides = 1, use_bias =  False, 
        data_format = data_format, return_info = True, method = conv2d_method, rate = conv2d_rate)
      results["h3c"] = tf.nn.relu(results["h3c"])

      if data_format == "channels_first":
        results["h3p"] = tf.nn.max_pool(results["h3c"], ksize = [1, 1, 2, 2], strides = [1, 1, 2, 2], padding = "SAME", data_format = "NCHW")
      else:
        results["h3p"] = tf.nn.max_pool(results["h3c"], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME", data_format = "NHWC") 

      results["h3f"] = tf.layers.flatten(results["h3p"])

    with tf.variable_scope("fc1"):
      results["fc1"], params["fc1"], hypers["fc1"] = dense_tensor(results["h3f"], output_units = 64, use_bias = False, 
        return_info = True, method = dense_method, rate = dense_rate)
      results["fc1"] = tf.nn.relu(results["fc1"])

    with tf.variable_scope("fc2"):
      results["y"], params["fc2"], hypers["fc2"] = dense_tensor(results["fc1"], output_units = 10, use_bias = True, return_info = True)

  return (results, params, hypers) if return_info else results["y"]