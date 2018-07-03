import tensorflow as tf
import numpy as np

# default initializers
INITIALIZER_WEIGHT = tf.truncated_normal_initializer(stddev = 0.1)
INITIALIZER_BIAS   = tf.constant_initializer(0.1)

# default settings for convolutional layers
DEFAULT_DATA_FORMAT = "channels_last"
DEFAULT_RETURN_INFO = False 
DEFAULT_USE_BIAS = False

# default settings for tensorization method
DEFAULT_METHOD = "normal"
DEFAULT_COMPRESSION_RATE = 0.1


# Part 0: Auxilary functions 

def generate_shape(number, order = None, base = None, reverse = False):
  print(number)
  assert number > 0 and (number & (number - 1)) == 0, \
    "The function only supports integers that is power of two."

  assert (order is None) + (base is None) == 1, \
    "You must either supply order or base, but not both or none."

  if order is None:
    shape = []
    while number >= base:
      shape.append(base)
      number /= base

    if number * number >= base or len(shape) == 0:
      shape.append(number)
    else:
      shape[0] *= int(number)

  else:
    shape, l = [1] * order, 0
    while number > 1:
      shape[l] *= 2
      l = (l + 1) % order
      number /= 2

  if reverse: shape.reverse()
  return shape  

# Fix the zero padding function in the built-in library
def fixed_padding_tensor(inputs, kernel_size, data_format):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  pad = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]] if data_format == "channels_first" \
    else [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]] 

  return tf.pad(inputs, pad)


# Warpper function for all 2D-convoltional layers
def conv2d_tensor(inputs, filters, kernel_size, strides = 1, data_format = DEFAULT_DATA_FORMAT, use_bias = DEFAULT_USE_BIAS, 
	return_info = DEFAULT_RETURN_INFO, method = DEFAULT_METHOD, rate = DEFAULT_COMPRESSION_RATE):

  options = {"normal": conv2d,
             "svd": conv2d_svd, "cp": conv2d_cp, "tk": conv2d_tk, "tt": conv2d_tt,
             "rcp": conv2d_rcp, "rtk": conv2d_rtk, "rtt": conv2d_rtt
            }

  assert method in options, "The method is not currently supported."
  conv2d_func = options[method]

  if strides > 1:
    inputs = fixed_padding_tensor(inputs, kernel_size, data_format)

  channels = inputs.shape.as_list()[3] if data_format == "channels_last" else inputs.shape.as_list()[1]
  data_format = "NHWC" if data_format == "channels_last" else "NCHW"

  # For 1x1 convolutional layer, CP, TK, TT layers all reduce to SVD-convolutional layer
  if kernel_size == 1 and method in ("cp", "tk", "tt"):
    method, conv2d_func = "svd", options["svd"]

  params = generate_params_conv2d_tensor(channels, filters, kernel_size, method = method, rate = rate)
  
  #print("params: ", params)
  ''' No need for our hypothesis as we are comparing nonreshaped td vs reshaped td
  # If the rank for a certain rate is too small, use standard convolutional layer instead
  if ("rank" in params and params["rank"] <= 1) or ("ranks" in params and min(params["ranks"]) <= 1):
    method, conv2d_func = "normal", options["normal"]
    params = generate_params_conv2d_tensor(channels, filters, kernel_size, method = "normal", rate = 1)
  '''
  #print("method: ", method)
  #print("inputs, filters, kernel_size: ", inputs, filters, kernel_size)

  kernels = generate_kernels_conv2d_tensor(channels, filters, kernel_size, use_bias = use_bias, method = method, params = params)

  outputs = conv2d_func(inputs, kernels, strides = strides, 
    padding = ("SAME" if strides == 1 else "VALID"), use_bias = use_bias, data_format = data_format)

  return outputs if not return_info else (outputs, params, kernels)


# Warpper function for all kernels generating function for convolutional layer
def generate_kernels_conv2d_tensor(input_filters, output_filters, kernel_size, 
  use_bias = DEFAULT_USE_BIAS, method = DEFAULT_METHOD, params = {}):
  
  options = {"normal": generate_kernels_conv2d,

             "svd": generate_kernels_conv2d_svd, "cp": generate_kernels_conv2d_cp,
             "tk": generate_kernels_conv2d_tk, "tt": generate_kernels_conv2d_tt,

             "rcp": generate_kernels_conv2d_rcp,
             "rtk": generate_kernels_conv2d_rtk,
             "rtt": generate_kernels_conv2d_rtt
            }

  assert method in options, "The method is not currently supported."
  generate_kernels_func = options[method]

  kernels = generate_kernels_func(input_filters, output_filters, kernel_size, use_bias = use_bias, params = params)
  return kernels


# Warpper function for all parameters generating function for convolutional layer
def generate_params_conv2d_tensor(input_filters, output_filters, kernel_size, 
  method = DEFAULT_METHOD, rate = DEFAULT_COMPRESSION_RATE):
  
  options = {"normal": generate_params_conv2d,

             "svd": generate_params_conv2d_svd, "cp": generate_params_conv2d_cp,
             "tk": generate_params_conv2d_tk, "tt": generate_params_conv2d_tt,

             "rcp": generate_params_conv2d_rcp,
             "rtk": generate_params_conv2d_rtk,
             "rtt": generate_params_conv2d_rtt
            }

  assert method in options, "The method is not currently supported."
  generate_params_func = options[method]

  params = generate_params_func(input_filters, output_filters, kernel_size, rate = rate)
  return params


# Wrapper function for all dense layers
def dense_tensor(inputs, output_units, use_bias = DEFAULT_USE_BIAS, return_info = DEFAULT_RETURN_INFO, 
  method = DEFAULT_METHOD, rate = DEFAULT_COMPRESSION_RATE):

  options = {"normal": dense, "cp": dense_cp, "tk": dense_tk, "tt": dense_tt}

  assert method in options, "The method is not currently supported."
  dense_func = options[method]

  input_units = inputs.shape.as_list()[-1]

  params = generate_params_dense_tensor(input_units, output_units, method = method, rate = rate)
  kernels = generate_kernels_dense_tensor(input_units, output_units, use_bias = use_bias, method = method, params = params)

  outputs = dense_func(inputs, kernels, use_bias = use_bias)
  return outputs if not return_info else outputs, params, kernels


def generate_kernels_dense_tensor(input_units, output_units, 
  use_bias = DEFAULT_USE_BIAS, method = DEFAULT_METHOD, params = {}):
  
  options = {"normal": generate_kernels_dense, "cp": generate_kernels_dense_cp, 
              "tk": generate_kernels_dense_tk, "tt": generate_kernels_dense_tt}

  assert method in options, "The method is not currently supported."
  generate_kernels_func = options[method]

  kernels = generate_kernels_func(input_units, output_units, use_bias = use_bias, params = params)
  return kernels


def generate_params_dense_tensor(input_units, output_units, 
  method = DEFAULT_METHOD, rate = DEFAULT_COMPRESSION_RATE):

  options = {"normal": generate_params_dense, "cp": generate_params_dense_cp, 
              "tk": generate_params_dense_tk, "tt": generate_params_dense_tt}

  assert method in options, "The method is not currently supported."
  generate_params_func = options[method]

  params = generate_params_func(input_units, output_units, rate = rate)
  return params


## Part 1: 2D-convolutional layer (Direct factorization)

# Normal 2D-convolutional layer
def conv2d(inputs, kernels, strides = 1, padding = "SAME", use_bias = False, data_format = "NHWC"):
  assert len(kernels) == 1 + use_bias, "The number of kernels is illegal."

  strides = [1, strides, strides, 1] if data_format == "NHWC" else [1, 1, strides, strides]

  outputs = tf.nn.conv2d(inputs, kernels["kernel"], strides = strides, padding = padding, data_format = data_format)
  if use_bias: outputs = outputs + kernels["bias"]

  return outputs 


def generate_kernels_conv2d(input_filters, output_filters, kernel_size, use_bias, params):
  kernels = {}

  kernels["kernel"] = tf.get_variable("kernel", [kernel_size, kernel_size, input_filters, output_filters], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels


def generate_params_conv2d(input_filters, output_filters, kernel_size, rate = 1):
  params = {}
  return params


# SVD-convolutional layer
def conv2d_svd(inputs, kernels, strides = 1, padding = 'SAME', use_bias = False, data_format = "NHWC"):
  assert len(kernels) == 2 + use_bias, "The number of kernels is illegal."

  strides_0 = [1, strides, 1, 1] if data_format == "NHWC" else [1, 1, strides, 1]
  strides_1 = [1, 1, strides, 1] if data_format == "NHWC" else [1, 1, 1, strides]

  tensor = tf.nn.conv2d(inputs, kernels["kernel_0"], strides = strides_0, padding = padding, data_format = data_format)  
  outputs = tf.nn.conv2d(tensor, kernels["kernel_1"], strides = strides_1, padding = padding, data_format = data_format)
  if use_bias: outputs = outputs + kernels["bias"]

  return outputs

def generate_kernels_conv2d_svd(input_filters, output_filters, kernel_size, use_bias, params):
  rank = params["rank"]

  kernels = {}
  kernels["kernel_0"] = tf.get_variable("kernel_0", [kernel_size, 1, input_filters, rank], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_1"] = tf.get_variable("kernel_1", [1, kernel_size, rank, output_filters], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_svd(input_filters, output_filters, kernel_size, rate):
  original_size = kernel_size * kernel_size * input_filters * output_filters
  unit_size = kernel_size * input_filters + kernel_size * output_filters

  rank = (rate * original_size) / unit_size
  rank = np.int(np.ceil(rank))

  rank = min(input_filters, output_filters, rank)
  params = {"rank": rank}
  return params


# Parafac-convolutional layer
def conv2d_cp(inputs, kernels, strides = 1, padding = "SAME", use_bias = False, data_format = "NHWC"):
  assert len(kernels) == 3 + use_bias, "The number of kernels does not match."

  strides = [1, strides, strides, 1] if data_format == "NHWC" else [1, 1, strides, strides]

  tensor = tf.nn.conv2d(inputs, kernels["kernel_0"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)
  tensor = tf.nn.depthwise_conv2d(tensor, kernels["kernel_1"], strides = strides, padding = padding, data_format = data_format)
  outputs = tf.nn.conv2d(tensor, kernels["kernel_2"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_conv2d_cp(input_filters, output_filters, kernel_size, use_bias, params):
  rank = params["rank"]

  kernels = {}
  kernels["kernel_0"] = tf.get_variable("kernel_0", [1, 1, input_filters, rank], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_1"] = tf.get_variable("kernel_1", [kernel_size, kernel_size, rank, 1], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_2"] = tf.get_variable("kernel_2", [1, 1, rank, output_filters], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_cp(input_filters, output_filters, kernel_size, rate):
  original_size = kernel_size * kernel_size * input_filters * output_filters
  unit_size = kernel_size * kernel_size + input_filters + output_filters

  rank = (rate * original_size) / unit_size
  rank = np.int(np.ceil(rank))

  params = {"rank": rank}
  return params


# Tucker 2D-convolutional layer
def conv2d_tk(inputs, kernels, strides = 1, padding = 'SAME', use_bias = False, data_format = "NHWC"):
  assert len(kernels) == 3 + use_bias, "The number of kernels does not match."

  strides = [1, strides, strides, 1] if data_format == "NHWC" else [1, 1, strides, strides]

  tensor = tf.nn.conv2d(inputs, kernels["kernel_0"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)  
  tensor = tf.nn.conv2d(tensor, kernels["kernel_1"], strides = strides, padding = padding, data_format = data_format)
  outputs = tf.nn.conv2d(tensor, kernels["kernel_2"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)

  if use_bias: outputs = outputs + kernels["bias"]  
  return outputs

def generate_kernels_conv2d_tk(input_filters, output_filters, kernel_size, use_bias, params):
  ranks = params["ranks"]

  kernels = {}
  kernels["kernel_0"] = tf.get_variable("kernel_0", [1, 1, input_filters, ranks[0]], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_1"] = tf.get_variable("kernel_1", [kernel_size, kernel_size, ranks[0], ranks[1]], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_2"] = tf.get_variable("kernel_2", [1, 1, ranks[1], output_filters], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_tk(input_filters, output_filters, kernel_size, rate):
  original_size = kernel_size * kernel_size * input_filters * output_filters

  # use quadratic formula to find closest integer rank
  a = kernel_size * kernel_size
  b = input_filters + output_filters
  c = - rate * original_size
  rank = np.int(np.ceil((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a)))

  params = {"ranks": [rank] * 2}
  return params


# Tensor-train-convolutional layer
def conv2d_tt(inputs, kernels, strides = 1, padding = 'SAME', use_bias = False, data_format = "NHWC"):
  assert len(kernels) == 4 + use_bias, "The number of kernels does not match."

  strides_0 = [1, strides, 1, 1] if data_format == "NHWC" else [1, 1, strides, 1]
  strides_1 = [1, 1, strides, 1] if data_format == "NHWC" else [1, 1, 1, strides]

  tensor = tf.nn.conv2d(inputs, kernels["kernel_0"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)  
  tensor = tf.nn.conv2d(tensor, kernels["kernel_1"], strides = strides_0, padding = padding, data_format = data_format)  
  tensor = tf.nn.conv2d(tensor, kernels["kernel_2"], strides = strides_1, padding = padding, data_format = data_format)
  outputs = tf.nn.conv2d(tensor, kernels["kernel_3"], strides = [1, 1, 1, 1], padding = padding, data_format = data_format)

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_conv2d_tt(input_filters, output_filters, kernel_size, use_bias, params):
  ranks = params["ranks"]

  kernels = {}
  kernels["kernel_0"] = tf.get_variable("kernel_0", [1, 1, input_filters, ranks[0]], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_1"] = tf.get_variable("kernel_1", [kernel_size, 1, ranks[0], ranks[1]], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_2"] = tf.get_variable("kernel_2", [1, kernel_size, ranks[1], ranks[2]], initializer = INITIALIZER_WEIGHT)
  kernels["kernel_3"] = tf.get_variable("kernel_3", [1, 1, ranks[2], output_filters], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_tt(input_filters, output_filters, kernel_size, rate):
  original_size = kernel_size * kernel_size * input_filters * output_filters

  # use quadratic formula to find closest integer rank
  a = kernel_size + kernel_size
  b = input_filters + output_filters
  c = - rate * original_size
  rank = np.int(np.ceil((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a)))
  
  rank = min(rank, input_filters, output_filters)
  params = {"ranks": [rank] * 3}
  return params


## Part 2: Dense layer (Fully connected layer)

# Standard dense layer
def dense(inputs, kernels, use_bias = False):
  assert len(kernels) == 1 + use_bias, "The number of kernels does not match."

  outputs = tf.matmul(inputs, kernels["kernel"])
  if use_bias: outputs = outputs + kernels["bias"]

  return outputs

def generate_kernels_dense(input_units, output_units, use_bias, params = {}):
  kernels = {}
  kernels["kernel"] = tf.get_variable("kernel", [input_units, output_units], initializer = INITIALIZER_WEIGHT)
  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_units], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_dense(input_units, output_units, rate = 1):
  params = {}
  return params


# Parafac-dense layer
def dense_cp(inputs, kernels, use_bias = False):
  order = len(kernels) - use_bias

  # Extract parameters from the kernels
  input_shape, output_shape, rank = [0] * order, [0] * order, 0
  for l in range(order):
    shape = kernels["kernel_" + str(l)].shape.as_list()
    assert len(shape) == 3, "The kernels should be 3-order."
    if l: assert(shape[0] == rank), "The 1st-dimension of the kernels should match."
    rank, input_shape[l], output_shape[l] = shape

  # (1) contract with the first kernel
  tensor = tf.reshape(inputs, [-1] + input_shape)
  tensor = tf.tensordot(tensor, kernels["kernel_0"], axes = [[1], [1]])
  tensor = tf.transpose(tensor, perm = [order] + list(range(order)) + [order + 1])

  # (2) partial contract with the remaining kernels
  contract = lambda var: (tf.tensordot(var[0], var[1], axes = [[1], [0]]), 0)
  for l in range(1, order):
    tensor, _ = tf.map_fn(contract, (tensor, kernels["kernel_" + str(l)]))
  tensor = tf.reduce_sum(tensor, axis = 0)

  outputs = tf.reshape(tensor, [-1, np.prod(output_shape)])
  if use_bias: outputs = outputs + kernels["bias"]
  return outputs 

def generate_kernels_dense_cp(input_units, output_units, use_bias, params):
  input_shape, output_shape, rank = params["input_shape"], params["output_shape"], params["rank"]

  order = len(input_shape)
  assert len(output_shape) == order, \
    "The lengths of input_shape and output_shape should be equal."

  assert input_units == np.prod(input_shape), \
    "The product of input_shape should be equal to input_units."
  assert output_units == np.prod(output_shape), \
    "The product of output_shape should be equal to output_units."

  kernels = {}
  for l in range(order):
    kernels["kernel_" + str(l)] = tf.get_variable("kernel_" + str(l),
      [rank, input_shape[l], output_shape[l]], initializer = INITIALIZER_WEIGHT)
    if use_bias: kernels["bias"] = tf.get_variable("bias", [output_units], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_dense_cp(input_units, output_units, rate):
  input_shape = generate_shape(input_units, base = DEFAULT_FACTORIZATION_BASE)
  order = len(input_shape)
  assert order >= 2, "The input_units is too small to be further factorized." 

  output_shape = generate_shape(output_units, order = order, reverse = True)

  original_size = input_units * output_units
  unit_size = np.sum(np.multiply(input_shape, output_shape))
  rank = (original_size * rate) / unit_size
  rank = np.int(np.ceil(rank))

  params = {"input_shape": input_shape,
            "output_shape": output_shape, 
            "rank" : rank}

  return params


# Tucker-dense layer
def dense_tk(inputs, kernels, use_bias = False):
  # extract parameters from the kernels
  ranks = kernels["core_kernel"].shape.as_list()

  input_order, input_shape = 0, []
  while "input_kernel_" + str(input_order) in kernels:
    shape = kernels["input_kernel_" + str(input_order)].shape.as_list()
    assert len(shape) == 2, "The input_kernels should be 2-order."
    assert ranks[input_order] == shape[1], \
      "The 2nd-dimension of the input_kernel should match the dimension in the core_kernel."
    input_shape.append(shape[0])
    input_order += 1

  output_order, output_shape = 0, []
  while "output_kernel_" + str(output_order) in kernels:
    shape = kernels["output_kernel_" + str(output_order)].shape.as_list()
    assert len(shape) == 2, "The output_kernels should be 2-order."
    assert ranks[input_order + output_order] == shape[0], \
      "The 1st-dimension of the output_kernel should match the dimension in the core_kernel."
    output_shape.append(shape[1])
    output_order += 1

  assert input_order + output_order == len(ranks), \
    "The length of ranks should be equal to the sum of lengths of input_shape and output_shape."

  # (1) operate with the input kernels
  tensor = tf.reshape(inputs, [-1] + input_shape)
  for l in range(input_order):
    tensor = tf.tensordot(tensor, kernels["input_kernel_" + str(l)], axes = [[1], [0]])

  # (2) operate with the core kernel
  tensor = tf.tensordot(tensor, kernels["core_kernel"], axes = [list(range(1, input_order + 1)), list(range(input_order))])

  # (3) operate with the output kernels
  for l in range(output_order):
    tensor = tf.tensordot(tensor, kernels["output_kernel_" + str(l)], axes = [[1], [0]])

  outputs = tf.reshape(tensor, [-1, np.prod(output_shape)])
  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_dense_tk(input_units, output_units, use_bias, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  input_order, output_order = len(input_shape), len(output_shape)
  assert input_order + output_order == len(ranks), \
    "The length of ranks should be equal to the sum of lengths of input_shape and output_shape."

  assert input_units == np.prod(input_shape), \
    "The product of input_shape should be equal to input_units."
  assert output_units == np.prod(output_shape), \
    "The product of output_shape should be equal to output_units."

  kernels = {}
  for l in range(input_order):
    kernels["input_kernel_" + str(l)] = tf.get_variable("input_kernel_" + str(l), 
      [input_shape[l], ranks[l]], initializer = INITIALIZER_WEIGHT)

  kernels["core_kernel"] = tf.get_variable("core_kernel_" + str(l), 
    ranks, initializer = INITIALIZER_WEIGHT)

  for l in range(output_order):
    kernels["output_kernel_" + str(l)] = tf.get_variable("output_kernel_" + str(l), 
      [ranks[l], output_shape[l]], initializer = INITIALIZER_WEIGHT)

  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_units], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_dense_tk(input_units, output_units, rate):
  input_shape = generate_shape(input_units, base = DEFAULT_FACTORIZATION_BASE)
  output_shape = generate_shape(output_units, base = DEFAULT_FACTORIZATION_BASE)
  order = len(input_shape) + len(output_shape)

  original_size = input_units * output_units
  compressed_size = original_size * rate

  rank = np.power(original_size * rate, 1.0 / order)
  rank = np.int(np.ceil(rank))
  while np.power(rank, order) + (np.sum(input_shape) + np.sum(output_shape)) * rank < compressed_size:
    rank += 1 

  params = {"input_shape": input_shape,
            "output_shape": output_shape,
            "ranks": [rank] * order}

  return params


# Tensor-train-dense layer
def dense_tt(inputs, kernels, use_bias = False):
  order = len(kernels) - use_bias

  # extract parameters from the kernels
  input_shape, output_shape, ranks = [0] * order, [0] * order, [1] * (order+1)
  for l in range(order):
    shape = kernels["kernel_" + str(l)].shape.as_list()
    assert len(shape) == 4, "The kernels should be 4-order"
    assert shape[2] == ranks[l], \
      "The 3rd-dimension of each kernel should be equal to the 4th-dimension of its previous kernel."
    input_shape[l], output_shape[l], _, ranks[l+1] = shape
  assert ranks[-1] == 1, \
    "The 4th-dimension of last kernel should be 1."

  tensor = tf.reshape(inputs, [-1] + input_shape + [1])
  for l in range(order):
    tensor = tf.tensordot(tensor, kernels["kernel_" + str(l)], axes = [[1, -1], [0, 2]])

  outputs = tf.reshape(tensor, [-1, np.prod(output_shape)])

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_dense_tt(input_units, output_units, use_bias, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  order = len(input_shape)
  assert len(output_shape) == order, \
    "The lengths of input_shape and output_shape should be equal."
  assert len(ranks) == order - 1, \
    "The length of ranks should be one less than that of input_shape."

  assert input_units == np.prod(input_shape), \
    "The product of input_shape should be equal to input_units."
  assert output_units == np.prod(output_shape), \
    "The product of output_shape should be equal to output_units."

  ranks = [1] + ranks + [1]

  kernels = {}
  for l in range(order):
    kernels["kernel_" + str(l)] = tf.get_variable("kernel_" + str(l),
      [input_shape[l], output_shape[l], ranks[l], ranks[l+1]], initializer = INITIALIZER_WEIGHT)

  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_units], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_dense_tt(input_units, output_units, rate):
  input_shape = generate_shape(input_units, base = 8)
  order = len(input_shape)
  output_shape = generate_shape(output_units, order = order, reverse = True)

  original_size = input_units * output_units
  compressed_size = (rate + 0.0) * original_size

  a = np.sum(np.multiply(input_shape[1:-1], output_shape[1:-1]))
  b = input_shape[0] * output_shape[0] + input_shape[-1] * output_shape[-1]
  c = - compressed_size
  rank = np.int(np.ceil((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a)))

  params = {"input_shape": input_shape,
            "output_shape": output_shape, 
            "ranks": [rank] * (order - 1)}

  return params


## Part 3: Advance 2D-convolutional layer

# Advanced Parafac-convolutional layer 
def conv2d_rcp(inputs, kernels, strides = 1, padding = "SAME", use_bias = False, data_format = "NHWC"):
  # extract parameters from the kernels

  # (1) dense kernels
  order, input_shape, output_shape = 0, [], []
  while "kernel_" + str(order) in kernels:
    shape = kernels["kernel_" + str(order)].shape.as_list()
    assert len(shape) == 3, "The dense kernels should be 3-order."
    if order: assert shape[0] == rank, "The 1st-dimension of the dense kernels should match the rank."
    else: rank = shape[0]
    input_shape.append(shape[1])
    output_shape.append(shape[2])
    order += 1

  # (2) convolutional kernel (optional)
  if "kernel_conv" in kernels:
    shape = kernels["kernel_conv"].shape.as_list()
    assert len(shape) == 3, "The convolutional kernel should be 3-order."
    assert shape[2] == rank, "The 3rd-dimension of the convolutional kernel should match the rank."

  # axes for tensor contraction
  axes = [[3], [0]] if data_format == "NHWC" else [[1], [0]] 
  
  # (1.1) operate with first dense kernel
  if data_format == "NHWC":
    tensor = tf.reshape(inputs, [-1] + inputs.shape.as_list()[1:3] + input_shape)
  else:
    tensor = tf.reshape(inputs, [-1] + input_shape + inputs.shape.as_list()[2:4])

  kernel = tf.transpose(kernels["kernel_0"], perm = [1, 2, 0])
  tensor = tf.tensordot(tensor, kernel, axes = axes)

  tensor = tf.transpose(tensor, perm = [order + 3] + list(range(order + 3)))

  # (1.2) operate with the other dense kernels
  contract = lambda var: (tf.tensordot(var[0], var[1], axes = axes), 0)
  for l in range(1, order):
    tensor, _ = tf.map_fn(contract, (tensor, kernels["kernel_" + str(l)]))

  tensor = tf.reshape(tensor, [rank] + [-1] + tensor.shape.as_list()[2:4] + [np.prod(output_shape)])

  # (2.1) operate with the convolutional kernel
  if "kernel_conv" in kernels:
    if data_format == "NHWC":
      tensor = tf.transpose(tensor, perm = [1, 2, 3, 4, 0])
      kernel = tf.reshape(kernels["kernel_conv"], [shape[0], shape[1], 1, rank, 1])
    else:
      tensor = tf.transpose(tensor, perm = [1, 0, 4, 2, 3])
      kernel = tf.reshape(kernels["kernel_conv"], [1, shape[0], shape[1], rank, 1])

    strides_3d = [1, strides, strides, 1, 1] if data_format == "NHWC" else [1, 1, 1, strides, strides]
    data_format_3d = "NDHWC" if data_format == "NHWC" else "NCDHW"
    tensor = tf.nn.conv3d(tensor, kernel, strides = strides_3d, padding = padding, data_format = data_format_3d)

    if data_format == "NHWC":
      outputs = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:4])
    else:
      outputs = tf.reshape(tensor, [-1] + tensor.shape.as_list()[2:5])

  # (2.2) downsample the feature maps if there is no convolutional kernel
  else:
    tensor = tf.reduce_sum(tensor, axis = 0)

    if strides > 1:
      tensor = tf.nn.avg_pool(tensor, ksize = [1, strides, strides, 1], strides = [1, strides, strides, 1], padding = padding, data_format = "NHWC")

    if data_format == "NCHW":
      outputs = tf.transpose(tensor, perm = [0, 3, 1, 2])

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_conv2d_rcp(input_filters, output_filters, kernel_size, use_bias, params):
  input_shape, output_shape, rank = params["input_shape"], params["output_shape"], params["rank"]

  order = len(input_shape)
  assert len(output_shape) == order, \
    "The lengths of input shape and output shape should match."

  assert np.prod(input_shape) == input_filters, \
    "The product of input shape should be equal to input filters."
  assert np.prod(output_shape) == output_filters, \
    "The product of output shape should be equal to output filters."

  kernels = {}
  for l in range(len(input_shape)):
    kernels["kernel_" + str(l)] = tf.get_variable("kernel_" + str(l), 
      [rank, input_shape[l], output_shape[l]], initializer = INITIALIZER_WEIGHT)

  if kernel_size > 1:
    kernels["kernel_conv"] = tf.get_variable("kernel_conv", [kernel_size, kernel_size, rank], initializer = INITIALIZER_WEIGHT)

  if use_bias: 
    kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_rcp(input_filters, output_filters, kernel_size, rate):
  input_shape = generate_shape(input_filters, order = 2)
  order = len(input_shape)
  output_shape = generate_shape(output_filters, order = order)

  original_size = (kernel_size ** 2) * input_filters * output_filters
  unit_size = (2 * kernel_size if kernel_size > 1 else 0) + np.sum(input_shape) + np.sum(output_shape)

  rank = (rate + 0.0) * original_size / unit_size
  rank = np.int(np.ceil(rank)) 

  params = {"input_shape": input_shape, 
            "output_shape": output_shape, 
            "rank": rank}
  return params


# Advanced Tucker-convolutional layer
def conv2d_rtk(inputs, kernels, strides = 1, padding = "SAME", use_bias = False, data_format = "NHWC"):
  # Extract parameters from the kernels
  ranks = []

  # (1) input kernels
  input_order, input_shape = 0, []
  while "kernel_input_" + str(input_order) in kernels:
    shape = kernels["kernel_input_" + str(input_order)].shape.as_list()
    assert len(shape) == 2, "The input kernels should be 2-order."
    input_shape.append(shape[0])
    ranks.append(shape[1])
    input_order = input_order + 1

  # (2) output kernels
  output_order, output_shape = 0, []
  while "kernel_output_" + str(output_order) in kernels:
    shape = kernels["kernel_output_" + str(output_order)].shape.as_list()
    assert len(shape) == 2, "The output kernels should be 2-order."
    ranks.append(shape[0])
    output_shape.append(shape[1])
    output_order = output_order + 1

  # (3) core (convolutional) kernel
  shape = kernels["kernel_core"].shape.as_list()
  assert len(shape) == 4, "The core kernel should be 4-order."
  assert shape[2] == np.prod(ranks[:input_order]), \
    "The 3rd-dimension of core kernel should match the product of input shape."
  assert shape[3] == np.prod(ranks[input_order:]), \
    "The 4th-dimension of core kernel should match the product of output shape."

  # axes for tensor contraction
  axes = [[3], [0]] if data_format == "NHWC" else [[1], [0]]

  # (1) operate with input kernels
  if data_format == "NHWC":
    tensor = tf.reshape(inputs, [-1] + inputs.shape.as_list()[1:3] + input_shape)
  else:
    tensor = tf.reshape(inputs, [-1] + input_shape + inputs.shape.as_list()[2:4])

  for l in range(input_order):
    tensor = tf.tensordot(tensor, kernels["kernel_input_" + str(l)], axes = axes)

  tensor = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:3] + [np.prod(ranks[:input_order])])

  # (2) operate with the core kernel
  if data_format == "NCHW":
    tensor = tf.transpose(tensor, perm = [0, 3, 1, 2])

  strides = [1, strides, strides, 1] if data_format == "NHWC" else [1, 1, strides, strides]
  tensor = tf.nn.conv2d(tensor, kernels["kernel_core"], strides = strides, padding = padding, data_format = data_format)

  # (3) operate with the output kernels
  if data_format == "NHWC":
    tensor = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:3] + ranks[input_order:])
  else:
    tensor = tf.reshape(tensor, [-1] + ranks[input_order:] + tensor.shape.as_list()[2:4])

  for l in range(output_order):
    tensor = tf.tensordot(tensor, kernels["kernel_output_" + str(l)], axes = axes)

  outputs = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:3] + [np.prod(output_shape)])
  
  if data_format == "NCHW":
    outputs = tf.transpose(outputs, perm = [0, 3, 1, 2])

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_conv2d_rtk(input_filters, output_filters, kernel_size, use_bias, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  input_order, output_order = len(input_shape), len(output_shape)
  assert input_order + output_order == len(ranks), \
    "The length of ranks should be equal to the sum of lengths of input_shape and output_shape."

  assert input_filters == np.prod(input_shape), \
    "The product of input shape should be equal to input_filters."
  assert output_filters == np.prod(output_shape), \
    "The product of output shape should be equal to output_filters."

  kernels = {}
  for l in range(input_order):
    kernels["kernel_input_" + str(l)] = tf.get_variable("kernel_input_" + str(l), 
      [input_shape[l], ranks[l]], initializer = INITIALIZER_WEIGHT)

  kernels["kernel_core"] = tf.get_variable("kernel_core", 
    [kernel_size, kernel_size, np.prod(ranks[:input_order]), np.prod(ranks[input_order:])], initializer = INITIALIZER_WEIGHT)

  for l in range(output_order):
    kernels["kernel_output_" + str(l)] = tf.get_variable("kernel_output_" + str(l), 
      [ranks[l + input_order], output_shape[l]], initializer = INITIALIZER_WEIGHT)

  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels

def generate_params_conv2d_rtk(input_filters, output_filters, kernel_size, rate):
  input_shape = generate_shape(input_filters, order = 2)
  output_shape = generate_shape(output_filters, order = 2)
  shape = input_shape + output_shape
  order = len(shape)

  original_size = kernel_size * kernel_size * input_filters * output_filters
  compressed_size = (rate + 0.0) * original_size

  rank = np.power(compressed_size / (kernel_size * kernel_size), 1.0 / order)
  rank = np.int(np.floor(rank))
  ranks, l = [rank] * order, 0
  while (kernel_size ** 2) * np.prod(ranks) + np.sum(np.multiply(shape, ranks)) < compressed_size:
    if ranks[l] < shape[l]: ranks[l] += 1
    l = (l + 1) % order 

  params = {"input_shape": input_shape, 
            "output_shape": output_shape, 
            "ranks": ranks}
  return params


# Advanced Tensor-train-convolutional layer
def conv2d_rtt(inputs, kernels, strides = 1, padding = "SAME", use_bias = False, data_format = "NHWC"):
  # extract parameters from the kernels

  # (1) dense kernels
  order, input_shape, output_shape, ranks = 0, [], [], []
  while "kernel_" + str(order) in kernels:
    shape = kernels["kernel_" + str(order)].shape.as_list()

    assert len(shape) == 4, "The dense kernels should be 4-order."
    assert shape[2] == (ranks[-1] if order else 1), \
      "The 3rd-dimension of each dense kernel should match the 4th-dimension of its previous one."

    input_shape.append(shape[0]) 
    output_shape.append(shape[1])
    ranks.append(shape[3])
    order += 1

  # (2) convolutional kernel (optional)
  if "kernel_conv" in kernels:
    shape = kernels["kernel_conv"].shape.as_list()
    assert len(shape) == 3, "The convolutional kernel should be 3-order."
    assert shape[2] == ranks[-1], \
      "The 3rd-dimension of the convolutional kernel should match the 4th-dimension of the last dense kernel."
  else:
    assert ranks[-1] == 1, \
      "The 4th-dimension of the last dense kernel should be 1 if the kernel size is 1."

  # (1) operate with the dense kernels
  if data_format == "NHWC":
    tensor = tf.reshape(inputs, [-1] + inputs.shape.as_list()[1:3] + input_shape + [1])
  else:
    tensor = tf.reshape(inputs, [-1] + input_shape + inputs.shape.as_list()[2:4] + [1])

  axes = [[3, -1], [0, 2]] if data_format == "NHWC" else [[1, -1], [0, 2]]
  for l in range(order):
    tensor = tf.tensordot(tensor, kernels["kernel_" + str(l)], axes = axes)

  tensor = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:3] + [np.prod(output_shape)] + [ranks[-1]])

  if data_format == "NCHW":
    tensor = tf.transpose(tensor, perm = [0, 4, 3, 1, 2])

  # (2) operate with the convolutional kernel (optional)
  if "kernel_conv" in kernels:
    if data_format == "NHWC":
      kernel = tf.reshape(kernels["kernel_conv"], [shape[0], shape[1], 1, shape[2], 1])
    else:
      kernel = tf.reshape(kernels["kernel_conv"], [1, shape[0], shape[1], shape[2], 1])

    strides_3d = [1, strides, strides, 1, 1] if data_format == "NHWC" else [1, 1, 1, strides, strides]
    data_format_3d = "NDHWC" if data_format == "NHWC" else "NCDHW"

    tensor = tf.nn.conv3d(tensor, kernel, strides = strides_3d, padding = padding, data_format = data_format_3d)

  if data_format == "NHWC":
    outputs = tf.reshape(tensor, [-1] + tensor.shape.as_list()[1:4])
  else:
    outputs = tf.reshape(tensor, [-1] + tensor.shape.as_list()[2:5])

  if "kernel_conv" not in kernels and strides > 1:
    strides = [1, strides, strides, 1] if data_format == "NHWC" else [1, 1, strides, strides]
    outputs = tf.nn.avg_pool(outputs, ksize = strides, strides = strides, padding = padding, data_format = data_format)

  if use_bias: outputs = outputs + kernels["bias"]
  return outputs

def generate_kernels_conv2d_rtt(input_filters, output_filters, kernel_size, use_bias, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"] 

  order = len(input_shape)
  assert len(output_shape) == order and len(ranks) == order, \
    "The lengths of input shape, output shape and ranks should match."

  assert np.prod(input_shape) == input_filters, \
    "The product of input shape should match input_filters."
  assert np.prod(output_shape) == output_filters, \
    "The product of output shape should match output_filters."

  kernels = {}
  ranks = [1] + ranks
  for l in range(order):
    kernels["kernel_" + str(l)] = tf.get_variable("kernel_" + str(l), 
      [input_shape[l], output_shape[l], ranks[l], ranks[l+1]], initializer = INITIALIZER_WEIGHT)

  if kernel_size > 1:
    kernels["kernel_conv"] = tf.get_variable("kernel_conv", 
      [kernel_size, kernel_size, ranks[-1]], initializer = INITIALIZER_WEIGHT)

  if use_bias: kernels["bias"] = tf.get_variable("bias", [output_filters], initializer = INITIALIZER_BIAS)

  return kernels 

def generate_params_conv2d_rtt(input_filters, output_filters, kernel_size, rate):
  input_shape = generate_shape(input_filters, order = 3)
  order = len(input_shape)
  output_shape = generate_shape(output_filters, order = order, reverse = True)

  params = {"input_shape": input_shape, "output_shape": output_shape}

  original_size = (kernel_size ** 2) * input_filters * output_filters
  compressed_size = (rate + 0.0) * original_size

  # Strategy 1:
  a = np.sum(np.multiply(input_shape[1:], output_shape[1:]))
  b = input_shape[0] * output_shape[0] + kernel_size ** 2
  c = -compressed_size

  rank = np.int(np.ceil((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a)))
  if rank <= kernel_size ** 2 and rank <= input_shape[0] * output_shape[0]:
    params["ranks"] = [rank] * order
    return params 

  # Strategy 2:
  compressed_size -= kernel_size ** 4

  a = np.sum(np.multiply(input_shape[1:-1], output_shape[1:-1]))
  b = input_shape[0] * output_shape[0] + input_shape[-1] * output_shape[-1] * (kernel_size ** 2)
  c = - compressed_size

  rank = np.int(np.ceil((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2*a)))
  if rank <= input_shape[0] * output_shape[0]:
    params["ranks"] = [rank] * (order - 1) + [kernel_size ** 2]
    return params

  # Strategy 3:
  compressed_size -= (input_shape[0] * output_shape[0]) ** 2
  unit_size = np.sum(np.multiply(input_shape[1:-1], output_shape[1:-1]))

  rank = np.power(compressed_size / unit_size, 0.5)
  rank = np.int(np.ceil(rank))

  params["ranks"] = [input_shape[0] * output_shape[0]] + [rank] * (order - 2) + [kernel_size ** 2]  
  return params