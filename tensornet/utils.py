import tensorflow as tf
import numpy as np

from tensorly.decomposition import parafac, tucker, partial_tucker
from tensorly.tenalg import mode_dot

from tensornet.layers import generate_params_conv2d_tensor, generate_params_dense_tensor

# Warpper of parameters loading functions for conv2d layers
def load_params_conv2d_tensor(sess, reference, tensorized, use_bias = False, method = "normal", rate = 1):

  options = {"normal": load_params_conv2d,

             "svd": load_params_conv2d_svd,
             "cp":  load_params_conv2d_cp,
             "tk":  load_params_conv2d_tk,
             "tt":  load_params_conv2d_tt,

             "rcp": load_params_conv2d_rcp,
             "rtk": load_params_conv2d_rtk,
             "rtt": load_params_conv2d_rtt
            }

  assert method in options, "The method is not currently supported." 
  load_params_func = options[method]

  # convert the reference into numpy array
  if not isinstance(reference["kernel"], np.ndarray):
    reference["kernel"] = sess.run(reference["kernel"])
  if use_bias and not isinstance(reference["bias"], np.ndarray):
    reference["bias"] = sess.run(reference["bias"])

  kernel_size, _, input_filters, output_filters = reference["kernel"].shape
  
  # For 1x1 convolutional layer, CP, TK, TT layers all reduce to SVD-convolutional layer
  if kernel_size == 1 and method in ("cp", "tk", "tt"):
    method, load_params_func = "svd", options["svd"]

  params = generate_params_conv2d_tensor(input_filters, output_filters, kernel_size, method = method, rate = rate)
  
  ''' No need for our hypothesis as we are comparing nonreshaped td vs reshaped td
  # If the rank for a certain rate is too small, use standard convolutional layer instead
  if ("rank" in params and params["rank"] <= 1) or ("ranks" in params and min(params["ranks"]) <= 1):
    method, load_params_func = "normal", options["normal"]
    params = generate_params_conv2d_tensor(input_filters, output_filters, kernel_size, method = "normal", rate = 1)
  '''

  load_params_func(sess, reference, tensorized, use_bias, params)


# Warpper of parameters loading functions for dense layers
def load_params_dense_tensor(sess, reference, tensorized, use_bias = False, method = "normal", rate = 1):

  options = {"normal": load_params_dense,

             "cp": load_params_dense_cp,
             "tk": load_params_dense_tk,
             "tt": load_params_dense_tt
            }

  assert method in options, "The method is not currently supported." 
  load_params_func = options[method]

  input_units, output_units = reference["kernel"].shape
  params = generate_params_dense_tensor(input_units, output_units, method = method, rate = rate)

  # convert the reference into numpy array
  if not isinstance(reference["kernel"], np.ndarray):
    reference["kernel"] = sess.run(reference["kernel"])
  if use_bias and not isinstance(reference["bias"], np.ndarray):
    reference["bias"] = sess.run(reference["bias"])

  load_params_func(sess, reference, tensorized, use_bias, params)


# Part 0: Auxilary factorization functions

# Singular value decomposition (SVD) for tensor
def factorize_svd(tensor, location, rank):
  shape = tensor.shape
  assert 0 < location < len(shape), \
    "The location should be smaller than the tensor order."

  L, R = np.prod(shape[:location]), np.prod(shape[location:])
  assert rank <= L and rank <= R, "The rank is too large."

  tensor = np.reshape(tensor, (L, R))
  U, s, V = np.linalg.svd(tensor)

  factor_l = np.matmul(U[:,:rank], np.diag(np.sqrt(s[:rank])))
  factor_l = np.reshape(factor_l, shape[:location] + (rank,))

  factor_r = np.matmul(np.diag(np.sqrt(s[:rank])), V[:rank,:])
  factor_r = np.reshape(factor_r, (rank,) + shape[location:])

  return [factor_l, factor_r]

# Tensor Train Decomposition (TTD) for tensor (Sequential)
def factorize_tt(tensor, ranks):
  shape = tensor.shape
  assert len(shape) == len(ranks) + 1, \
    "The length of ranks should be one smaller than the tensor order."

  factors = [None] * len(shape)
  tensor = np.reshape(tensor, (1,) + shape)

  for l in range(len(ranks)):
    factors[l], tensor = factorize_svd(tensor, 2, ranks[l])

  factors[-1] = np.reshape(tensor, (ranks[-1], shape[-1], 1)) 

  return factors

# TODO: Tensor Train Decomposition (TTD) for tensor (Bipartition)
# def factorize_tt_v2(tensor, ranks)


## Part 1: Factorization functions for 2D-convolutional layer (Direct factorization)

# Standard 2D-convolutional layer
def load_params_conv2d(sess, reference, tensorized, use_bias, params):
  sess.run(tensorized["kernel"].assign(reference["kernel"]))
  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


# SVD-convolutional layer
def factorize_conv2d_svd(tensor, params):
  rank = params["rank"]

  shape = tensor.shape
  assert len(shape) == 4, "The tensor should be 4-order."

  tensor = np.moveaxis(tensor, 2, 0)
  factor_l, factor_r = factorize_svd(tensor, 2, rank)

  factor_l = np.swapaxes(factor_l, 0, 1)
  factor_l = np.reshape(factor_l, (shape[0], 1, shape[2], rank))

  factor_r = np.swapaxes(factor_r, 0, 1) 
  factor_r = np.reshape(factor_r, (1, shape[1], rank, shape[3]))

  return [factor_l, factor_r]

def load_params_conv2d_svd(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_conv2d_svd(tensor, params)

  sess.run(tensorized["kernel_0"].assign(factors[0]))
  sess.run(tensorized["kernel_1"].assign(factors[1]))
  if use_bias: sess.run(tensorized["bias"].assign(tensorized["bias"]))


# CP-convolutional layer
def factorize_conv2d_cp(tensor, params):
  rank = params["rank"]

  shape = tensor.shape
  assert len(shape) == 4, "The input tensor should be 4-order."

  tensor = np.moveaxis(tensor, 2, 0)
  tensor = np.reshape(tensor, (shape[2], shape[0] * shape[1], shape[3]))
  factors = parafac(tensor, rank)

  factors[0] = np.reshape(factors[0], (1, 1, shape[2], rank))
  factors[1] = np.reshape(factors[1], (shape[0], shape[1], rank, 1))
  factors[2] = np.reshape(np.transpose(factors[2]), (1, 1, rank, shape[3]))

  return factors

def load_params_conv2d_cp(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_conv2d_cp(tensor, params)

  sess.run(tensorized["kernel_0"].assign(factors[0]))
  sess.run(tensorized["kernel_1"].assign(factors[1]))
  sess.run(tensorized["kernel_2"].assign(factors[2]))
  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


# TK-convolutional layer
def factorize_conv2d_tk(tensor, params):
  ranks = params["ranks"] 

  shape = tensor.shape
  assert len(shape) == 4, "The input tensor should be 4-order."
  assert len(ranks) == 2, "The length of ranks should be 2."

  core, factors = partial_tucker(tensor, [2, 3], ranks)
  factors[0] = np.reshape(factors[0], (1, 1, shape[2], ranks[0]))
  factors[1] = np.reshape(np.transpose(factors[1]), (1, 1, ranks[1], shape[3]))

  return [factors[0], core, factors[1]] 

def load_params_conv2d_tk(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_conv2d_tk(tensor, params)

  sess.run(tensorized["kernel_0"].assign(factors[0]))
  sess.run(tensorized["kernel_1"].assign(factors[1]))
  sess.run(tensorized["kernel_2"].assign(factors[2]))
  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


# TT-convolutional layer
def factorize_conv2d_tt(tensor, params):
  ranks = params["ranks"]

  shape = tensor.shape
  assert len(shape) == 4, "The input tensor should be 4-order."
  assert len(ranks) == 3, "The length of ranks should be 3."

  tensor = np.moveaxis(tensor, 2, 0)
  tensor_l, tensor_r = factorize_svd(tensor, 2, ranks[1])
  factor_1, factor_2 = factorize_svd(tensor_l, 1, ranks[0])
  factor_3, factor_4 = factorize_svd(tensor_r, 2, ranks[2])

  factor_1 = np.reshape(factor_1, (1, 1, shape[2], ranks[0]))

  factor_2 = np.swapaxes(factor_2, 0, 1)
  factor_2 = np.reshape(factor_2, (shape[0], 1, ranks[0], ranks[1]))

  factor_3 = np.swapaxes(factor_3, 0, 1)
  factor_3 = np.reshape(factor_3, (1, shape[1], ranks[1], ranks[2]))

  factor_4 = np.reshape(factor_4, (1, 1, ranks[2], shape[3]))

  return [factor_1, factor_2, factor_3, factor_4]

def load_params_conv2d_tt(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_conv2d_tt(tensor, params)

  sess.run(tensorized["kernel_0"].assign(factors[0]))
  sess.run(tensorized["kernel_1"].assign(factors[1]))
  sess.run(tensorized["kernel_2"].assign(factors[2]))
  sess.run(tensorized["kernel_3"].assign(factors[3]))
  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


## Part 2: Factorization functions for dense layer (fully connected layer)

# Standard dense layer
def load_params_dense(sess, reference, tensorized, use_bias, params = {}):
  sess.run(tensorized["kernel"].assign(reference["kernel"]))
  if use_bias: sess.run(tensorized["kernel"].assign(reference["bias"]))


# Parafac dense layer
def factorize_dense_cp(tensor, params):
  input_shape, output_shape, rank = params["input_shape"], params["output_shape"], params["rank"]

  shape = tensor.shape
  assert len(shape) == 2, "The tensor should be 2-order."

  order = len(input_shape)
  assert len(output_shape) == order, \
    "The lengths of input and output shape should match."

  assert shape[0] == np.prod(input_shape), \
    "The product of input_shape should match 1st-dimension of the tensor."
  assert shape[1] == np.prod(output_shape), \
    "The product of output_shape should match 2nd-dimension of the tensor."

  tensor = np.reshape(tensor, input_shape + output_shape)
  tensor = np.transpose(tensor, axes = [val for pair in zip(range(order), range(order, 2*order)) for val in pair])
  tensor = np.reshape(tensor, np.multiply(input_shape, output_shape))

  factors = parafac(tensor, rank)
  for l in range(order):
    factors[l] = np.reshape(np.transpose(factors[l]), [rank, input_shape[l], output_shape[l]])

  return factors

def load_params_dense_cp(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_dense_cp(tensor, params)

  for l in range(len(params["input_shape"])):
    sess.run(tensorized["kernel_" + str(l)].assign(factors[l]))

  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


# Tucker dense layer
def factorize_dense_tk(tensor, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  shape = tensor.shape
  assert len(shape) == 2, "The input tensor should be 2-order."

  input_order, output_order = len(input_shape), len(output_shape)
  assert input_order + output_order == len(ranks), \
    "The length of ranks should be equal to the sum of lengths of input_shape and output_shape."

  assert shape[0] == np.prod(input_shape), \
    "The product of input shape should be equal to the first dimension of the input tensor."
  assert shape[1] == np.prod(output_shape), \
    "The product of output shape should be equal to the second dimension of the input tensor."

  tensor = np.reshape(tensor, input_shape + output_shape)
  core_factor, factors = tucker(tensor, ranks)

  input_factors, output_factors = factors[:input_order], factors[input_order:]
  for l in range(output_order):
    output_factors[l] = np.transpose(output_factors[l])

  return [input_factors, core_factor, output_factors] 

def load_params_dense_tk(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  input_factors, core_factor, output_factors = factorize_dense_tk(tensor, params)

  for l in range(len(params["input_shape"])):
    sess.run(tensorized["input_kernel_" + str(l)].assign(input_factors[l]))

  sess.run(tensorized["core_kernel"].assign(core_factor))

  for l in range(len(params["output_shape"])):
    sess.run(tensorized["output_kernel_" + str(l)].assign(output_factors[l]))

  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


# TT-dense layer
def factorize_dense_tt(tensor, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  shape = tensor.shape
  assert len(shape) == 2, "The input tensor should be 2-order."

  order = len(input_shape)
  assert len(output_shape) == order, \
    "The length of input_shape and output_shape should match."
  assert len(ranks) + 1 == order, \
    "The length of ranks should be one less than that of input_shape."

  assert shape[0] == np.prod(input_shape), \
    "The product of input_shape should match the 1st-dimension of the tensor."
  assert shape[1] == np.prod(output_shape), \
    "The product of output_shape should match the 2nd-dimension of the tensor."

  tensor = np.reshape(tensor, input_shape + output_shape)
  tensor = np.transpose(tensor, axes = [val for pair in zip(range(order), range(order, 2*order)) for val in pair])
  tensor = np.reshape(tensor, np.multiply(input_shape, output_shape))

  factors = factorize_tt(tensor, ranks)
  ranks = [1] + ranks + [1]
  for l in range(order):
    factors[l] = np.reshape(np.swapaxes(factors[l], 0, 1), (input_shape[l], output_shape[l], ranks[l], ranks[l+1]))

  return factors

def load_params_dense_tt(sess, reference, tensorized, use_bias, params):
  tensor = reference["kernel"]
  factors = factorize_dense_tt(tensor, params)

  for l in range(len(params["input_shape"])):
    sess.run(tensorized["kernel_" + str(l)].assign(factors[l]))

  if use_bias: sess.run(tensorized["bias"].assign(reference["bias"]))


## Part 3: Factorization functions for 2D-convolutional layer (Advanced factorization)

# Advanced Parafac-convolutional layer
def factorize_conv2d_rcp(tensor, params):
  input_shape, output_shape, rank = params["input_shape"], params["output_shape"], params["rank"]

  shape = tensor.shape
  assert len(shape) == 4, "The tensor should be 4-order."

  order = len(input_shape) 
  assert len(output_shape) == order, \
    "The lengths of input shape and output shape should match."

  assert shape[2] == np.prod(input_shape), \
    "The product of input shape should match the 3rd-dimension of the tensor."
  assert shape[3] == np.prod(output_shape), \
    "The product of output shape should match the 4th-dimension of the tensor."

  if shape[0] == 1:
    tensor = np.reshape(tensor, input_shape + output_shape)
    tensor = np.transpose(tensor, axes = [val for pair in zip(range(order), range(order, 2*order)) for val in pair])
    tensor = np.reshape(tensor, np.multiply(input_shape, output_shape))
  else:
    tensor = np.reshape(tensor, [shape[0] * shape[1]] + input_shape + output_shape)
    tensor = np.transpose(tensor, axes = [0] + [val for pair in zip(range(1, 1+order), range(1+order, 1+2*order)) for val in pair])
    tensor = np.reshape(tensor, [shape[0] * shape[1]] + list(np.multiply(input_shape, output_shape)))

  dense_factors = parafac(tensor, rank)

  if shape[0] == 1:
    conv_factor = []
  else:
    dense_factors, conv_factor = dense_factors[1:], np.reshape(dense_factors[0], [shape[0], shape[1], rank])

  for l in range(order):
    dense_factors[l] = np.reshape(np.transpose(dense_factors[l]), [rank, input_shape[l], output_shape[l]])

  return dense_factors, conv_factor

def load_params_conv2d_rcp(sess, reference, tensorized, use_bias, params):
  dense_factors, conv_factor = factorize_conv2d_rcp(reference["kernel"], params)

  for l in range(len(params["input_shape"])):
    sess.run(tensorized["kernel_" + str(l)].assign(dense_factors[l]))

  if "kernel_conv" in tensorized:
    sess.run(tensorized["kernel_conv"].assign(conv_factor))

  if use_bias: 
    sess.run(tensorized["bias"].assign(reference["bias"]))


# Advanced Tucker-convolutional layer
def factorize_conv2d_rtk(tensor, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"] 

  shape = tensor.shape
  assert len(tensor.shape) == 4, "The input tensor should be 4-order."

  input_order, output_order = len(input_shape), len(output_shape)
  assert input_order + output_order == len(ranks), \
    "The length of ranks should be the sum of lengths of input and output shapes."

  assert shape[2] == np.prod(input_shape), \
    "The product of input_shape should match the 3rd-dimension of the tensor."
  assert shape[3] == np.prod(output_shape), \
    "The product of output_shape should match the 4th-dimension of the tensor."

  tensor = np.reshape(tensor, list(tensor.shape[:2]) + input_shape + output_shape)
  core_factor, factors = partial_tucker(tensor, list(range(2, 2 + input_order + output_order)), ranks)

  input_factors = factors[:input_order]

  core_factor = np.reshape(core_factor, (shape[0], shape[1], np.prod(ranks[:input_order]), np.prod(ranks[input_order:])))
  
  output_factors = factors[input_order:]
  for l in range(output_order):
    output_factors[l] = np.transpose(output_factors[l])

  return [input_factors, core_factor, output_factors]

def load_params_conv2d_rtk(sess, reference, tensorized, use_bias, params):
  input_order, output_order = len(params["input_shape"]), len(params["output_shape"])

  tensor = reference["kernel"]
  input_factors, core_factor, output_factors = factorize_conv2d_rtk(tensor, params)

  for l in range(input_order):
    sess.run(tensorized["kernel_input_" + str(l)].assign(input_factors[l]))

  sess.run(tensorized["kernel_core"].assign(core_factor))

  for l in range(output_order):
    sess.run(tensorized["kernel_output_" + str(l)].assign(output_factors[l]))

  if use_bias: 
    sess.run(tensorized["bias"].assign(reference["bias"]))


# Advanced Tensor-train-convolutional layer
def factorize_conv2d_rtt(tensor, params):
  input_shape, output_shape, ranks = params["input_shape"], params["output_shape"], params["ranks"]

  shape = tensor.shape
  assert len(shape) == 4, "The input tensor should be 4-order."

  order = len(input_shape)
  assert len(output_shape) == order and len(ranks) == order, \
    "The lengths of input_shape, output_shape and ranks should match."

  assert shape[2] == np.prod(input_shape), \
    "The product of input_shape should match the 3rd-dimension of the tensor."
  assert shape[3] == np.prod(output_shape), \
    "The product of output_shape should match the 4th-dimension of the tensor."

  tensor = np.transpose(tensor, axes = [2, 3, 0, 1])

  if shape[0] == 1:
    tensor = np.reshape(tensor, input_shape + output_shape)
    tensor = np.transpose(tensor, axes = [val for pair in zip(range(order), range(order, 2*order)) for val in pair])
    tensor = np.reshape(tensor, list(np.multiply(input_shape, output_shape)))
    factors = factorize_tt(tensor, ranks[:-1])
  else:
    tensor = np.reshape(tensor, input_shape + output_shape + [shape[0] * shape[1]])
    tensor = np.transpose(tensor, axes = [val for pair in zip(range(order), range(order, 2*order)) for val in pair] + [2*order])
    tensor = np.reshape(tensor, list(np.multiply(input_shape, output_shape)) + [shape[0] * shape[1]])
    factors = factorize_tt(tensor, ranks)

  ranks = [1] + ranks
  for l in range(order):
    factors[l] = np.reshape(np.swapaxes(factors[l], 0, 1), (input_shape[l], output_shape[l], ranks[l], ranks[l+1]))

  if shape[0] > 1:
    factors[-1] = np.reshape(np.swapaxes(factors[-1], 0, 1), (shape[0], shape[1], ranks[-1]))
  else:
    factors.append([])

  return factors[:-1], factors[-1]

def load_params_conv2d_rtt(sess, reference, tensorized, use_bias, params):
  dense_factors, conv_factor = factorize_conv2d_rtt(reference["kernel"], params)

  for l in range(len(params["input_shape"])):
    sess.run(tensorized["kernel_" + str(l)].assign(dense_factors[l]))

  if "kernel_conv" in tensorized:
    sess.run(tensorized["kernel_conv"].assign(conv_factor))

  if use_bias: 
    sess.run(tensorized["bias"].assign(reference["bias"]))
