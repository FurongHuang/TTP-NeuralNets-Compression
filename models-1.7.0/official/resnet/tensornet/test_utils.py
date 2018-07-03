import tensorflow as tf
import numpy as np

from utils import factorize_conv2d_svd
from layers import generate_params_conv2d_svd, 

DEFAULT_COMPRESSION_RATE = 0.1

def check_conv2d_svd(tensor, tensor_0, tensor_1):
  kx, ky, s, t = tensor.shape
  r = tensor_0.shape[3]

  tensor = np.moveaxis(tensor, 2, 0)
  matrix = np.reshape(tensor, (s * kx, ky * t))

  matrix_0 = np.reshape(tensor_0, (kx, s, r))
  matrix_0 = np.swapaxes(matrix_0, 0, 1)
  matrix_0 = np.reshape(matrix_0, (s * kx, r))

  matrix_1 = np.reshape(tensor_1, (ky, r, t))
  matrix_1 = np.swapaxes(matrix_1, 0, 1)
  matrix_1 = np.reshape(matrix_1, (r, ky * t))

  matrix_r = np.matmul(matrix_0, matrix_1)
  
  return np.sum(np.square(matrix_r - matrix)) / np.sum(np.square(matrix))

tensor = np.random.randn(3, 3, 64, 64)

shape = tensor.shape
kernel_size = shape[0]
input_filters = shape[2]
output_filters = shape[3]

params = generate_params_conv2d_svd(input_filters, output_filters, kernel_size, rate = DEFAULT_COMPRESSION_RATE)
factors = factorize_conv2d_svd(tensor, params["rank"])
err = check_conv2d_svd(tensor, factors[0], factors[1])

print(err)
