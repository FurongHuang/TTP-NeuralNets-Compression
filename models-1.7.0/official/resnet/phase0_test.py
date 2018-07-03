import tensorflow as tf
import resnet_model
from resnet_run_loop import ResnetArgParser
from tensornet.utils import load_params_conv2d_tensor, load_params_dense_tensor
from cifar10_main import Cifar10Model, cifar10_model_fn
from cifar10_main import input_fn as cifar_input_fn
from model_fns import cifar10_model_conversion_fn
import numpy as np
from resnet_model import DEFAULT_COMPRESSION_RATE
import tensornet.utils as utils
import tensornet.layers as layers


# In[2]:

options = {
    "svd": (utils.factorize_conv2d_svd, layers.generate_params_conv2d_svd),
    "cp":  (utils.factorize_conv2d_cp, layers.generate_params_conv2d_cp),
    "tk":  (utils.factorize_conv2d_tk, layers.generate_params_conv2d_tk),
    "tt":  (utils.factorize_conv2d_tt, layers.generate_params_conv2d_tt),
    "rcp": (utils.factorize_conv2d_rcp, layers.generate_params_conv2d_rcp),
    "rtk": (utils.factorize_conv2d_rtk, layers.generate_params_conv2d_rtk),
    "rtt": (utils.factorize_conv2d_rtt, layers.generate_params_conv2d_rtt),
    }

# We want to check that each parameter is loaded correctly
resnet_size, batch_size, version = 32, 128, 2
method, scope= 'svd','svd'
compression_rate = 0.05

model_dir = '/home/jingling/models/cifar10/phase0'
normal_model_dir = '/home/jingling/models/cifar10/normal'
filename = 'model.ckpt-97678.meta'
method_model_dir = '%s/%s/rate%s/' %(model_dir, method, compression_rate)
filename_m = 'model.ckpt.meta'


# In[4]:

saver = tf.train.import_meta_graph("%s/%s" %(normal_model_dir, filename))
var_normal_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=resnet_model.DEFAULT_SCOPE)
var_normal_list = [v for v in var_normal_list if 'Momentum' not in v.name]

var_normal_list_vals = [] 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(normal_model_dir))    
    for v in var_normal_list:
        value = sess.run(v)
        var_normal_list_vals.append((v.name, value))
        
conv_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'conv2d' in n]
dense_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'dense' in n]
bn_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'batch_normalization' in n]


# In[5]:

tf.reset_default_graph()

saver = tf.train.import_meta_graph(method_model_dir+filename_m)
var_tensor_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=method)
var_tensor_list = [v for v in var_tensor_list if 'Momentum' not in v.name]

var_tensor_list_vals = [] 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(method_model_dir))    
    for v in var_tensor_list:
        value = sess.run(v)
        var_tensor_list_vals.append((v.name, value))
        
conv_tensor_list_vals = [(n, v) for (n, v) in var_tensor_list_vals if 'conv2d' in n]
dense_tensor_list_vals = [(n, v) for (n, v) in var_tensor_list_vals if 'dense' in n]
bn_tensor_list_vals = [(n, v) for (n, v) in var_tensor_list_vals if 'batch_normalization' in n]


# In[6]:

conv_diff = []
for (name, value) in conv_normal_list_vals:
    ref_scope = name.split('kernel')[0]
    new_scope = ref_scope.replace(resnet_model.DEFAULT_SCOPE, method)
    new_kernels = [(n, v) for (n, v) in conv_tensor_list_vals if new_scope in n]
    
    shape = value.shape
    kernel_size, input_filters, output_filters = shape[0], shape[2], shape[3]
    
    if kernel_size == 1 and method in ("cp", "tk", "tt"):
        factorize_func, gen_params_func = options['svd']
    else:
        factorize_func, gen_params_func = options[method]
    
    params = gen_params_func(input_filters, output_filters, kernel_size, rate = compression_rate)
    
    if 'rank' in params:
        rank = params['rank']
        ranks = params['rank']
    elif 'ranks' in params:
        rank = params['ranks'][0]
        ranks = params['ranks']
    
    if rank <= 0:
        print("THIS SHOULD NEVER BE CALLED.")
        diff = np.linalg.norm(np.reshape(new_kernels[0][1] - value, -1)) / np.linalg.norm(np.reshape(value, -1))
        conv_diff.append(diff)
    else:
        factors = factorize_func(value, ranks)
        local_diff = 0
        for i in range(len(factors)):
            cur_diff = np.linalg.norm(np.reshape(factors[i] - new_kernels[i][1], -1))
            local_diff = local_diff + cur_diff / np.linalg.norm(np.reshape(factors[i], -1))
            
        conv_diff.append(local_diff)       
print("Avg conv layer difference: %f" %(np.sum(conv_diff) / len(conv_diff)))

dense_diff = []
for i in range(len(dense_tensor_list_vals)):
    (_, v1) = dense_tensor_list_vals[i]
    (_, v2) = dense_normal_list_vals[i]
    dense_diff.append(np.linalg.norm(np.reshape(v1-v2, -1)))
print("Avg dense layer difference: %f" %(np.sum(dense_diff) / len(dense_diff)))
    
bn_diff = []
for i in range(len(bn_tensor_list_vals)):
    (_, v1) = bn_tensor_list_vals[i]
    (_, v2) = bn_normal_list_vals[i]
    bn_diff.append(np.linalg.norm(np.reshape(v1-v2, -1)))
print("Avg batch norm layer difference: %f" %(np.sum(bn_diff) / len(bn_diff)))

print("total layer difference: %f for %d layers" %(np.sum(conv_diff)+np.sum(dense_diff)+np.sum(bn_diff), 
                                                  len(conv_diff)+len(dense_diff)+len(bn_diff)))