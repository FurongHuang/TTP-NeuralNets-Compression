# ==============================================================================
# Tensorized Spectrum Preserving Compression for Neural Networks
# https://arxiv.org/pdf/1805.10352.pdf
# Developed by Jiahao Su (jiahaosu@terpmail.umd.edu) and
# Jingling Li (jingling@cs.umd.edu)
# May 18, 2018
#
# This file conducts weight factorization phase
# ==============================================================================

import tensorflow as tf
import resnet_model
from resnet_run_loop import ResnetArgParser
from tensornet.utils import load_params_conv2d_tensor, load_params_dense_tensor
from resnet_model import DEFAULT_COMPRESSION_RATE, DEFAULT_STARTING_RATE, rate_function_linear
from cifar10_main import Cifar10Model, cifar10_model_fn
from imagenet_main import ImagenetModel, imagenet_model_fn
from cifar10_main import input_fn as cifar_input_fn
from imagenet_main import input_fn as imagenet_input_fn
from model_fns import cifar10_model_conversion_fn, imagenet_model_conversion_fn
from utils.logging import hooks_helper
import sys, os
import shutil
import logging

def main(argv):
    parser = ResnetArgParser()

    parser.add_argument(
        '--model_class', '-mc', default = 'cifar10',
        help = "[default: %(default)s] The model you are performing experiment on.",
        metavar = '<MC>'
    )

    parser.add_argument(
        '--output_path', '-op', default = '/tmp/output',
        help = "[default: %(default)s] The location of the tensorized model of phase0.",
        metavar = '<OP>'
    )

    parser.add_argument(
        '--pretrained_model_dir', '-pmd', default = '/tmp/cifar10_model_tensor_based/normal/',
        help = "[default: %(default)s] The location of the pretrained model for phase0.",
        metavar = '<PMD>'
    )
    
    # Set defaults that are reasonable for this model.
    parser.set_defaults(data_dir='/tmp/cifar10_data',
                    pretrained_model_dir='/tmp/pretrained_model', # pretrained model after loading into our scope
                    resnet_size=32,
                    batch_size=128,
                    version=2,
                    output_path='/tmp/output',
                    method='svd',
                    scope='svd',
                    rate=0.5,
                    rate_decay='flat')

    flags = parser.parse_args(args=argv[1:])
    compression_rate = flags.rate
    rate_decay = flags.rate_decay

    '''Define the parameters we need for each experiment'''
    if flags.model_class == 'cifar10':
        model_class, input_fn, model_fn = Cifar10Model, cifar_input_fn, cifar10_model_fn
        model_conversion_fn = cifar10_model_conversion_fn
        total_params, growth_params = 463616, 1256192
    else:
        model_class, input_fn, model_fn = ImagenetModel, imagenet_input_fn, imagenet_model_fn
        model_conversion_fn = imagenet_model_conversion_fn
        total_params, growth_params = 23445504, 83640320
     
    if rate_decay == 'linear_inc':
        starting_rate = 0
    elif rate_decay == 'linear_dec':
        starting_rate = DEFAULT_STARTING_RATE
    else:
        starting_rate = 1

    data_dir = flags.data_dir

    resnet_size, batch_size, version = flags.resnet_size, flags.batch_size, flags.version
    method, scope= flags.method, flags.scope

    pretrained_model_dir = flags.pretrained_model_dir 

    output_path = flags.output_path
    checkpoint_file = '%s/%s/rate%s/%s' %(output_path, method, compression_rate, 'model.ckpt')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    logging.basicConfig(level=logging.INFO,
                    datefmt='%m-%d %H:%M',
                    filename='%s/%s_%s.log' %(output_path, method, compression_rate),
                    filemode='w+')

    logging.info("Starting phase0...")
    print("Starting phase0...")
    msg = "method: %s\n scope: %s\n compression rate: %s" %(method, scope, compression_rate)
    print(msg)
    logging.info(msg)

    pretrained_model_file = tf.train.latest_checkpoint(pretrained_model_dir)+'.meta'
    saver = tf.train.import_meta_graph(pretrained_model_file)
    var_normal_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=resnet_model.DEFAULT_SCOPE)
    var_normal_list = [v for v in var_normal_list if 'Momentum' not in v.name]

    var_normal_list_vals = [] 

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print(pretrained_model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(pretrained_model_dir))    
        for v in var_normal_list:
            value = sess.run(v)
            var_normal_list_vals.append((v.name, value))

    conv_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'conv2d' in n]
    dense_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'dense' in n]
    bn_normal_list_vals = [(n, v) for (n, v) in var_normal_list_vals if 'batch_normalization' in n]
    
    tf.reset_default_graph()

    model = model_class(resnet_size=resnet_size, method=method, scope=scope, rate=compression_rate, rate_decay=rate_decay)
    dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    outputs = model(next_element[0], False)

    var_tensor_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=method)
    conv_tensor_list = [v for v in var_tensor_list if 'conv2d' in v.name]
    dense_tensor_list = [v for v in var_tensor_list if 'dense' in v.name]
    bn_tensor_list = [v for v in var_tensor_list if 'batch_normalization' in v.name]
    
    new_model_dir = '%s/%s/rate%s/' %(output_path, method, compression_rate)
    if os.path.exists(new_model_dir):
        shutil.rmtree(new_model_dir)
    else:
        os.makedirs(new_model_dir)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: 
        sess.run(tf.global_variables_initializer())

        # load weights for the first convolutional layer (the one that connects to the input)
        sess.run(var_tensor_list[0].assign(var_normal_list_vals[0][1]))

        # load weights batch normalization layers
        for i in range(len(bn_normal_list_vals)):
            sess.run(bn_tensor_list[i].assign(bn_normal_list_vals[i][1]))

        # load weights for remaining convolutional layers
        for (name, value) in conv_normal_list_vals:
            reference, tensorized = {}, {}
            reference["kernel"] = value

            ref_scope = name.split('kernel')[0]
            new_scope = ref_scope.replace(resnet_model.DEFAULT_SCOPE, method)
            new_kernels = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_scope)        

            if len(new_kernels) > 1:
                for v in new_kernels:
                    kernel_name = v.name.split(new_scope)[1].split(":")[0]
                    tensorized[kernel_name] = v
            else:
                print("THIS SHOULD NEVER BE CALLED IN THE CURRENT STRUCTURE. layer %s will use original layer" %new_kernels[0].name)
                logging.info("THIS SHOULD NEVER BE CALLED IN THE CURRENT STRUCTURE. layer %s will use original layer" %new_kernels[0].name)
                tensorized["kernel"] = new_kernels[0]

            block_name = name.split('/')[1]
            block_num = int(block_name.split('block')[1]) + 1
            if rate_decay == 'flat':
                cur_rate = compression_rate
            else:
                cur_rate = rate_function_linear(compression_rate, starting_rate, block_num, total_params, growth_params)

            load_params_conv2d_tensor(sess, reference, tensorized, method=method, rate=cur_rate)

        #load weight for the last layer (dense layer with bias)
        for i in range(len(dense_normal_list_vals)):      
            sess.run(dense_tensor_list[i].assign(dense_normal_list_vals[i][1]))

        new_saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope))
        new_saver.save(sess, checkpoint_file)
        print("phase0: resnet model saved to %s" %checkpoint_file)
        logging.info("phase0: resnet model saved to %s" %checkpoint_file)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
