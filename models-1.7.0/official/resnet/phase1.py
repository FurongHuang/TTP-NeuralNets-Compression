# ==============================================================================
# Tensorized Spectrum Preserving Compression for Neural Networks
# https://arxiv.org/pdf/1805.10352.pdf
# Developed by Jiahao Su (jiahaosu@terpmail.umd.edu) and
# Jingling Li (jingling@cs.umd.edu)
# May 18, 2018
# ==============================================================================

import tensorflow as tf
import resnet_model
from resnet_run_loop import ResnetArgParser
from tensornet.utils import load_params_conv2d_tensor, load_params_dense_tensor
from resnet_model import DEFAULT_COMPRESSION_RATE
from cifar10_main import Cifar10Model, cifar10_model_fn
from imagenet_main import ImagenetModel, imagenet_model_fn
from cifar10_main import input_fn as cifar_input_fn
from imagenet_main import input_fn as imagenet_input_fn
from model_fns import cifar10_model_conversion_fn, imagenet_model_conversion_fn
from official.utils.logging import hooks_helper
import sys, os
import shutil
import logging
import time

def list_multiply(arr):
    res = 1
    for a in arr:
        res = res*a
    return res

def calculate_total_params(vars_list):
    total = 0
    for v in vars_list:
        cur_params = list_multiply(v.shape.as_list())
        total += cur_params
    return total

def main(argv):
    parser = ResnetArgParser()

    parser.add_argument(
        '--model_class', '-mc', default = 'cifar10',
        help = "[default: %(default)s] The model you are performing experiment on.",
        metavar = '<MC>'
    )

    parser.add_argument(
        '--output_path', '-op', default = '/tmp/cifar10_model_tensor_based',
        help = "[default: %(default)s] The location of the tensorized model of phase1.",
        metavar = '<OP>'
    )

    parser.add_argument(
        '--pretrained_model_dir', '-pmd', default = '/tmp/cifar10_model_tensor_based/normal/',
        help = "[default: %(default)s] The location of the pretrained model for phase0.",
        metavar = '<PMD>'
    )
    
    parser.add_argument(
        '--phase_zero', '-pz', default = '/home/jingling/models/cifar10/phase0',
        help = "[default: %(default)s] The directory where we stored the results from phase0",
        metavar = '<PZ>'
    )

    parser.add_argument(
        '--filename', '-fn', default = 'model.ckpt',
        help = "[default: %(default)s] The filename of checkpoint in phase0 and phase1.",
        metavar = '<FN>'
    )
    
    parser.add_argument(
        '--exp_growth', '-eg', default = False,
        help = "controls the weights we assigned to different components in the loss function",
        metavar = '<EG>'
    )

    parser.add_argument(
        '--continue_training', '-ct', default = -1, type=int, 
        help = "continue training from block i (-1 means do not continue training)",
        metavar = '<CT>'
    )

    parser.add_argument(
        '--continue_checkpoint_file', '-ccf', default = "",
        help = "the path tp the checkpoint file needed for continue training",
        metavar = '<CCF>'
    )


    # Set defaults that are reasonable for this model.
    parser.set_defaults(data_dir='/tmp/cifar10_data',
                    pretrained_model_dir='/home/jingling/models/cifar10/normal/', # our resnet model, not the official one
                    resnet_size=32,
                    batch_size=128,
                    version=2,
                    output_path='/home/jingling/models/cifar10/phase1', 
                    method='svd',
                    scope='svd',
                    train_epochs=50,
                    rate=0.5)

    flags = parser.parse_args(args=argv[1:])
    
    '''Define the parameters we need for each experiment'''
    if flags.model_class == 'cifar10':
        model_class, input_fn, model_fn = Cifar10Model, cifar_input_fn, cifar10_model_fn
        model_conversion_fn = cifar10_model_conversion_fn
        testing_every_n_epochs = 5
    else:
        model_class, input_fn, model_fn = ImagenetModel, imagenet_input_fn, imagenet_model_fn
        model_conversion_fn = imagenet_model_conversion_fn
        testing_every_n_epochs = 1

    data_dir = flags.data_dir

    resnet_size, batch_size, version = flags.resnet_size, flags.batch_size, flags.version
    method, scope = flags.method, flags.scope
    compression_rate, epoch_num = flags.rate, flags.train_epochs

    is_training = False
    
    pretrained_model_dir = flags.pretrained_model_dir 
    phase0_store = flags.phase_zero

    continue_training = flags.continue_training
    continue_checkpoint_file = flags.continue_checkpoint_file

    filename = flags.filename

    checkpoint_file = '%s/%s/rate%s/%s' %(phase0_store, method, compression_rate, filename)

    phase1_store= flags.output_path
    output_dir = '%s/%s/rate%s' %(phase1_store, method, compression_rate)

    if not os.path.exists(phase1_store):
        os.makedirs(phase1_store)

    logging.basicConfig(level=logging.INFO,
                    datefmt='%m-%d %H:%M',
                    filename='%s/%s_%s.log' %(phase1_store, method, compression_rate),
                    filemode='w+')

    logging.info("Starting phase1...")

    rate_decay = flags.rate_decay

    dataset = input_fn(is_training=False, data_dir=data_dir, batch_size=batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    x = tf.placeholder(tf.float32, shape = next_element[0].shape)
    y = tf.placeholder(tf.float32, shape = next_element[1].shape)

    model_method = model_class(resnet_size=resnet_size, method=method, scope=scope, rate=compression_rate, rate_decay=rate_decay)
    model_normal = model_class(resnet_size=resnet_size)

    logits_m, results_m  = model_method(x, is_training)
    logits_n, results_n = model_normal(x, is_training)

    predictions = tf.argmax(logits_m, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), predictions), tf.float32))

    optimizer = tf.train.AdamOptimizer(1e-3)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.makedirs(output_dir)

    if continue_training != -1:
        start_block = continue_training
        checkpoint_file = continue_checkpoint_file
    else:
        start_block = 0

    step = 1
    for block_i in range(len(results_m)):
        name = "block%d" %block_i
        output_file = '%s/%s_%s' %(output_dir, name, filename)
        cur_loss = tf.losses.mean_squared_error(results_m[name], results_n[name])

        if block_i == 0:
            current_checkpoint_file = checkpoint_file
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "%s/kernel" %scope)
            continue_training = -1
        else:
            if continue_training != -1:
                current_checkpoint_file = continue_checkpoint_file
                continue_training = -1
            else:
                current_checkpoint_file = '%s/block%d_%s-%d' %(output_dir, block_i-1, filename, step)
            train_vars = []
        
        train_vars.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "%s/block%d" %(scope, block_i))) 
        total_params = calculate_total_params(train_vars)
        train_phase_1 = tf.train.AdamOptimizer(1e-3).minimize(cur_loss, var_list = train_vars)     

        print("\nStart training %s which has %d parameters\n" %(name, total_params))   
        logging.info("\nStart training %s which has %d parameters\n" %(name, total_params))

        config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            saver_m = tf.train.Saver(var_list = [v for v in var_list if 'Adam' not in v.name])
            saver_m.restore(sess, current_checkpoint_file)
            
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='normal')
            saver_p = tf.train.Saver(var_list = [v for v in var_list if 'Adam' not in v.name])
            saver_p.restore(sess, tf.train.latest_checkpoint(pretrained_model_dir))
            
            for epoch in range(1, epoch_num + 1):
                dataset_tr = input_fn(is_training=True, data_dir=data_dir, batch_size=batch_size)
                iterator_tr = dataset_tr.make_initializable_iterator()
                next_element = iterator_tr.get_next()
                sess.run(iterator_tr.initializer)
                
                print("epoch %d: " % (epoch))
                logging.info("epoch %d: " % (epoch))
                t0 = time.time()
                
                while True:
                    try:
                        (features, labels) = sess.run(next_element)
                        train_phase_1.run(feed_dict = {x: features, y: labels})                
                        if step % 100 == 0:
                            train_accuracy = accuracy.eval(feed_dict = {x: features, y: labels})
                            mse_val = cur_loss.eval(feed_dict = {x: features, y: labels})
                            t1 = time.time()
                            print("step %d, training accuracy: %.6f" % (step, train_accuracy))
                            print("mse: %.6f (%.3f sec)" %(mse_val, t1-t0))
                            logging.info("step %d, training accuracy: %.6f" % (step, train_accuracy))
                            logging.info("mse: %.6f (%.3f sec)" %(mse_val, t1-t0))
                            t0 = time.time()

                        step = step + 1
                    except tf.errors.OutOfRangeError:
                        break
                
                if epoch % testing_every_n_epochs == 0:
                    dataset_test = input_fn(is_training=False, data_dir=data_dir, batch_size=batch_size)
                    iterator_test = dataset_test.make_initializable_iterator()
                    next_test_ele = iterator_test.get_next()
                    sess.run(iterator_test.initializer)
                        
                    test_accuracy = 0.0
                    num_examples = 0
                    while True:
                        try:
                            (features, labels) = sess.run(next_test_ele)
                            acc = accuracy.eval(feed_dict = {x: features, y: labels})
                            test_accuracy = test_accuracy + acc*labels.shape[0]
                            num_examples = num_examples + labels.shape[0]                    
                        except tf.errors.OutOfRangeError:
                            break                        
                        
                    test_accuracy = test_accuracy/(num_examples+0.0)
                    print("Epoch %d, step %d, testing accuracy: %.6f" % (epoch, step, test_accuracy))
                    logging.info("Epoch %d, step %d, testing accuracy: %.6f" % (epoch, step, test_accuracy))

                    saver_m.save(sess, output_file, global_step=step)
                    print("phase1: resnet model (%s) saved to %s with global_step %d" %(name, output_dir, step))
                    logging.info("phase1: resnet model (%s) saved to %s with global_step %d" %(name, output_dir, step))
            
            dataset_test = input_fn(is_training=False, data_dir=data_dir, batch_size=batch_size)
            iterator_test = dataset_test.make_initializable_iterator()
            next_test_ele = iterator_test.get_next()
            sess.run(iterator_test.initializer)
                        
            test_accuracy = 0.0
            num_examples = 0
            while True:
                try:
                    (features, labels) = sess.run(next_test_ele)
                    acc = accuracy.eval(feed_dict = {x: features, y: labels})
                    test_accuracy = test_accuracy + acc*labels.shape[0]
                    num_examples = num_examples + labels.shape[0]                    
                except tf.errors.OutOfRangeError:
                    break                        
                    
            test_accuracy = test_accuracy/(num_examples+0.0)
            print("%s final testing accuracy (epoch %d, step %d): %.6f" % (name, epoch, step, test_accuracy))
            logging.info("%s final testing accuracy (epoch %d, step %d): %.6f" % (name, epoch, step, test_accuracy))

            saver_m.save(sess, output_file, global_step=step)
            print("phase1: final resnet model (%s) saved to %s with global_step %d" %(name, output_dir, step))
            logging.info("phase1: final resnet model (%s) saved to %s with global_step %d" %(name, output_dir, step))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
