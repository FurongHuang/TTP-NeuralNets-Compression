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

def list_multiply(arr):
    res = 1
    for a in arr:
        res = res*a
    return res


def main(argv):
    parser = ResnetArgParser()

    parser.add_argument(
        '--model_class', '-mc', default = 'cifar10',
        help = "[default: %(default)s] The model you are performing experiment on.",
        metavar = '<MC>'
    )

    parser.add_argument(
        '--output_path', '-op', default = '/phase2/output_path',
        help = "[default: %(default)s] The location of the estimator model after phase2.",
        metavar = '<OP>'
    )
    
    parser.add_argument(
        '--phase_one', '-pz', default = '/Users/jinglingli/Study/CMSC/Spring_2018/Tensor_decompostion_DL/phase1',
        help = "[default: %(default)s] The directory where we stored the results from phase1",
        metavar = '<PZ>'
    )

    parser.add_argument(
        '--filename', '-fn', default = 'model.ckpt-9000',
        help = "[default: %(default)s] The filename of checkpoint in phase1.",
        metavar = '<FN>'
    )

    # Set defaults that are reasonable for this model.
    parser.set_defaults(data_dir='/tmp/cifar10_data',
                    resnet_size=32,
                    batch_size=128,
                    version=2,
                    output_path='/tmp/models/cifar10/phase2', 
                    method='cp',
                    scope='cp',
                    rate=0.15,
                    rate_decay='flat')

    flags = parser.parse_args(args=argv[1:])
    
    '''Define the parameters we need for each experiment'''
    if flags.model_class == 'cifar10':
        model_class, input_fn, model_fn = Cifar10Model, cifar_input_fn, cifar10_model_fn
        model_conversion_fn = cifar10_model_conversion_fn
    else:
        model_class, input_fn, model_fn = ImagenetModel, imagenet_input_fn, imagenet_model_fn
        model_conversion_fn = imagenet_model_conversion_fn
        
    data_dir = flags.data_dir

    resnet_size, batch_size, version = flags.resnet_size, flags.batch_size, flags.version
    method, scope = flags.method, flags.scope
    compression_rate, epoch_num = flags.rate, flags.train_epochs
    
    phase1_store, output_path, filename = flags.phase_one, flags.output_path, flags.filename

    checkpoint_file = '%s/%s/rate%s/%s' %(phase1_store, method, compression_rate, filename)
           
    session_config = tf.ConfigProto(
      device_count={'GPU':1},
      inter_op_parallelism_threads=5,
      intra_op_parallelism_threads=10,
      allow_soft_placement=True)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                session_config=session_config)
    
    model_output_dir = "%s/%s/rate%s/" %(output_path, method, compression_rate)
    
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)
    else:
        os.makedirs(model_output_dir)
    

    classifier = tf.estimator.Estimator(
      model_fn=model_conversion_fn,
      model_dir=model_output_dir, 
      config=run_config, 
      params={
          'resnet_size': resnet_size,
          'data_format': None,
          'batch_size': batch_size,
          'multi_gpu': flags.multi_gpu,
          'version': version,
          'checkpoint': checkpoint_file,
          'method': method,
          'scope': scope,
          'rate': compression_rate,
          'rate_decay': flags.rate_decay,
      })
    
    train_hooks = hooks_helper.get_train_hooks(flags.hooks, batch_size=batch_size)
    
    def input_fn_train():
        return input_fn(True, data_dir, batch_size, 1, 10, False)
    
    classifier.train(input_fn=input_fn_train, hooks=train_hooks, max_steps=1)
    
    print("phase2 model saved to %s" %model_output_dir)
    
    def input_fn_eval():
        return cifar_input_fn(False, data_dir, batch_size, 1, 10, False)
    
    eval_results = classifier.evaluate(input_fn=input_fn_eval, steps=None)
    print(eval_results)
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
