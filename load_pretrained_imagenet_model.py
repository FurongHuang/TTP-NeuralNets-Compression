'''
This module is used to load the weights from a pretrained official resnet model  
trained on imagenet to our resnet model with modified scope. This module is needed
as we are using different variable scopes.
'''
import tensorflow as tf
import resnet_model
from resnet_run_loop import ResnetArgParser
from imagenet_main import ImagenetModel, input_fn
from model_fns import imagenet_model_conversion_fn
from utils.logging import hooks_helper
import sys, os
import shutil
 
def main(argv):
    parser = ResnetArgParser()

    parser.add_argument(
        '--output_path', '-op', default = '/tmp/cifar10_model_tensor_based',
        help = "[default: %(default)s] The location of the tensorized model of phase0.",
        metavar = '<OP>'
    )
    
    parser.add_argument(
        '--inter_store', '-is', default = '/tmp/intermediate_storage/',
        help = "[default: %(default)s] The tmp location of intermediate results",
        metavar = '<IS>'
        )

    # Set defaults that are reasonable for this model.
    parser.set_defaults(
                    filename='normal_weights.ckpt',
                    method='normal',
                    scope='normal',
                    rate_decay='flat')

    flags = parser.parse_args(args=argv[1:])
    
    '''
    Save the weights ftom original resnet model to our model with modified scopes.
    The variable names are changes. Assume they have the same structures
    '''
    
    checkpoint = tf.train.latest_checkpoint(flags.model_dir) + ".meta"
    saver = tf.train.import_meta_graph(checkpoint)
    var_p_values = []
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(flags.model_dir))
        
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        var = [v for v in var_list if 'Momentum' not in v.name]
        
        for i in range(1, len(var)):
            var_p_values.append(sess.run(var[i]))
    
    tf.reset_default_graph()
    
    model = ImagenetModel(flags.resnet_size, flags.data_format, version=flags.version)
    dataset = input_fn(is_training=False, data_dir=flags.data_dir, batch_size=flags.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    outputs = model(next_element[0], False)
       
    checkpoint_file = flags.inter_store + "/" + flags.filename #intermidate storage
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(len(var_list)):
            sess.run(var_list[i].assign(var_p_values[i]))
        
        new_saver = tf.train.Saver(var_list)
        new_saver.save(sess, checkpoint_file)
        
    '''
    Load the weights above (with modified names) into our resnet model 
    and save it via estimator
    '''
    session_config = tf.ConfigProto(
      inter_op_parallelism_threads=5,
      intra_op_parallelism_threads=10,
      allow_soft_placement=True)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                session_config=session_config)
    output_model_path = flags.output_path
    if os.path.exists(output_model_path):
        shutil.rmtree(output_model_path)
    else:
        os.makedirs(output_model_path)
    
    classifier = tf.estimator.Estimator(
      model_fn=imagenet_model_conversion_fn,
      model_dir=output_model_path,
      config=run_config, 
      params={
          'resnet_size': flags.resnet_size,
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'multi_gpu': True,
          'version': flags.version,
          'checkpoint': checkpoint_file,
          'method': flags.method,
          'scope': flags.scope,
          'rate': flags.rate,
          'rate_decay': flags.rate_decay,
      })
    
    train_hooks = hooks_helper.get_train_hooks(flags.hooks, batch_size=flags.batch_size)
    
    def input_fn_train():
        return input_fn(True, flags.data_dir, flags.batch_size, 1, 10, False)
    
    classifier.train(input_fn=input_fn_train, hooks=train_hooks, max_steps=1)
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
