
import resnet_model
import tensorflow as tf
import resnet_run_loop
from cifar10_main import Cifar10Model
from imagenet_main import ImagenetModel

def cifar10_model_conversion_fn(features, labels, mode, params):
  """Model conversion function for CIFAR-10."""

  _HEIGHT = 32
  _WIDTH = 32
  _NUM_CHANNELS = 3
  _NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
  }

  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
      decay_rates=[1, 0.1, 0.01, 0.001])

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(name):
    return True

  return resnet_model_conversion_fn(features, labels, mode, Cifar10Model,
                                         resnet_size=params['resnet_size'], 
                                         weight_decay=weight_decay,
                                         learning_rate_fn=learning_rate_fn,
                                         momentum=0.0,
                                         data_format=None,
                                         version=params['version'],
                                         checkpoint_file=params['checkpoint'],
                                         loss_filter_fn=loss_filter_fn,
                                         multi_gpu=params['multi_gpu'],                                         
                                         method=params['method'],
                                         scope=params['scope'],
                                         rate=params['rate'],
                                         rate_decay=params['rate_decay'])

def imagenet_model_conversion_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  
  _DEFAULT_IMAGE_SIZE = 224
  _NUM_CHANNELS = 3
  _NUM_CLASSES = 1001

  _NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
  }

  _NUM_TRAIN_FILES = 1024
  _SHUFFLE_BUFFER = 1500
  
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

  return resnet_model_conversion_fn(features, labels, mode, ImagenetModel,
                                         resnet_size=params['resnet_size'],
                                         weight_decay=1e-4,
                                         learning_rate_fn=learning_rate_fn,
                                         momentum=0.0,
                                         data_format=params['data_format'],
                                         version=params['version'],
                                         checkpoint_file=params['checkpoint'],
                                         loss_filter_fn=None,
                                         multi_gpu=params['multi_gpu'],
                                         method=params['method'],
                                         scope=params['scope'],
                                         rate=params['rate'],
                                         rate_decay=params['rate_decay'])

def resnet_model_conversion_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum, rate_decay,
                    data_format, version, checkpoint_file, method=resnet_model.DEFAULT_METHOD,
                    scope=resnet_model.DEFAULT_SCOPE, rate=resnet_model.DEFAULT_COMPRESSION_RATE,
                    loss_filter_fn=None, multi_gpu=False):

  print("rate== %f" %rate)
  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  model = model_class(resnet_size=resnet_size, data_format=data_format, version=version, 
                      method=method, scope=scope, rate=rate, rate_decay=rate_decay)

  logits, _ = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # Load some portion of the variables from pretrained data
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  var_names = [v.name for v in var_list[1:]]
  assignment_map = {}
  for v in var_names:
     assignment_map[v.split(':')[0]] = v.split(':')[0]
  
  print("assignment_map: ", len(assignment_map))
  tf.train.init_from_checkpoint(checkpoint_file, assignment_map)
  # Generate a summary node for the images

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  if not loss_filter_fn:
    def loss_filter_fn(name):
      return 'batch_normalization' not in name

  # Add weight decay to the loss.
  loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = 0

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if multi_gpu:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
