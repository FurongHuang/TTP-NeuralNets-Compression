
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from official.utils.logging import hooks_helper


# In[2]:

# We want to check that each parameter is loaded correctly
modified_model_dir = '/home/jingling/models/cifar10/phase2/rtt/rate0.1/'
filename_m = 'model.ckpt-1.meta'
pretrained_model_dir = '/home/jingling/models/cifar10/phase2/rtt/rate0.1/'
filename = 'model.ckpt-1.meta'#'model.ckpt-97675.meta'

# In[3]:

var_m_values = []
with tf.Session() as sess:
    saver_m = tf.train.import_meta_graph(modified_model_dir+filename_m)
    saver_m.restore(sess, tf.train.latest_checkpoint(modified_model_dir))
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_m = [v for v in var_list]
    
    for i in range(len(var_m)):
        o_m = sess.run(var_m[i])
        var_m_values.append((var_m[i].name, o_m))
    
tf.reset_default_graph()

var_p_values = []
with tf.Session() as sess:
    saver_p = tf.train.import_meta_graph(pretrained_model_dir+filename)
    saver_p.restore(sess, tf.train.latest_checkpoint(pretrained_model_dir))
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_p = [v for v in var_list]
    
    for i in range(len(var_p)):
        o_p = sess.run(var_p[i])
        var_p_values.append((var_p[i].name, o_p))


# In[ ]:

diff = []
for i in range(1, len(var_m_values)):
    diff.append(np.linalg.norm(var_m_values[i][1] - var_p_values[i][1]))
print(diff)

# In[ ]:



