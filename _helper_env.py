'''
A file that contains functions for experiment setup
'''
import os,random
import numpy as np
import tensorflow as tf
from keras import backend as K

def setup_seed(seed_value= 0):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value

    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value

    tf.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
