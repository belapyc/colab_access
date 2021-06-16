'''
A file that contains functions for experiment setup
'''
import os,random
import numpy as np
import tensorflow as tf
from keras import backend as K
import logging

loggerHelper = logging.getLogger("MainLogger.helper")

def setup_seed(seed_value= 0):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    loggerHelper.info("Setting seeds for reproducibility...")
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
    loggerHelper.info("Seeds are set.")

def set_logger(level, logger):
    '''
    Setting the levels for logger output.
    Creates File and Stream handlers for outputting messages to a file and
    console respectively.
    File output level is set to DEBUG, ie all of the messages will be output
    to the file regardless of the level
    Console output level can be set by user through passing --verbose
    argument in command line. The default level for console output is WARNING
    Logging levels:
    10 - debug, 20 - info, 30 - warning, 40 - error, 50 - critical

    :level: integer, representing level of output to console
    # https://docs.python.org/3/howto/logging-cookbook.html
    # Using logging in multiple modules (for further reference)
    '''

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')



    # Creating console handler for the logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logPath = "logs"
    fileName = "logLauncher.log"
    should_roll_over = os.path.isfile(logPath+"/"+fileName)
    # TODO: logs dont output to file
    # Creating file handler for logger
    # fh = logging.handlers.RotatingFileHandler(logPath+"/"+fileName, backupCount=5)
    # if should_roll_over:  # log already exists, roll over!
    #     fh.doRollover()
    # fh.setFormatter(formatter)
    # fh.setLevel(logging.DEBUG)

    #logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Logger is set.")
