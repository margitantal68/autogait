from enum import Enum
import numpy as np
import os

# RAW data directory
ZJU_BASE_FOLDER = './zju-gaitacc'
IDNET_BASE_FOLDER = './idnet'

# Directory where feature CSVs are stored
FEAT_DIR = 'features'

# Temporary files
TEMP_DIR = 'temp'

# Trained models directory
TRAINED_MODELS_DIR = 'models'

# Range of users from session0 to be used as negative samples
NEG_USER_RANGE = range(12, 23)

# Total number of users
NUM_USERS = 153

# ZJU: folder names for sessions + number of subjects
SESSIONS = {'session_0': 22, 'session_1': 153, 'session_2': 153}

# frame size for raw data processing
SEQUENCE_LENGTH = 128

# Constant for converting from Nanoseconds to seconds
TIME_CONV = 1000000000

class FeatureType(Enum):
    """
    There are three types of features
    RAW:    Using the raw data as features
    MANUAL: 59 manually designed features
    AUTOMATIC: Features extracted by different types of Autoencoders
    """
    RAW = 0
    MANUAL = 1
    AUTOMATIC = 2

class BiometricSystemType(Enum):
    """
    There are two types of biometric systems
    IDENTIFICATION: classification, builds an 1:N classifier
    VERIFICATION: authentication, builds N binary classifiers
    """
    IDENTIFICATION = 0
    VERIFICATION = 1

class AutoencoderModelType(Enum):
    """
    Type of autoencoder used for feature extraction
    NONE - no reshape
    DENSE, DENOISING - 384
    LSTM - 128 x 3 (SEQUENCE_LENGTH x num_features)
    """
    NONE = 0 
    DENSE = 1
    LSTM = 2
    DENOISING = 3
    CONV1D = 4
    VARIATIONAL = 5

class MeasurementProtocol(Enum):
    """
    Type of measurement
    SAME_DAY: both training and test data are taken from the same session. Our case: session1
    CROSS_DAY: training data are taken from session2, while testing data are taken from session2
    MIXED_DAY: both training and test data are taken from session1 and session2 
    (first half for training and second half for testing)
    """
    SAME_DAY = 0
    CROSS_DAY = 1
    MIXED_DAY = 2

# Random states for reproducibility
SEED_VALUE = 0
RANDOM_STATE = np.random.seed(SEED_VALUE)
RANDOM_STATE_SAMPLE = 1

# Number of steps to drop - used for IDNet
DROP_FRAMES = 4

G3 = 29.4

# Reproducibility

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(SEED_VALUE)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED_VALUE)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(SEED_VALUE)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.initializers import glorot_uniform
MY_INIT = glorot_uniform(SEED_VALUE)