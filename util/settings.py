from util.const import FeatureType, AUTOENCODER_MODEL_TYPE, MEASUREMENT_PROTOCOL

##
#  Modify the values below to test the model with different parameters.
#



# Type of protocol used for measurements
#
MEASUREMENT_PROTOCOL_TYPE = MEASUREMENT_PROTOCOL.SAME_DAY

##
#  MANUAL - Use the 59 time-based features (these are already extracted)
#
#  AUTOMATIC - Use features extracted by an autoencoder
FEATURE_TYPE = FeatureType.AUTOMATIC

##
# What type of autoencoder to use for feature extraction
# Used only when FEATURE_TYPE = AUTOMATIC
#
# LSTM - Long Short Term Memory
#
# DENSE - Fully Connected layers
#
AUTOENCODER_TYPE = AUTOENCODER_MODEL_TYPE.LSTM

##
# If the features are already extracted using a given
# type of autoencoder, then set TRAIN_AUTOENCODER to False
#
TRAIN_AUTOENCODER = True
##
# Ignore the first and the last frames/windows of the raw signal
# This is used only in the case of deep features (autoencoders)
# In manually extracted features (59 features) these are already ignored.
IGNORE_FIRST_AND_LAST_FRAMES = True

##
#  Use all negative data for testing (False) - UNBALANCED
#  OR use num_positive samples from the negative data (True) - BALANCED
#
BALANCED_NEGATIVE_TEST_DATA = True

##
#  True  - data is segmented using the annotated step cycle boundaries
#  False - data is segmented into fixed length frames of 128
#
CYCLE = False

##
#  Number of consecutive cycles used for evaluation
#  To be varied between 1 and 10
#
NUM_CYCLES =5

##
#  True  - negative samples are selected from users of session1 
#          (registered)
#  False - negative samples are selected from users of session0
#          (unregistered: u11-u22)
#
REGISTERED_NEGATIVES = True






