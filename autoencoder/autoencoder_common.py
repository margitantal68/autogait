# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import pandas as pd

from util.load_data import load_recordings_from_session, load_IDNet_data
from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType, TRAINED_MODELS_DIR
from util.identification import evaluation
from autoencoder.autoencoder_dense import train_dense_autoencoder
from util.settings import MEASUREMENT_PROTOCOL_TYPE
from util.const import AUTOENCODER_MODEL_TYPE, RANDOM_STATE, MEASUREMENT_PROTOCOL

from random import random

import matplotlib.pyplot as plt


from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.initializers import glorot_uniform
my_init = glorot_uniform(seed_value)

# taken from: https://github.com/fdavidcl/ae-review-resources

def correntropy_loss(sigma = 0.2):
    def robust_kernel(alpha):
        return 1. / (np.sqrt(2 * np.pi) * sigma) * K.exp(- K.square(alpha) / (2 * sigma * sigma))

    def loss(y_pred, y_true):
        return -K.sum(robust_kernel(y_pred - y_true))

    return loss


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss


def add_random(x):
    return x -0.2 + 0.4* random.random()

def add_array( data ):
    return  [add_random(y) for y in data]

# Data augmentation used for ZJU GaitAcc
def data_augmentation(data, ydata):
    data_random = add_array( data )
    data = np.concatenate((data, data_random), axis=0)
    ydata = np.concatenate((ydata, ydata), axis=0)
    return data, ydata

# Create training dataset for ZJU - GaitAcc
# Users [1, 103)
# recordings [1, 7)
# 
def create_zju_training_dataset(augmentation = False):
    modeltype=AUTOENCODER_MODEL_TYPE.DENSE
    print('Create training dataset for autoencoder')
    feature_type = FeatureType.AUTOMATIC
    data1, ydata1 = load_recordings_from_session('session_1', 1, 103, 1, 7,  modeltype, feature_type)
    data2, ydata2 = load_recordings_from_session('session_2', 1, 103, 1, 7,  modeltype, feature_type)
   
    data = np.concatenate((data1, data2), axis=0)
    ydata = np.concatenate((ydata1, ydata2), axis=0)

    if augmentation == True:
        data, ydata = data_augmentation(data, ydata)

    print('ZJU GaitAcc dataset - Data shape: '+str(data.shape)+' augmentation: '+ str(augmentation))
    return data, ydata

# Create 
def create_idnet_training_dataset():
    print('Training autoencoder  - IDNet dataset')
    data = load_IDNet_data()
    ydata = np.full(data.shape[0], 1)
    print('IDNet dataset - Data shape: '+str(data.shape))    
    return data, ydata

    

#
# ZJU-GaitAcc
# Extracts and saves features for session_1 and session_2
# users: [start, stop)
# Output: session_1.csv, session_2.csv
# 
def extract_and_save_features(encoder, modeltype, start_user, stop_user):
    mysessions = ['session_1', 'session_2']
    feature_type = FeatureType.AUTOMATIC
    for session in mysessions:
        print('Extract features from '+session)
        data, ydata = load_recordings_from_session(session, start_user, stop_user, 1, 7,  modeltype, feature_type)
    
        print('data shape: '+ str(data.shape))
        if (modeltype==AUTOENCODER_MODEL_TYPE.CONV1D):
            data = np.array(data)[:, :, np.newaxis]

        # Extract features
        encoded_frames = encoder.predict(data)

        # Normalize data
        scaled_data = preprocessing.scale(encoded_frames)

        num_features = encoded_frames.shape[1]

        # Concatenate features(encoded_frames) with labels (ydata)
        df1 = pd.DataFrame(data=scaled_data)
        df2 = pd.DataFrame(data=ydata)

        df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
        df3 = pd.concat([df1, df2], axis=1)

        # Save data into a CSV file
        df3.to_csv('./'+FEAT_DIR + '/'+session + ".csv", header=False, index=False)

# [start_user, stop_user)
# [start_recording, stop_recording)
# Extracts features and normalizes them
# 
def extract_features(encoder, session, modeltype, start_user, stop_user, start_recording, stop_recording):
    feature_type = FeatureType.AUTOMATIC
    data, ydata = load_recordings_from_session(session, start_user, stop_user, start_recording, stop_recording,  modeltype, feature_type)    
    print('data shape: '+ str(data.shape))

    # Extract features
    encoded_frames = encoder.predict(data)
    # Normalize data
    #print(encoded_frames)
    scaled_data = preprocessing.scale(encoded_frames)
    #print('Scaling')
    #print(scaled_data)
    #print('features shape: ' + str(encoded_frames.shape))

    return scaled_data, ydata

# Autoencoder trained on ZJU-GaitAcc, Users: [1, 103)
# Evaluation on ZJU-GaitAcc, Users: [103, 154)
# 
def test_dense_autoencoder( train = False, augm = True, num_epochs = 10 ):
    modelName = 'Dense_' + '_trained.h5'
    if train == True:
        data, ydata = create_zju_training_dataset( augm )
        # Train autoencoder
        encoder = train_dense_autoencoder( num_epochs, data, ydata)
        # saving the trained model    
        print('Saved model: ' + modelName)
        encoder.save(TRAINED_MODELS_DIR + '/' + modelName)
    else:
        # load model 
        modelName = TRAINED_MODELS_DIR+ '/' + modelName
        encoder = load_model(modelName)
        print('Loaded model: ' + modelName)
        print('session_1')
        X_train, y_train = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.DENSE, 103, 154, 1, 5)
        X_test, y_test = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.DENSE, 103, 154, 5, 7) 
        evaluation(X_train, y_train, X_test, y_test)
        print('session_2')
        X_train, y_train = extract_features(encoder, 'session_2', AUTOENCODER_MODEL_TYPE.DENSE, 103, 154, 1, 5)
        X_test, y_test = extract_features(encoder, 'session_2', AUTOENCODER_MODEL_TYPE.DENSE, 103, 154, 5, 7) 
        evaluation(X_train, y_train, X_test, y_test)
    

# Autoencoder trained on IDNet
# Evaluation on ZJU-GaitAcc, Users: [1, 154)
# 
def test_IDNet_dense_autoencoder( train = False, num_epochs = 10):
    modelName = 'Dense_IDNet' + '_trained.h5'
    if train == True:
        data, ydata = create_idnet_training_dataset()
        # Train autoencoder
        encoder = train_dense_autoencoder( num_epochs, data, ydata)
        # saving the trained model    
        print('Saved model: ' + modelName)
        encoder.save(TRAINED_MODELS_DIR + '/' + modelName)
    else:
        # load model 
        print(TRAINED_MODELS_DIR)
        print(modelName)
        modelName = TRAINED_MODELS_DIR+ '/' + modelName
        print(modelName)
        encoder = load_model(modelName)
        print('Loaded model: ' + modelName)

        if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.SAME_DAY:
            print('session_1')
            X_train, y_train = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 1, 5)
            X_test, y_test = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 5, 7) 
            evaluation(X_train, y_train, X_test, y_test)
            print('session_2')
            X_train, y_train = extract_features(encoder, 'session_2', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 1, 5)
            X_test, y_test = extract_features(encoder, 'session_2', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 5, 7) 
            evaluation(X_train, y_train, X_test, y_test)
        if MEASUREMENT_PROTOCOL_TYPE == MEASUREMENT_PROTOCOL.CROSS_DAY:   
            X_train, y_train = extract_features(encoder, 'session_1', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 1, 7)
            X_test, y_test = extract_features(encoder, 'session_2', AUTOENCODER_MODEL_TYPE.DENSE, 1, 154, 1, 7) 
            evaluation(X_train, y_train, X_test, y_test)

        