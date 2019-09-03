import pandas as pd
import numpy as np

from util.load_data import load_recordings_from_session, load_IDNet_data
from util.identification import evaluation
from autoencoder.autoencoder_dense import train_dense_autoencoder
from util.settings import MEASUREMENT_PROTOCOL_TYPE
import util.const as const

from random import random

import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

def correntropy_loss(sigma=0.2):
    """
    Source: https://github.com/fdavidcl/ae-review-resources
    """
    def robust_kernel(alpha):
        return 1. / (np.sqrt(2 * np.pi) * sigma) * K.exp(- K.square(alpha) / (2 * sigma * sigma))

    def loss(y_pred, y_true):
        return - K.sum(robust_kernel(y_pred - y_true))

    return loss

def custom_loss(layer):
    """
    Create a loss function that adds the MSE loss to the mean of all squared
    activations of a specific layer.
    """
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss

def add_random(x):
    """
    Auxiliary function for data augmentation
    """
    return x - 0.2 + 0.4 * random.random()

def add_array(data):
    return [add_random(y) for y in data]

def data_augmentation(data, ydata):
    """
    Data augmentation used for ZJU GaitAcc
    """
    data_random = add_array(data)
    data = np.concatenate((data, data_random), axis=0)
    ydata = np.concatenate((ydata, ydata), axis=0)
    return data, ydata

def create_zju_training_dataset(augmentation=False):
    """
    Create training dataset for autoencoder using the ZJU-GaitAcc dataset
    Users [1, 103)
    Recordings [1, 7)
    """
    modeltype = const.AutoencoderModelType.DENSE
    print('Create training dataset for autoencoder')
    feature_type = const.FeatureType.AUTOMATIC
    data1, ydata1 = load_recordings_from_session('session_1', 1, 103, 1, 7, modeltype, feature_type)
    data2, ydata2 = load_recordings_from_session('session_2', 1, 103, 1, 7, modeltype, feature_type)

    data = np.concatenate((data1, data2), axis=0)
    ydata = np.concatenate((ydata1, ydata2), axis=0)

    if augmentation is True:
        data, ydata = data_augmentation(data, ydata)

    print('ZJU GaitAcc dataset - Data shape: ' + str(data.shape) + ' augmentation: ' + str(augmentation))
    return data, ydata

def create_idnet_training_dataset():
    """
    Create training set for autoencoder using the IDNet dataset
    """
    print('Training autoencoder  - IDNet dataset')
    data = load_IDNet_data()
    ydata = np.full(data.shape[0], 1)
    print('IDNet dataset - Data shape: ' + str(data.shape))    
    return data, ydata

def extract_and_save_features(encoder, modeltype, start_user, stop_user):
    """
    Automatic features using the autoencoder
    ZJU-GaitAcc
    Extract and save features for session_1 and session_2
    
    encoder: the encoder part of a trained autoencoder
    modeltype: used for data shape
    users: [start_user, stop_user)
    Output: session_1.csv, session_2.csv
    """
    mysessions = ['session_1', 'session_2']
    feature_type = const.FeatureType.AUTOMATIC
    for session in mysessions:
        print('Extract features from ' + session)
        data, ydata = load_recordings_from_session(session, start_user, stop_user, 1, 7, modeltype, feature_type)
    
        print('data shape: ' + str(data.shape))
        if (modeltype == const.AutoencoderModelType.CONV1D):
            data = np.array(data)[:, :, np.newaxis]

        # Extract features
        encoded_frames = encoder.predict(data)

        # Normalize data
        scaled_data = preprocessing.scale(encoded_frames)

        num_features = encoded_frames.shape[1]

        # Concatenate features (encoded_frames) with labels (ydata)
        df1 = pd.DataFrame(data=scaled_data)
        df2 = pd.DataFrame(data=ydata)

        df2[0] = df2[0].apply(lambda x: x.replace('subj_', 'u'))
        df3 = pd.concat([df1, df2], axis=1)

        # Save data into a CSV file
        df3.to_csv('./' + const.FEAT_DIR + '/' + session + ".csv", header=False, index=False)

def extract_features(encoder, session, modeltype, start_user, stop_user, start_recording, stop_recording):
    """
    Users: [start_user, stop_user)
    Recordings: [start_recording, stop_recording)
    Extract and normalize features 
    """
    feature_type = const.FeatureType.AUTOMATIC
    data, ydata = load_recordings_from_session(session, start_user, stop_user, start_recording, stop_recording, modeltype, feature_type)    
    print('data shape: ' + str(data.shape))

    # Extract features
    encoded_frames = encoder.predict(data)
    # Normalize data
    #print(encoded_frames)
    scaled_data = preprocessing.scale(encoded_frames)
    #print('Scaling')
    #print(scaled_data)
    #print('features shape: ' + str(encoded_frames.shape))

    return scaled_data, ydata

def test_dense_autoencoder(train = False, augm = True, num_epochs = 10):
    """
    Autoencoder trained on ZJU-GaitAcc, Users: [1, 103)
    Evaluation on ZJU-GaitAcc, Users: [103, 154)
    """
    modelName = 'Dense_' + '_trained.h5'
    if train is True:
        data, ydata = create_zju_training_dataset(augm)
        # Train autoencoder
        encoder = train_dense_autoencoder(num_epochs, data, ydata)
        # saving the trained model    
        print('Saved model: ' + modelName)
        encoder.save(const.TRAINED_MODELS_DIR + '/' + modelName)
    else:
        # load model 
        modelName = const.TRAINED_MODELS_DIR+ '/' + modelName
        encoder = load_model(modelName)
        print('Loaded model: ' + modelName)
        print('session_1')
        X_train, y_train = extract_features(encoder, 'session_1', const.AutoencoderModelType.DENSE, 103, 154, 1, 5)
        X_test, y_test = extract_features(encoder, 'session_1', const.AutoencoderModelType.DENSE, 103, 154, 5, 7) 
        evaluation(X_train, y_train, X_test, y_test)
        print('session_2')
        X_train, y_train = extract_features(encoder, 'session_2', const.AutoencoderModelType.DENSE, 103, 154, 1, 5)
        X_test, y_test = extract_features(encoder, 'session_2', const.AutoencoderModelType.DENSE, 103, 154, 5, 7) 
        evaluation(X_train, y_train, X_test, y_test)

def test_IDNet_dense_autoencoder( train = False, num_epochs = 10):
    """
    Autoencoder trained on IDNet
    Evaluation on ZJU-GaitAcc, Users: [1, 154)
    """
    modelName = 'Dense_IDNet' + '_trained.h5'
    if train is True:
        data, ydata = create_idnet_training_dataset()
        # Train autoencoder
        encoder = train_dense_autoencoder( num_epochs, data, ydata)
        # saving the trained model    
        print('Saved model: ' + modelName)
        encoder.save(const.TRAINED_MODELS_DIR + '/' + modelName)
    else:
        # load model 
        print(const.TRAINED_MODELS_DIR)
        print(modelName)
        modelName = const.TRAINED_MODELS_DIR+ '/' + modelName
        print(modelName)
        encoder = load_model(modelName)
        print('Loaded model: ' + modelName)

        if MEASUREMENT_PROTOCOL_TYPE == const.MeasurementProtocol.SAME_DAY:
            print('session_1')
            X_train, y_train = extract_features(encoder, 'session_1', const.AutoencoderModelType.DENSE, 1, 154, 1, 5)
            X_test, y_test = extract_features(encoder, 'session_1', const.AutoencoderModelType.DENSE, 1, 154, 5, 7) 
            evaluation(X_train, y_train, X_test, y_test)
            print('session_2')
            X_train, y_train = extract_features(encoder, 'session_2', const.AutoencoderModelType.DENSE, 1, 154, 1, 5)
            X_test, y_test = extract_features(encoder, 'session_2', const.AutoencoderModelType.DENSE, 1, 154, 5, 7) 
            evaluation(X_train, y_train, X_test, y_test)
            
        if MEASUREMENT_PROTOCOL_TYPE == const.MeasureuentProtocol.CROSS_DAY:   
            X_train, y_train = extract_features(encoder, 'session_1', const.AutoencoderModelType.DENSE, 1, 154, 1, 7)
            X_test, y_test = extract_features(encoder, 'session_2', const.AutoencoderModelType.DENSE, 1, 154, 1, 7) 
            evaluation(X_train, y_train, X_test, y_test)
