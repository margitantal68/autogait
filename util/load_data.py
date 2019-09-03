# Data processing for ZJU_GaitAcc dataset
import os
import glob
import pandas as pd
import numpy as np
from util.const import FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AutoencoderModelType, FeatureType, DROP_FRAMES, G3
from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

def files_of_a_folder(session, userid):
    path = ZJU_BASE_FOLDER + '/'+session + '/' + userid + '/'
    return os.listdir(path)

def users_of_a_folder(session):
    path = ZJU_BASE_FOLDER + '/'+session + '/'
    return os.listdir(path)

def read_recording(filename, modeltype, featuretype):
    """
    Reads the data from a CSV file and returns the data segmented into
    fixed length blocks
    """
    g = 9.8
    df = pd.DataFrame({'X': [],
                       'Y': [],
                       'Z': []})

    with open(filename) as f:
        lines = list(map(lambda line: [float(x) for x in line.strip().split(',')], f.readlines()))
        df['X'] = lines[0]
        df['Y'] = lines[1]
        df['Z'] = lines[2]

    data = df.values
    #     print(df.values.shape)
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_frames = (int)(num_samples / SEQUENCE_LENGTH)
    data = data[ 0 : num_frames * SEQUENCE_LENGTH]
    
    if featuretype == FeatureType.MANUAL:
        # compute magnitude
        mag = np.sqrt(np.sum(np.power(data, 2), axis=1))
        data = np.c_[data, mag]
        num_features = num_features + 1
    data = reshape_data(data, num_frames, num_features, modeltype)
    return data

def reshape_data(data, num_frames, num_features, modeltype):
    """
    Reshape types
    (1) frame: (x,y,z) x 128 = 384
    (2) frame: 128 x 3
    """
    # Drop the first and last frames
    if IGNORE_FIRST_AND_LAST_FRAMES is True:
        data = data[SEQUENCE_LENGTH : (num_frames-1) * SEQUENCE_LENGTH]
        num_frames = num_frames - 2

    #   DENSE autoencoder + DENOISING
    if modeltype == AutoencoderModelType.DENSE or modeltype == AutoencoderModelType.DENOISING:
        data = data.reshape(num_frames, SEQUENCE_LENGTH * num_features)

    #   LSTM autoencoder
    if modeltype == AutoencoderModelType.LSTM or modeltype == AutoencoderModelType.NONE:
        data = data.reshape(num_frames, SEQUENCE_LENGTH, num_features)

    #   1D Convolutional autoencoder
    if modeltype == AutoencoderModelType.CONV1D:
        data = data.reshape(num_frames, SEQUENCE_LENGTH * num_features)
    
    return data

def load_recordings_from_session(session, user_start, user_stop, rec_start, 
                                 rec_stop, modeltype, featuretype):
    """
    Loads the given recordings for all users of a session
    The range of recordings can be between 1..6
    ession: string 
    range of users [user_start, user_stop)
    range of recordings [rec_start, rec_stop)
    Examples 
    ('session_1', 1,  3, 1, 6, modeltype):  loads data of the first 2 users and the first 5 recordings of each user
    ('session_1', 1, 11, 1, 7, modeltype): loads data of the first 10 users and all the 6 recordings of each user
    """
    X_all = list()
    y_all = list()
    subjects = users_of_a_folder(session)
    SUBJECTS = subjects[user_start-1:user_stop-1]
    for subject in SUBJECTS:
        #print(subject)
        recordings = files_of_a_folder(session, subject)
        num_recordings = len(recordings)
        #print("\t"+recordings[rec_start-1] )
        X = read_recording(ZJU_BASE_FOLDER+'/' + session + '/' + subject + '/' + recordings[rec_start-1] + '/' + '3.txt', modeltype, featuretype)
        for i in range(rec_start, rec_stop-1):
            #print("\t"+recordings[i]) 
            X_temp = read_recording(ZJU_BASE_FOLDER+'/' + session + '/' + subject + '/' + recordings[i] + '/' + '3.txt', modeltype, featuretype)
            X = np.concatenate((X, X_temp), axis=0)
        y = np.full((X.shape[0], 1), subject)

        if len(X_all) == 0:
            X_all = X
            y_all = y
        else:
            X_all = np.concatenate((X_all, X), axis=0)
            y_all = np.concatenate((y_all, y), axis=0)
    print('X: ' + str(X_all.shape))
    print('y: ' + str(y_all.shape))

    return X_all, y_all

def test_load_data():
    X, y = load_recordings_from_session(
        'session_0', 1, 23, 1, 7, AutoencoderModelType.NONE, FeatureType.MANUAL)
    print('test_load_data: ' + str(X.shape))

def read_IDNet_file(filename):
    df = pd.read_csv(filename, usecols = ['x','y','z'])
    data = df.values
    #print(filename+": "+ str(df.shape))

    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_frames = (int)(num_samples / SEQUENCE_LENGTH)
    data = data[ 0 : num_frames * SEQUENCE_LENGTH]
    
    data = reshape_data(data, num_frames, num_features, AutoencoderModelType.DENSE)
    return data

def load_IDNet_data():
    path = 'IDNet_interpolated'
    all_files = glob.glob(path + "/*.log")

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, usecols = ['x','y','z'], header=0)
        num_samples = df.shape[0]
        num_frames = (int)(num_samples / SEQUENCE_LENGTH)
        # drop the first and the last frame
        df = df[ DROP_FRAMES*SEQUENCE_LENGTH : (num_frames-DROP_FRAMES) * SEQUENCE_LENGTH ]
        li.append(df)

    df = pd.concat(li, axis=0)
    print(df.shape)
    data = df.values
     
    data = data / G3
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_frames = (int)(num_samples / SEQUENCE_LENGTH)
    data = data[ 0 : num_frames * SEQUENCE_LENGTH]
    
    data = reshape_data(data, num_frames, num_features, AutoencoderModelType.DENSE)
    print( data.shape) 
    return data