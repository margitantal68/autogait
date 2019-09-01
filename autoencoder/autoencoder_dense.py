# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
import pandas as pd
import matplotlib.pyplot as plt

# from util.load_data import load_recordings_from_session, load_IDNet_data
# from util.const import seed_value, sessions, FEAT_DIR, ZJU_BASE_FOLDER, SEQUENCE_LENGTH, AUTOENCODER_MODEL_TYPE, FeatureType, TRAINED_MODELS_DIR
# from util.identification import evaluation
# from util.settings import IGNORE_FIRST_AND_LAST_FRAMES

from util.const import seed_value, SEQUENCE_LENGTH

from random import random
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model



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



# Trains a dense autoencoder and returns the encoder
# Can be used for both datasets: ZJU-GaitAcc and IDNet
# 
def train_dense_autoencoder( num_epochs, data, ydata, activation_function = 'tanh'):
    
    print('Training autoencoder')
    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=seed_value)

    input_size = SEQUENCE_LENGTH * 3
    input_frame = Input(shape=(input_size,))
    x = Dense(input_size, activation=activation_function, kernel_initializer=my_init)(input_frame)

    encoded1 = Dense(256, activation=activation_function, kernel_initializer=my_init)(x)
    encoded2 = Dense(128, activation=activation_function, kernel_initializer=my_init)(encoded1)
    encoded3 = Dense(64, activation=activation_function, kernel_initializer=my_init)(encoded2)
    
    y = Dense(32, activation=activation_function, kernel_initializer=my_init)(encoded3)
    
    #encoded4 = Dense(32, activation=activation_function, kernel_initializer=my_init)(encoded3)
    #y = Dense(16, activation=activation_function, kernel_initializer=my_init)(encoded4)
    #decoded4 = Dense(32, activation=activation_function, kernel_initializer=my_init)(y)
    
    
    decoded3 = Dense(64, activation=activation_function, kernel_initializer=my_init)(y)
    decoded2 = Dense(128, activation=activation_function, kernel_initializer=my_init)(decoded3)
    decoded1 = Dense(256, activation=activation_function, kernel_initializer=my_init)(decoded2)

    z = Dense(input_size, activation='linear', kernel_initializer=my_init)(decoded1)
    autoencoder = Model(input_frame, z)

    # encoder is the model of the autoencoder slice in the middle
    encoder = Model(input_frame, y)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')  # reporting the loss
    # autoencoder.compile(optimizer='adadelta', loss='kullback_leibler_divergence')  # reporting the loss
    # autoencoder.compile(optimizer='adadelta', loss='mean_squared_logarithmic_error')  # reporting the loss
    # autoencoder.compile(optimizer='adadelta', loss='mean_absolute_percentage_error')  # reporting the loss
    # autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')  # reporting the loss
    # autoencoder.compile(optimizer='adadelta', loss=correntropy_loss())  # reporting the loss
    print(autoencoder.summary())
    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epochs,
                              batch_size=128,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return encoder