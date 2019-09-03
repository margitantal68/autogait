import pandas as pd
import matplotlib.pyplot as plt

from util.const import SEED_VALUE, SEQUENCE_LENGTH, MY_INIT

from random import random
from keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model

def train_dense_autoencoder(num_epochs, data, ydata, activation_function='tanh'):
    """
    Trains a dense autoencoder and returns the encoder
    Can be used for both datasets: ZJU-GaitAcc and IDNet
    """
    print('Training autoencoder')
    X_train, X_test, y_train, y_test = train_test_split(
        data, ydata, test_size=0.2, random_state=SEED_VALUE)

    input_size = SEQUENCE_LENGTH * 3
    input_frame = Input(shape=(input_size,))
    x = Dense(input_size, activation=activation_function, kernel_initializer=MY_INIT)(input_frame)

    encoded1 = Dense(256, activation=activation_function, kernel_initializer=MY_INIT)(x)
    encoded2 = Dense(128, activation=activation_function, kernel_initializer=MY_INIT)(encoded1)
    encoded3 = Dense(64, activation=activation_function, kernel_initializer=MY_INIT)(encoded2)

    y = Dense(32, activation=activation_function, kernel_initializer=MY_INIT)(encoded3)

    #encoded4 = Dense(32, activation=activation_function, kernel_initializer=MY_INIT)(encoded3)
    #y = Dense(16, activation=activation_function, kernel_initializer=MY_INIT)(encoded4)
    #decoded4 = Dense(32, activation=activation_function, kernel_initializer=MY_INIT)(y)
  
    decoded3 = Dense(64, activation=activation_function, kernel_initializer=MY_INIT)(y)
    decoded2 = Dense(128, activation=activation_function, kernel_initializer=MY_INIT)(decoded3)
    decoded1 = Dense(256, activation=activation_function, kernel_initializer=MY_INIT)(decoded2)

    z = Dense(input_size, activation='linear', kernel_initializer=MY_INIT)(decoded1)
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
