import click
import warnings
import numpy as np

from util.statistics import main_statistics
from util.load_data  import load_recordings_from_session, load_IDNet_data
from util.const import AutoencoderModelType, FeatureType
from util.manual_feature_extraction import feature_extraction
from util.identification import test_identification_raw, test_identification_59feat
from autoencoder.autoencoder_common import  test_dense_autoencoder, test_IDNet_dense_autoencoder

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')

def main():
    #main_statistics()
    #X, y = load_recordings_from_session('session_0', 1, 2, 1, 2, AutoencoderModelType.NONE)
    #print('test_load_data: ' + str(X.shape))
    #X = feature_extraction(X)
    test_identification_raw(AutoencoderModelType.DENSE, FeatureType.RAW)
    # test_identification_59feat(AutoencoderModelType.LSTM, FeatureType.MANUAL)

   
    # test_dense_autoencoder(train=False, augm=True, num_epochs=10)
    # test_IDNet_dense_autoencoder(train=False, num_epochs=10)
    
if __name__ == '__main__':
    main()
    
