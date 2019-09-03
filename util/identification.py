from util.load_data import load_recordings_from_session
from util.const import RANDOM_STATE, MeasurementProtocol
from util.manual_feature_extraction import feature_extraction
from util.settings import MEASUREMENT_PROTOCOL_TYPE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(session, model_type, feature_type):
    """
    Load raw data
    if feature_type == FeatureType.MANUAL data is augmented with the magnitude
    model_type determines the shape of the data
    """
    # loads the data from the given session: 2/3 as training and 1/3 as testing
    if MEASUREMENT_PROTOCOL_TYPE == MeasurementProtocol.SAME_DAY:
        print('Training data SAME SESSION')
        X_train, y_train = \
            load_recordings_from_session(session, 1, 154, 1, 5, model_type, feature_type)
        print('Testing data SAME SESSION')
        X_test, y_test = \
            load_recordings_from_session(session, 1, 154, 5, 7, model_type, feature_type)
        return X_train, y_train, X_test, y_test
    
    # training data: session_1, testing data: session_2
    if MEASUREMENT_PROTOCOL_TYPE == MeasurementProtocol.CROSS_DAY:
        print('Training data CROSS SESSION')
        X_train, y_train = \
            load_recordings_from_session('session_1', 1, 154, 1, 7, model_type, feature_type)
        print('Testing data CROSS SESSION')
        X_test, y_test = \
            load_recordings_from_session('session_2', 1, 154, 1, 7, model_type, feature_type)
        return X_train, y_train, X_test, y_test



def dimensionality_reduction(data, num_components):
    """
    Dimensionality reduction with PCA
    """
    # Standardize Features
    sc = StandardScaler()
    # Fit the scaler to the features and transform
    data_std = sc.fit_transform(data)
    pca = PCA(num_components)
    pca.fit( data_std )
    data_pca = pca.transform(data_std)
    return data_pca 


def evaluation(X_train, y_train, X_test, y_test):
    """
    Performs evaluation with RandomForest classifier.
    """
    model = RandomForestClassifier(n_estimators = 100, random_state=RANDOM_STATE)
    #X_train = dimensionality_reduction(X_train, 59)
    model = model.fit(X_train, y_train)
    #X_test = dimensionality_reduction(X_test, 59)
    predictions = model.predict(X_test)
    #y_test = np.ravel(y_test)
    
    print('Score:'+ str(model.score(X_test, y_test)))
    #conf_matrix = confusion_matrix(y_test, predictions)
    #print(conf_matrix)

def test_identification_raw(model_type, feature_type):
    """
    Model_type should be AutoencoderModelType.DENSE, feature_type: FeatureType.RAW
    """
    if MEASUREMENT_PROTOCOL_TYPE == MeasurementProtocol.SAME_DAY:
        print('session1:')
        X_train, y_train, X_test, y_test = load_data('session_1', model_type, feature_type)
        evaluation(X_train, y_train, X_test, y_test)
        print('session2:')
        X_train, y_train, X_test, y_test = load_data('session_2', model_type, feature_type)
        evaluation(X_train, y_train, X_test, y_test)
    if MEASUREMENT_PROTOCOL_TYPE == MeasurementProtocol.CROSS_DAY:
        print('Training: session_1 + Testing: session_2')
        X_train, y_train, X_test, y_test = load_data( 'session', model_type, feature_type)
        evaluation(X_train, y_train, X_test, y_test)

def test_identification_59feat(model_type, feature_type):
    """
    model_type should be AutoencoderModelType.LSTM, feature_type: FeatureType.MANUAL
    """
    if MEASUREMENT_PROTOCOL_TYPE == MeasurementProtocol.SAME_DAY:
        print('session1:')
        X_train, y_train, X_test, y_test = load_data('session_1', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)
        print('session2:')
        X_train, y_train, X_test, y_test = load_data('session_2', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)

    if MEASUREMENT_PROTOCOL_TYPE == MeasruementProtocol.CROSS_DAY:
        X_train, y_train, X_test, y_test = load_data( 'session', model_type, feature_type)
        X_train = feature_extraction(X_train)
        X_test = feature_extraction(X_test)
        evaluation(X_train, y_train, X_test, y_test)




    

