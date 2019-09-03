import numpy as np

def average_absolute_difference(x, avg_x):
    y = np.abs(x - avg_x)
    return np.mean(y)

def zero_crossing_rate(x):
    counter = 0
    for i in range(1, len(x)):
        if (x[i-1] <= 0 and x[i] > 0) or (x[i-1] >= 0 and x[i] < 0):
            counter = counter + 1
    return counter / len(x)

    #zero_crossings2 = np.where(np.diff(np.sign(x)))[0]  
    #return len(zero_crossings2)/len(x)

# input: data - raw data segmented into frames of 128 samples (ax, ay, az, am) 128 x 4 = 512 values


def feature_extraction(data):
    #data = data.reshape(num_frames, SEQUENCE_LENGTH, num_features)
    print(np.shape(data))
    frames, _rows, _columns = np.shape(data)
    num_features = 59
    result = np.zeros((frames, num_features))
    # Extract 59 features from each row (512 values)
    for frame in range(0, frames):
        ax = data[frame, :, 0]
        ay = data[frame, :, 1]
        az = data[frame, :, 2]
        am = data[frame, :, 3]

        min_ax = np.min(ax)
        min_ay = np.min(ay)
        min_az = np.min(az)
        min_am = np.min(am)
        result[frame, 0] = min_ax
        result[frame, 1] = min_ay
        result[frame, 2] = min_az
        result[frame, 3] = min_am

        avg_ax = np.mean(ax)
        avg_ay = np.mean(ay)
        avg_az = np.mean(az)
        avg_am = np.mean(am)
        result[frame, 4] = avg_ax
        result[frame, 5] = avg_ay
        result[frame, 6] = avg_az
        result[frame, 7] = avg_am

        std_ax = np.std(ax)
        std_ay = np.std(ay)
        std_az = np.std(az)
        std_am = np.std(am)
        result[frame, 8]  = std_ax
        result[frame, 9]  = std_ay
        result[frame, 10] = std_az
        result[frame, 11] = std_am

        aad_ax = average_absolute_difference(ax, avg_ax)
        aad_ay = average_absolute_difference(ay, avg_ay)
        aad_az = average_absolute_difference(az, avg_az)
        aad_am = average_absolute_difference(am, avg_am)
        result[frame, 12] = aad_ax
        result[frame, 13] = aad_ay
        result[frame, 14] = aad_az
        result[frame, 15] = aad_am

        zcr_ax = zero_crossing_rate(ax)
        zcr_ay = zero_crossing_rate(ay)
        zcr_az = zero_crossing_rate(az)
        result[frame, 16] = zcr_ax
        result[frame, 17] = zcr_ay
        result[frame, 18] = zcr_az

        histo_ax, _bin_edges = np.histogram(ax, 10, (-1.5, 1.5), density=True)
        histo_ay, _bin_edges = np.histogram(ay, 10, (-1.5, 1.5), density=True)
        histo_az, _bin_edges = np.histogram(az, 10, (-1.5, 1.5), density=True)
        histo_am, _bin_edges = np.histogram(am, 10, (-2.0, 2.0), density=True)
        for i in range(0, 10):
            result[frame, 19 + i] = histo_ax[i]
        for i in range(0, 10):
            result[frame, 29 + i] = histo_ay[i]
        for i in range(0, 10):
            result[frame, 39 + i] = histo_az[i]
        for i in range(0, 10):
            result[frame, 49 + i] = histo_am[i]
    return result
