import os
import sys
import csv

import numpy as np
import pandas as pd

from util.const import TEMP_DIR, IDNET_BASE_FOLDER, ZJU_BASE_FOLDER, TIME_CONV

def IDNet_statistics(path, filename):
    """
    Create a csv with basic statistics of IDNet dataset
    """
    csv_file = open(TEMP_DIR + '/' + filename, mode='w')
    csv_file.write('file, numsamples, mean_dt, std_dt, fs, total_time (min)\n')
    dataset_total_time = 0
    
    users_file = open(TEMP_DIR + '/' + 'idnet_users.csv', mode='w')
    users_file.write('user, time\n')
    fs_list = []
    users = {}
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isdir(current):
            filename = current + '/' + file + '_accelerometer.log'
            df = pd.read_csv(filename, delimiter='\t')
            t = df['accelerometer_timestamp']	
            dt= np.diff(t)
            mean_t = np.mean(dt)
            std_t = np.std(dt)
            #accelerometer_x_data	accelerometer_y_data	accelerometer_z_data
            linecounter = df.shape[0]    
            fs = 1 / (mean_t / TIME_CONV )
            fs_list.append(fs)
            total_time = (t[t.size-1] - t[ 0 ]) / (60 * TIME_CONV) 
            dataset_total_time = dataset_total_time + total_time
            line = file + ', ' + str(linecounter) + ', ' + str(mean_t) + ', ' + \
                str(std_t) + ', ' + str(fs) + ', ' + str(total_time)
            print(line)
            csv_file.write(line + "\n")
            username = file[0:4]
            time = users.get(username)
            if time is None:
                users[username] = total_time
            else:
                users[username] = time + total_time


    csv_file.close()
    print('Sampling rate - MIN: ' + str(np.min(fs_list)) + ' MAX: ' + str(np.max(fs_list)) + \
        ' MEAN: ' + str(np.mean(fs_list)) + ' STD: ' + str(np.std(fs_list)))
    print('IDNet total time: ' + str(dataset_total_time / 60) +' hours')

    for (key, value) in users.items():
        users_file.write(key + ", " + str(value) + '\n')
    users_file.close()

def ZJU_statistics(basepath, sessiondir, numusers, output_file):
    """
    Create a csv with basic statistics of ZJU-GaiAcc dataset
    """
    csv_file = open(TEMP_DIR + '/' + output_file, mode='w')
    csv_file.write('file, total_time (min)\n')
    path = basepath + '/' + sessiondir
    print(path)
    session_time = 0
    for file in os.listdir(path):
        current = os.path.join(path, file)
        if os.path.isdir(current):
            user_time = 0
            for i in range(1,7):
                filename = current + '/rec_' + str(i) + '/cycles.txt'
                df = pd.read_csv(filename, header = None, delimiter=',')
                data = df.values
                n = data[-1].shape[0]
                x = data[ -1]
                total_time = x[n-1]
                user_time = user_time + total_time / 6000
            line = file + ', ' + str(user_time)
            csv_file.write(line + "\n")
            session_time = session_time + user_time
    csv_file.close()
    print(sessiondir + ': ' + str(session_time) + ' minutes')
    return session_time

def main_statistics():
    #path = IDNET_BASE_FOLDER
    #print(path)
    #IDNet_statistics(path, 'idnet_statistics.csv')

    path = ZJU_BASE_FOLDER
    t0 = ZJU_statistics(path, 'session_0', 22, 'zju_session0.csv')
    t1 = ZJU_statistics(path, 'session_1', 153, 'zju_session1.csv')
    t2 = ZJU_statistics(path, 'session_2', 153, 'zju_session2.csv')
    print('Total time: ' + str((t0 + t1 + t2) / 60) +' hours')
