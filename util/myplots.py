import pandas as pd
import matplotlib.pyplot as plt

from const import SEQUENCE_LENGTH

path = 'IDNet_interpolated'
user = 'u001'
recording = 'w001'
filename = path + '/' + user+'_'+recording+'_accelerometer.log'

df = pd.read_csv(filename, usecols = ['x','y','z'], header=0)
num_samples = df.shape[0]
num_frames = (int)(num_samples / SEQUENCE_LENGTH)
# drop the first and the last frame
# df = df[ 1*SEQUENCE_LENGTH : (num_frames-1) * SEQUENCE_LENGTH ]
data = df.values
print(data.shape)

x = data[:,0]
y = data[:,1]
z = data[:,2]

start_sequence = 0
end_sequence = 10

xx = x[start_sequence * SEQUENCE_LENGTH: end_sequence * SEQUENCE_LENGTH]
yy = y[start_sequence * SEQUENCE_LENGTH: end_sequence * SEQUENCE_LENGTH]
zz = z[start_sequence * SEQUENCE_LENGTH: end_sequence * SEQUENCE_LENGTH]

plt.plot(xx)
plt.plot(yy)
plt.plot(zz)
plt.title('Sequences: ' + str(start_sequence) + ' - ' + str(end_sequence))
plt.ylabel('Magnitude')
plt.xlabel('Time')
plt.legend(['x', 'y', 'z'], loc='upper left')
plt.show()
