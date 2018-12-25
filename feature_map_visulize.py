import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np


def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print (data.shape)
    plt.imshow(data)
    plt.show()


features = sio.loadmat('featuremaps/LSTM_side.mat')

# up_conv = features['up_conv5']
# conv_last = features['conv_last']
LSTM_input = features['LSTM_input']
tmp = LSTM_input[3, :, :, :] - LSTM_input[1, :, :, :]
tmp2 = LSTM_input[2, :, :, :] - LSTM_input[0, :, :, :]
tmp = tmp + tmp2
# minus_conv_last = LSTM_input[3, :, :, :] - LSTM_input[0, :, :, :]
num = 61
plt.subplot(1, 3, 1)
plt.imshow(LSTM_input[3, num, :, :])

plt.subplot(1, 3, 2)
plt.imshow(LSTM_input[0, num, :, :])

plt.subplot(1, 3, 3)
plt.imshow(tmp[num, :, :])
plt.show()
# vis_square(LSTM_input[0, :, :, :])
# vis_square(LSTM_input[1, :, :, :])
# vis_square(LSTM_input[1, :, :, :] - LSTM_input[0, :, :, :])

