import numpy as np
import torch
import random
import pickle as pkl

class AD_RNNIterator:
    def __init__(self, filename, stat_file, batch_size):
        with open(filename, 'rb') as fp:
            self.in_nums = pkl.load(fp) # data_len x rnn_len+1 x data_dim

        self.idx = 0
        self.b_size = batch_size
        self.rnn_len = self.in_nums.shape[1] - 1
        self.in_n = self.in_nums.shape[2]
        self.data_len = len(self.in_nums)

        fp = open(stat_file, 'r')
        lines = fp.readlines()
        self.x_avg= np.asarray([float(s) for s in lines[0].split(',')])
        self.x_std= np.asarray([float(s) for s in lines[1].split(',')])
        fp.close()

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0

    def __next__(self):
        x_data = np.zeros((self.b_size, self.rnn_len, self.in_n))
        y_data = np.zeros((self.b_size, 1, self.in_n))
        end_of_data=0

        b_len=0
        for i in range(self.b_size):
            if self.idx+i >= self.data_len:
                self.reset()
                end_of_data=1

            x_data[i,:,:] = self.in_nums[self.idx+i,:-1,:]
            y_data[i,:,:] = self.in_nums[self.idx+i,-1,:]
            b_len += 1

        # obtain labels
        x_data = x_data[:b_len,:,:]
        y_data = y_data[:b_len,:,:]
        self.idx += b_len

        # split into data and label
        x_data = x_data[:,:,:-1]
        y_label = y_data[:,:,-1].astype(np.int32) # (bsz, 1)
        y_data = y_data[:,:,:-1]

        # normalize
        x_data = self.prepare_data(x_data)
        y_data = self.prepare_data(y_data)

        # transpose
        x_data = np.transpose(x_data, (1,0,2))
        y_data = np.transpose(y_data, (1,0,2))

        # convert to tensor
        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.float32)
        return x_data, y_data, None, y_label, end_of_data # (T B E), (B E)

    def prepare_data(self, arr):

        data = (arr - self.x_avg) / self.x_std

        return data

if __name__ == "__main__":
    filename = '/home/mi-lab02/autoregressor/data/cnsm_exp2_1_data/ar_train.rnn_len16.pkl'
    stat_file = '/home/mi-lab02/autoregressor/data/cnsm_exp2_1_data/ar_train.rnn_len16.pkl.stat'

    iter = AD_RNNIterator(filename, stat_file, batch_size=32)

    for iloop, (x_data, y_data, x_label, y_label, end_of_data) in enumerate(iter):
        if iloop % 1000 == 0:
            print(iloop)
