import numpy as np
import torch
import random
import pickle as pkl

class RNNIterator:
    def __init__(self, filename, stat_file, batch_size):
        with open(filename, 'rb') as fp:
            self.in_nums = pkl.load(fp) # data_len x (rnn_len+1) x data_dim

        self.idx = 0
        self.b_size = batch_size
        self.rnn_len = self.in_nums.shape[1]
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
        y_data = np.zeros((self.b_size, 1))
        end_of_data=0

        b_len=0
        for i in range(self.b_size):
            if self.idx+i >= self.data_len:
                self.reset()
                end_of_data=1

            x_data[i,:,:] = self.in_nums[self.idx+i,:,:]
            y_data[i,-1] = self.in_nums[self.idx+i,-1,-1]
            b_len += 1
        self.idx += b_len

        # exclude label column
        x_data = x_data[:,:,:-1] # exclude label
        x_data = self.prepare_data(x_data)

        # transpose
        x_data = np.transpose(x_data, (1,0,2))

        # convert to tensor
        x_data, y_data = torch.tensor(x_data).type(torch.float32), torch.tensor(y_data).type(torch.int32)
        return x_data, y_data, end_of_data # (T B E), (B E)

    def prepare_data(self, arr):

        data = (arr - self.x_avg) / self.x_std

        return data

if __name__ == "__main__":
    filename = '/home/mi-lab02/autoregressor/data/cnsm_exp2_1_data/ar_train.rnn_len16.pkl'
    stat_file = '/home/mi-lab02/autoregressor/data/cnsm_exp2_1_data/ar_train.rnn_len16.pkl.stat'

    iter = AD_RNNIterator(filename, stat_file, batch_size=32)

    for iloop, (x_data, y_data, end_of_data) in enumerate(iter):
        if iloop % 1000 == 0:
            print(iloop)
            import pdb; pdb.set_trace()
