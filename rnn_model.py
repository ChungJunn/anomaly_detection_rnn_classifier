import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_MODEL1(nn.Module):
    def __init__(self, dim_input, dim_lstm_hidden, dim_fc_hidden, dim_output):
        super(RNN_MODEL1, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden1 = dim_lstm_hidden
        self.dim_hidden2 = dim_fc_hidden
        self.rnn = nn.LSTM(input_size=dim_input, hidden_size=self.dim_hidden1)
        self.fc1 = nn.Linear(self.dim_hidden1, self.dim_hidden2)
        self.fc2 = nn.Linear(self.dim_hidden2, dim_output)
        self.relu = nn.ReLU()

    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input, hidden)

        output = self.fc1(output)
        output = self.relu(output)

        output = self.fc2(output)

        return F.log_softmax(output, dim=1)

    def init_hidden(self, Bn):
        hidden = torch.zeros(1, Bn, self.dim_hidden1)
        cell = torch.zeros(1, Bn, self.dim_hidden1)
        return hidden, cell

if __name__ == '__main__':
    model = RNN_MODEL1(106,64,32,2)
    x_data = torch.rand(32,32,106)

    hidden, cell = model.init_hidden(32)

    out = model(x_data, (hidden, cell))
    import pdb; pdb.set_trace()
