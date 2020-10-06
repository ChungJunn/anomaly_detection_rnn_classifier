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

class RNN_ESM(nn.Module):
    def __init__(self, n_cmt, dim_input, dim_lstm_hidden, dim_fc_hidden, dim_output, reload=False, loaddir=None, loadfile=None):
        self.dim_input = dim_input
        self.dim_lstm_hidden = dim_lstm_hidden
        self.dim_fc_hidden = dim_fc_hidden
        self.dim_out = dim_output
        self.n_cmt = n_cmt
        self.model_list = []
        
        if reload:
            self.load(loaddir, loadfile)

        else:
            for i in range(n_cmt):
                model = RNN_MODEL1(dim_input, dim_lstm_hidden, dim_fc_hidden, dim_output)
                self.model_list.append(model)
    
    def forward(self, input):
        out = torch.zeros((input.shape[0], input.shape[1], self.dim_out)).to(input.device)
        # output: (Tx Bn 2) -> (Bn 2)

        for model in self.model_list:
            out += model(input)

        out = out / self.n_cmt

        return out

    '''
    def save(self, savedir, savefile):
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        for i in range(self.n_cmt):
            torch.save(self.model_list[i], savedir + '/' + savefile + str(i))
    '''

    def load(self, loaddir, loadfile):
        for i in range(self.n_cmt):
            model = torch.load(loaddir + '/' + loadfile + str(i))
            self.model_list.append(model)

    def __call__(self, input):
        return self.forward(input)
    
    def to(self, device):
        for i in range(self.n_cmt):
            self.model_list[i].to(device)

        return self

    def eval(self):
        for i in range(self.n_cmt):
            self.model_list[i].eval()
        return

if __name__ == '__main__':
    model = RNN_MODEL1(106,64,32,2)
    x_data = torch.rand(32,32,106)

    hidden, cell = model.init_hidden(32)

    out = model(x_data, (hidden, cell))
    import pdb; pdb.set_trace()
