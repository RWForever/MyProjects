import torch
import torch.nn as nn
import torch.nn.functional as tf


class NetBP(torch.nn.Module):
    def __init__(self, n_features, n_hidden=50, n_output=1):
        # n_features input neurons num
        # n_hidden hidden neurons num
        # n_output output neurons num
        super(NetBP, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = tf.relu(self.hidden(x))
        x = self.predict(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=50, output_size=1, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.reg(x)


# Bi-LSTM
class BiLSTMNet(nn.Module):

    def __init__(self, input_size):
        super(BiLSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=50,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(100, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
