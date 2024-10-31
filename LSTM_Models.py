import torch
import torch.nn as nn
from torch.autograd import Variable
from AttentionCell import Attention
from LSTMCells import LSTMCell, VLSTMCell

class LSTM(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_layers=1, bias=True, output_size=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(LSTMCell(self.input_size,self.hidden_size,self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(LSTMCell(self.hidden_size,self.hidden_size,self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


class VLSTM(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_layers=1, bias=True, output_size=1):
        super(VLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(VLSTMCell(self.input_size,self.hidden_size,self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(VLSTMCell(self.hidden_size,self.hidden_size,self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
    

class AMVLSTM(nn.Module):
    def __init__(self, input_size=60, hidden_size=128, num_layers=1, bias=True, output_size=1):
        super(AMVLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        self.attention = Attention(input_size)

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(VLSTMCell(self.input_size,self.hidden_size,self.bias))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(VLSTMCell(self.hidden_size,self.hidden_size,self.bias))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hx=None):
        # Input of shape (batch_size, seqence length , input_size)
        #
        # Output of shape (batch_size, output_size)
        input = self.attention(input)
        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out