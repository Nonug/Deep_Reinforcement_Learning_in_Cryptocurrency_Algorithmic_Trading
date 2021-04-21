######################################
########## IMPORT LIBRARIES ##########
######################################
import torch
from torch import nn
import torch.nn.functional as F

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


##############################################
#################### LSTM ####################
##############################################

"""Reference - https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py"""

class LSTM(nn.Module):

    def __init__(self, input_size, sequence_length, hidden_size, output_size, dropout):

        super(LSTM, self).__init__()
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.num_layers = 1
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size) for batch_first=True
        self.lstm1 = nn.LSTM(input_size, hidden_size//3, self.num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size//3, hidden_size//3*2, self.num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size//3*2, hidden_size, self.num_layers, batch_first=True)

        self.layerNorm1 = nn.LayerNorm([5,hidden_size//3])
        self.layerNorm2 = nn.LayerNorm([5,hidden_size//3*2])
        self.layerNorm3 = nn.LayerNorm([5,hidden_size])

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        x = x[:, :-1]
        x = x.reshape([x.shape[0], self.sequence_length, self.input_size])
        
        # Forward propagate RNN
        # x: (batch_size, input_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size//3).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size//3).to(self.device)
        x, _ = self.lstm1(x, (h0,c0))
        x = self.dropout1(F.leaky_relu(self.layerNorm1(x)))
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size//3*2).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size//3*2).to(self.device)
        x, _ = self.lstm2(x, (h0,c0))
        x = self.dropout2(F.leaky_relu(self.layerNorm2(x)))

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm3(x, (h0,c0))
        x = self.dropout3(F.leaky_relu(self.layerNorm3(x)))
        
        # Decode the hidden state of the last time step
        x = x[:, -1, :]
         
        x = self.fc(x)
        return x
  