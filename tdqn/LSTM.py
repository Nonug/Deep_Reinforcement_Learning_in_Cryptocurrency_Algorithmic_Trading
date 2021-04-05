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

    def __init__(self, input_size, sequence_length, hidden_size, num_layers, output_size):

        super(LSTM, self).__init__()
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size) for batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x[:, :-1]
        x = x.reshape([x.shape[0], self.sequence_length, self.input_size])

        # Set initial hidden states and cell states for LSTM)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # x: (batch_size, sequence_length, input_size), h0: (num_layers, n, hidden_size)
        
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0,c0))  
        # x: (batch_size, input_size, hidden_size)
        
        # Decode the hidden state of the last time step
        x = x[:, -1, :]
        # x: (batch_size, hidden_size)
         
        x = self.fc(x)
        # x: (batch_size, output_size)
        return x
  