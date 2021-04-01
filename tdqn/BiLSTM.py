######################################
########## IMPORT LIBRARIES ##########
######################################
import torch
from torch import nn
import torch.nn.functional as F


################################################
#################### BiLSTM ####################
################################################
"""Bidirectional version of the above LSTM model"""
"""will argue in report that bidirectional may not be realistic in stock trading context"""

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super(BiLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size) for batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        x = x.reshape([x.shape[0], 1, x.shape[1]])

        # Set initial hidden states and cell states for LSTM)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0,c0))  
        
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        # x: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        x = x[:, -1, :]
        # x: (n, 128)
         
        x = self.fc(x)
        # x: (n, 10)
        return x