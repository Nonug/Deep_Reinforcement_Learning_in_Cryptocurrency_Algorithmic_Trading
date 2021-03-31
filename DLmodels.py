######################################
########## IMPORT LIBRARIES ##########
######################################
from torch import nn
import torch.nn.functional as F


##############################################
#################### TDQN ####################
##############################################
class TDQN(nn.Module):
    """
    GOAL: Implementing the Deep Neural Network of the DQN Reinforcement 
          Learning algorithm.
    
    VARIABLES:  - fc1: Fully Connected layer number 1.
                - fc2: Fully Connected layer number 2.
                - fc3: Fully Connected layer number 3.
                - fc4: Fully Connected layer number 4.
                - fc5: Fully Connected layer number 5.
                - dropout1: Dropout layer number 1.
                - dropout2: Dropout layer number 2.
                - dropout3: Dropout layer number 3.
                - dropout4: Dropout layer number 4.
                - bn1: Batch normalization layer number 1.
                - bn2: Batch normalization layer number 2.
                - bn3: Batch normalization layer number 3.
                - bn4: Batch normalization layer number 4.
                                
    METHODS:    - __init__: Initialization of the Deep Neural Network.
                - forward: Forward pass of the Deep Neural Network.
    """

    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons=numberOfNeurons, dropout=dropout):
        """
        GOAL: Defining and initializing the Deep Neural Network of the
              DQN Reinforcement Learning algorithm.
        
        INPUTS: - numberOfInputs: Number of inputs of the Deep Neural Network.
                - numberOfOutputs: Number of outputs of the Deep Neural Network.
                - numberOfNeurons: Number of neurons per layer in the Deep Neural Network.
                - dropout: Droupout probability value (handling of overfitting).
        
        OUTPUTS: /
        """

        # Call the constructor of the parent class (Pytorch torch.nn.Module)
        super(TDQN, self).__init__()

        # Definition of some Fully Connected layers
        self.fc1 = nn.Linear(numberOfInputs, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)

        # Definition of some Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)

        # Definition of some Dropout layers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Xavier initialization for the entire neural network
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

    
    def forward(self, input):
        """
        GOAL: Implementing the forward pass of the Deep Neural Network.
        
        INPUTS: - input: Input of the Deep Neural Network.
        
        OUTPUTS: - output: Output of the Deep Neural Network.
        """

        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)
        return output
 


##############################################
#################### LSTM ####################
##############################################

"""Reference - https://github.com/python-engineer/pytorch-examples/blob/master/rnn-lstm-gru/main.py"""

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size) for batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Set initial hidden states and cell states for LSTM)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0,c0))  
        
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        # x: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        x = x[:, -1, :]
        # x: (n, 128)
         
        x = self.fc(out)
        # x: (n, 10)
        return x
  


################################################
#################### BiLSTM ####################
################################################
"""Bidirectional version of the above LSTM model"""
"""will argue in report that bidirectional may not be realistic in stock trading context"""

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size) for batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        
        self.fc = nn.Linear(2*hidden_size, output_size)
        
    def forward(self, x):
        # Set initial hidden states and cell states for LSTM)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        x, _ = self.lstm(x, (h0,c0))  
        
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        # x: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        x = x[:, -1, :]
        # x: (n, 128)
         
        x = self.fc(out)
        # x: (n, 10)
        return x
  
  
####################################################
#################### DuelingDQN ####################
####################################################

""" https://github.com/cyoon1729/deep-Q-networks """

  class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
  
########################################################
#################### ConvDuelingDQN ####################
########################################################

""" https://github.com/cyoon1729/deep-Q-networks """
class ConvDuelingDQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


