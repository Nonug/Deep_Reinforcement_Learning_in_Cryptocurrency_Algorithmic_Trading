######################################
########## IMPORT LIBRARIES ##########
######################################
import torch
from torch import nn
import torch.nn.functional as F

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


####################################################
#################### DuelingDQN ####################
####################################################

""" https://github.com/cyoon1729/deep-Q-networks """

class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim, dropout):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # self.fc1 = nn.Linear(self.input_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 1)
        # self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, self.output_dim)

        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.xavier_uniform_(self.fc4.weight)
        # nn.init.xavier_uniform_(self.fc5.weight)
        # nn.init.xavier_uniform_(self.fc6.weight)

        # self.dropout1 = nn.Dropout(self.dropout)
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(128, self.output_dim),
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals