######################################
########## IMPORT LIBRARIES ##########
######################################
import torch
from torch import nn
import torch.nn.functional as F

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


########################################################
#################### ConvDuelingDQN ####################
########################################################

class ConvDuelingDQN(nn.Module):
    def __init__(self, in_channels, h, w, num_actions, dropout):
        super(ConvDuelingDQN, self).__init__()
        self.in_channels = in_channels
        self.h = h
        self.w = w
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)

        self.fc1_adv = nn.Linear(in_features=h*w*128, out_features=512)
        self.fc1_val = nn.Linear(in_features=h*w*128, out_features=512)

        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1_adv.weight)
        nn.init.xavier_uniform_(self.fc1_val.weight)
        nn.init.xavier_uniform_(self.fc2_adv.weight)
        nn.init.xavier_uniform_(self.fc2_val.weight)

    def forward(self, x):
        # x with shape (minibatch,in_channels,iH,iW)
        x = x[:, :-1]
        x = x.reshape(x.shape[0], self.in_channels, self.h, self.w)

        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.bn1(self.fc1_adv(x)))
        val = self.relu(self.bn1(self.fc1_val(x)))
        adv = self.dropout1(adv)
        val = self.dropout1(val)

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x