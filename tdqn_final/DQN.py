######################################
########## IMPORT LIBRARIES ##########
######################################
import torch
from torch import nn
import torch.nn.functional as F

# Default paramter related to the hardware acceleration (CUDA)
GPUNumber = 0


##############################################
#################### TDQN ####################
##############################################
class DQN(nn.Module):
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

    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons, dropout):
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
        super(DQN, self).__init__()

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