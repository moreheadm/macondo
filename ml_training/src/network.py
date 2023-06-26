import torch.nn as nn
import torch

class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0, max=1)

class NNUE(nn.Module):
    def __init__(self, input_dim=6131, hidden_dim=256, activation=ClippedReLU()):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation = activation

    def forward(self, x, skip_input=False):
        if not skip_input:
            x = self.fc1(x)
        x = self.activation(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def first_layer(self, x):
        return self.fc1(x)

    def initialize_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)



