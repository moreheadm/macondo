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

    def first_layer(self, x, without_bias=False):
        res = self.fc1(x)
        if without_bias:
            res -= self.fc1.bias
        return res

    def initialize_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

#    def test_nnue(self, next_state, next_action):
#
#        self.print = True
#
#        accum = self.first_layer(next_state, without_bias=True)
#        next_action_value = self.first_layer(next_action)
#
#        combined_after = next_action_value + accum
#        combined_before = self.first_layer(next_action + next_state)
#        
#        print('Inner', combined_after - combined_before)
#
#        nnue_out = self(combined_after, skip_input=True)
#        print(nnue_out, self(next_action + next_state))

#        self.print = False
        

