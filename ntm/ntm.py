import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead
from functools import partial


class FM(nn.Module):
    def __init__(self, features_num=None, k=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features_num, k), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.xavier_uniform_(self.bias.data)
        self.linear = nn.Linear(features_num, 1)

    def forward(self, X):
        out_1 = ((X @ self.weight) ** 2).sum(1, keepdim=True) + self.bias
        out_2 = ((X ** 2) @ (self.weight ** 2)).sum(1, keepdim=True)

        out_interaction = (out_1 - out_2) / 2
        out_linear = self.linear(X)
        return out_interaction + out_linear

class NTM(nn.Module):
    def __init__(self, vector_length, hidden_size, memory_size, output_length = 1, lstm_controller=True, activation = 'sigmoid', **kwargs):
        super(NTM, self).__init__()
        # self.controller = Controller(lstm_controller, vector_length + 1 + memory_size[1], hidden_size)
        self.controller = Controller(lstm_controller, vector_length + memory_size[1], hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        # self.fc = nn.Linear(hidden_size + memory_size[1], vector_length)
        output_layer = kwargs.get('output_layer', 'fc')
        if output_layer=='fm':
            self.fc = FM(hidden_size + memory_size[1], k = 5)
        else:    
            self.fc = nn.Linear(hidden_size + memory_size[1], output_length)
        if activation == 'linear':
            weight, bias = kwargs['weight'], kwargs['bias']
            self.activation = partial(F.linear, weight = weight, bias = bias)
        # elif activation == 'sigmoid':
        elif activation=='relu':
            self.activation = F.relu
        else:
            self.activation = F.sigmoid
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self, batch_size=1):
        self.memory.reset(batch_size)
        controller_state = self.controller.get_initial_state(batch_size)
        read = self.memory.get_initial_read(batch_size)
        read_head_state = self.read_head.get_initial_state(batch_size)
        write_head_state = self.write_head.get_initial_state(batch_size)
        return (read, read_head_state, write_head_state, controller_state)

    def forward(self, x, previous_state):
        previous_read, previous_read_head_state, previous_write_head_state, previous_controller_state = previous_state
        # previous_read = previous_read.to('cuda:0')
        controller_input = torch.cat([x, previous_read], dim=1)
        controller_output, controller_state = self.controller(controller_input, previous_controller_state)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)
        # Write
        write_head_state = self.write_head(controller_output, previous_write_head_state)
        fc_input = torch.cat((controller_output, read_head_output), dim=1)
        state = (read_head_output, read_head_state, write_head_state, controller_state)
        # return F.sigmoid(self.fc(fc_input)), state
        # print('Contrtoller output:', controller_output.mean(), controller_output.std())
        # print((controller_output<1e-6).cpu().numpy().mean())
        # print('Read Head output:', read_head_output.mean(), read_head_output.std())
        # print(read_head_output.mean(axis = 1), read_head_output.std(axis = 1))
        # print((read_head_output<1e-6).cpu().numpy().mean())
        # print(self.read_head.memory.read())
        return self.activation(self.fc(fc_input)), state
