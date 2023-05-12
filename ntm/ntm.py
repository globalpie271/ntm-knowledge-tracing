import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead
from functools import partial


class NTM(nn.Module):
    def __init__(self, vector_length, hidden_size, memory_size, output_length = 1, lstm_controller=True, activation = 'sigmoid', **kwargs):
        super(NTM, self).__init__()
        # self.controller = Controller(lstm_controller, vector_length + 1 + memory_size[1], hidden_size)
        self.controller = Controller(lstm_controller, vector_length + memory_size[1], hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        # self.fc = nn.Linear(hidden_size + memory_size[1], vector_length)
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
        return self.activation(self.fc(fc_input)), state
