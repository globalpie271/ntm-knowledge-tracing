import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, lstm_controller, vector_length, hidden_size):
        super(Controller, self).__init__()
        # We allow either a feed-forward network or a LSTM for the controller
        self._lstm_controller = lstm_controller
        if self._lstm_controller:
            self._controller = LSTMController(vector_length, hidden_size)
        else:
            # self._controller = FeedForwardController(vector_length, hidden_size)
            self._controller = TransformerEncoderController(vector_length, hidden_size)
            # self._controller = TransformerEncoderController(vector_length, hidden_size)

    def forward(self, x, state):
        return self._controller(x, state)

    def get_initial_state(self, batch_size):
        return self._controller.get_initial_state(batch_size)


class LSTMController(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(LSTMController, self).__init__()
        self.layer = nn.LSTM(input_size=vector_length, hidden_size=hidden_size)
        # The hidden state is a learned parameter
        self.lstm_h_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        self.lstm_c_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        for p in self.layer.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(vector_length + hidden_size))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, x, state):
        output, state = self.layer(x.unsqueeze(0), state)
        return output.squeeze(0), state

    def get_initial_state(self, batch_size):
        lstm_h = self.lstm_h_state.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_state.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c


class FeedForwardController(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(FeedForwardController, self).__init__()
        self.layer_1 = nn.Linear(vector_length, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        stdev = 5 / (np.sqrt(vector_length + hidden_size))
        nn.init.uniform_(self.layer_1.weight, -stdev, stdev)
        nn.init.uniform_(self.layer_2.weight, -stdev, stdev)

    def forward(self, x, state):
        x1 = F.relu(self.layer_1(x))
        output = F.relu(self.layer_2(x1))
        return output, state

    def get_initial_state(self, batch_size = None):
        return 0, 0

class MultiheadAttentionController(nn.Module):
    """
    Multhead attention controller
    """
    def __init__(self, vector_length, hidden_size, nhead = 1, dropout = 0):
        super(TransformerEncoderController, self).__init__()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model = vector_length, nhead = nhead)
        self.attention = nn.MultiheadAttention(vector_length, nhead, dropout)
        # self.linear = nn.Linear(vector_length, hidden_size)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x, state):
        output, attn_output_weights = self.attention(x)
        return output, state
    def get_initial_state(self, batch_size = None):
        return 0, 0


class TransformerEncoderController(nn.Module):
    """
    Transformer encoder controller
    """
    def __init__(self, vector_length, hidden_size, nhead = 1, num_layers = 1):
        super(TransformerEncoderController, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = vector_length, nhead = nhead)
        self.linear = nn.Linear(vector_length, hidden_size)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x, state):
        x1 = self.encoder_layer(x)
        x2 = self.linear(x1)
        # output = F.softmax(x2)
        output = F.relu(x2)
        return output, state
    def get_initial_state(self, batch_size = None):
        return 0, 0

class FactorizationMachineController(nn.Module):
    """
    Factorization machine controller
    """
    def __init__(self, vector_length, hidden_size, nhead = 1, num_layers = 1, k = 5):
        super(FactorizationMachineController, self).__init__()
        self.v = nn.Parameter(torch.randn(vector_length, k), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.V.data)
        self.linear = nn.Linear(vector_length, 1)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x, state):
        out_1 = ((x @ self.v) ** 2).sum(1, keepdim=True)
        out_2 = ((x ** 2) @ (self.v ** 2)).sum(1, keepdim=True)

        out_interaction = (out_1 - out_2) / 2
        out_linear = self.linear(x)
        return out_interaction + out_linear, state
    def get_initial_state(self, batch_size = None):
        return 0, 0


class FM(nn.Module):
    def __init__(self, features_num=None, k=2):
        super().__init__()
        self.V = nn.Parameter(torch.randn(features_num, k), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.V.data)
        self.linear = nn.Linear(features_num, 1)

    def forward(self, X):
        out_1 = ((X @ self.V) ** 2).sum(1, keepdim=True)
        out_2 = ((X ** 2) @ (self.V ** 2)).sum(1, keepdim=True)

        out_interaction = (out_1 - out_2) / 2
        out_linear = self.linear(X)
        return out_interaction + out_linear