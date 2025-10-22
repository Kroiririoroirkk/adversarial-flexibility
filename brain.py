from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PONDERING_LENGTH = 100
NUM_NEURONS = 100
INPUT_SIZE = 1
OUTPUT_SIZE = 1
LEARNING_RATE = 1e-3


class Clue(Enum):
    FIRING_RATE = 'Mean firing rate'
    PCS = 'First five PCs'
    NEURONS = '50% of neurons'
    TIME = 'First 80% of trial time points'


class RNNLayer(nn.Module):

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

        # Synaptic weights (fixed)
        # Random values for input and recurrent weights
        self.W_input = nn.Parameter(torch.randn(num_neurons, INPUT_SIZE),
                                    requires_grad=False)
        self.W_rec = nn.Parameter(torch.randn(num_neurons, num_neurons),
                                  requires_grad=False)
        # Zero bias for now
        self.b = nn.Parameter(torch.zeros(num_neurons), requires_grad=False)

        # Sigmoid parameters (trainable)
        self.sigmoid_steepness = nn.Parameter(torch.ones(num_neurons))
        self.sigmoid_offset = nn.Parameter(torch.zeros(num_neurons))
        self.sigmoid_amplitude = nn.Parameter(torch.ones(num_neurons))

    def forward(self, x, r_prev):
        lin_combo = F.linear(x, self.W_input) + F.linear(r_prev,
                                                         self.W_rec) + self.b
        r = self.sigmoid_amplitude * torch.sigmoid(
            self.sigmoid_steepness * (lin_combo - self.sigmoid_offset))
        return r


class Brain(nn.Module):

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.cell = RNNLayer(num_neurons)
        self.out = nn.Linear(num_neurons, OUTPUT_SIZE)
        for param in self.out.parameters():
            param.requires_grad = False
        self.last_run = None

    def forward(self, x):
        # TODO: x.size = (batch_size, seq_len, input_size)?
        batch_size, seq_len, _ = x.size()
        rs = np.zeros((PONDERING_LENGTH, batch_size, self.num_neurons))
        r = torch.zeros(batch_size, self.num_neurons, device=x.device)
        for t in range(PONDERING_LENGTH):
            r = self.cell(x[:, t, :], r)
            rs[t] = r.numpy(force=True)
        self.last_run = rs
        return self.out(r)

    def brain_response(self, x):
        return self.forward(x)

    def get_clues(self, clue):
        if clue == Clue.FIRING_RATE:
            return np.mean(self.last_run, axis=0)
        elif clue == Clue.PCS:
            pass  # TODO
        elif clue == Clue.NEURONS:
            return self.last_run[:, :, :self.num_neurons // 2]
        elif clue == Clue.TIME:
            return self.last_run[:int(0.8 * PONDERING_LENGTH), :, :]
