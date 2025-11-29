from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Clue(Enum):
    ANSWER = 'Answer'
    LAST_ACTIVITY = 'Last activity'
    RANDOM = 'Random'
    FIRING_RATE = 'Mean firing rate'
    PCS = 'First five PCs'
    NEURONS = '50% of neurons'
    TIME = 'First 80% of trial time points'


class RNNLayer(nn.Module):

    def __init__(self, num_neurons, train_synapses, train_intrinsic):
        super().__init__()
        self.num_neurons = num_neurons

        # Synaptic weights: random values for input and recurrent weights,
        # bias initialized to zero
        self.W_input = nn.Parameter(torch.randn(num_neurons, 1),
                                    requires_grad=False)
        self.W_rec = nn.Parameter(torch.randn(num_neurons, num_neurons),
                                  requires_grad=False)
        self.b = nn.Parameter(torch.zeros(num_neurons), requires_grad=False)

        # Intrinsic parameters: determine shape of sigmoid
        self.sigmoid_steepness = nn.Parameter(torch.ones(num_neurons),
                                              requires_grad=False)
        self.sigmoid_offset = nn.Parameter(torch.zeros(num_neurons),
                                           requires_grad=False)
        self.sigmoid_amplitude = nn.Parameter(torch.ones(num_neurons),
                                              requires_grad=False)

        if train_synapses:
            self.W_input.requires_grad = True
            self.W_rec.requires_grad = True
            self.b.requires_grad = True
        if train_intrinsic:
            self.sigmoid_steepness.requires_grad = True
            self.sigmoid_offset.requires_grad = True
            self.sigmoid_amplitude.requires_grad = True

    def forward(self, x, r_prev):
        lin_combo = F.linear(x, self.W_input) + F.linear(r_prev,
                                                         self.W_rec) + self.b
        r = self.sigmoid_amplitude * torch.sigmoid(
            self.sigmoid_steepness * (lin_combo - self.sigmoid_offset))
        return r


class Brain(nn.Module):

    def __init__(self, num_neurons, pondering_length, train_synapses,
                 train_intrinsic):
        super().__init__()
        self.num_neurons = num_neurons
        self.pondering_length = pondering_length
        self.cell = RNNLayer(num_neurons, train_synapses, train_intrinsic)
        self.out = nn.Linear(num_neurons, 1)
        for param in self.out.parameters():
            param.requires_grad = False
        self.last_run = None

    def forward(self, x):
        batch_size, _ = x.size()
        rs = np.zeros((self.pondering_length, batch_size, self.num_neurons))
        r = torch.zeros(batch_size, self.num_neurons, device=x.device)
        for t in range(self.pondering_length):
            r = self.cell(x, r)
            rs[t] = r.numpy(force=True)
        self.last_run = np.transpose(rs, axes=[1, 2, 0])
        return torch.sigmoid(self.out(r))

    def brain_response(self, x):
        return self.forward(x)

    def get_clues(self, clue):
        # Returns clue as a tensor of size (batch_size, clue_length)
        # self.last_run is of size (batch_size, num_neurons, pondering_length)
        batch_size = self.last_run.shape[0]
        if clue == Clue.ANSWER:
            return torch.sigmoid(
                self.out(torch.from_numpy(self.last_run[:, :, -1]).float()))
        elif clue == Clue.LAST_ACTIVITY:
            clue_arr = self.last_run[:, :, -1]
        elif clue == Clue.RANDOM:
            return torch.rand(batch_size, 1)
        elif clue == Clue.FIRING_RATE:
            clue_arr = np.mean(self.last_run, axis=1)
        elif clue == Clue.PCS:
            pass  # TODO
        elif clue == Clue.NEURONS:
            clue_arr = self.last_run[:, :self.num_neurons // 2, :]
        elif clue == Clue.TIME:
            clue_arr = self.last_run[:, :, :int(0.8 * self.pondering_length)]
        return torch.from_numpy(clue_arr.reshape(batch_size, -1)).float()

    def get_clue_length(self, clue_type):
        d = {
            Clue.ANSWER: 1,
            Clue.LAST_ACTIVITY: self.num_neurons,
            Clue.RANDOM: 1,
            Clue.FIRING_RATE: self.pondering_length,
            Clue.PCS: self.pondering_length * 5,
            Clue.NEURONS: self.pondering_length * (self.num_neurons // 2),
            Clue.TIME: int(0.8 * self.pondering_length) * self.num_neurons
        }
        return d[clue_type]
