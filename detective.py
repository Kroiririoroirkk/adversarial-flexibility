import numpy as np
import torch
import torch.nn as nn

OUTPUT_SIZE = 1

class Detective(nn.Module):
    
    def __init__(self, clue_length, hidden_size, layers):
        super().__init__
        self.clue_length = clue_length #length of the clue type this detective will train with
        # At some stage, the clue needs to be transformed into appropriate input tensor

        # Basic RNN : 
        self.rnn = nn.RNN(clue_length, hidden_size, layers)
        self.fc = nn.Linear(hidden_size, OUTPUT_SIZE)

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, clue):
        out, hidden = self.rnn(clue, hidden)
        guess = self.fc(out) #actual guess at Brain choice

        return guess, hidden

