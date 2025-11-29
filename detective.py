import torch.nn as nn


class Detective(nn.Module):

    def __init__(self, clue_length):
        super().__init__()
        self.clue_length = clue_length
        # Length of the clue type this detective will train with

        # Basic MLP (arbitrary layer sizes)
        self.mlp = nn.Sequential(nn.Linear(clue_length, 500), nn.ReLU(),
                                 nn.Linear(500, 500), nn.ReLU(),
                                 nn.Linear(500, 1))

    def forward(self, clue):
        return self.mlp(clue)
