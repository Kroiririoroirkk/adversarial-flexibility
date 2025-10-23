import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

from brain import Brain
from detective import Detective


class Controller():
    """
    Runs joint training of Brain and Detective.
    """
    
    def __init__(
            self,
            brain_batch_size=64,
            detective_batch_size=64,
            lr_brain=1e-4,
            lr_detective=1e-4,
            clue = 'Mean firing rate',
            brain_kwargs=None,
            detective_kwargs=None,
            device=None,
            seed=None
        ):

        self.brain_batch_size = brain_batch_size
        self.detective_batch_size = detective_batch_size
        self.lr_brain = lr_brain
        self.lr_detective = lr_detective
        self.clue = clue

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        brain_kwargs = brain_kwargs or {}
        detective_kwargs = detective_kwargs or {}

        # Initialize Brain and Detective
        self.brain = Brain(**brain_kwargs).to(self.device)
        self.detective = Detective(**detective_kwargs).to(self.device)
  
        # Optimizers
        self.opt_brain = optim.Adam(self.brain.parameters(), lr=self.lr_brain)
        self.opt_detective = optim.Adam(self.detective.parameters(), lr=self.lr_detective)

        # Loss function
        self.criterion = nn.BCEwithLogitsLoss()

    @torch.no_grad()
    def sample_inputs(self, batch_size):
        return torch.rand(batch_size, 1, device=self.device)
    

    def brain_forward(self, x):
        outputs = self.brain.brain_response(x)
        clues = self.brain.get_clues(self.clue)
        return outputs, clues
    

    def detective_forward(self, clues):
        preds = self.detective(clues)
        return preds
    

    def brain_step(self):
        x = self.sample_inputs(self.brain_batch_size)

        # Forward pass
        output, clues = self.brain_forward(x)
        preds = self.detective_forward(clues)

        # compute loss
        loss_brain = - self.criterion(preds, output) # negative because we want to maximize the BCE

        # Backward pass
        self.opt_brain.zero_grad(set_to_none=True)
        loss_brain.backward()
        self.opt_brain.step()

        return loss_brain.item()
    
    def detective_step(self):
        x = self.sample_inputs(self.detective_batch_size)

        with torch.no_grad():
            output, clues = self.brain_forward(x)

        # Forward pass
        preds = self.detective_forward(clues)

        # compute loss
        loss_detective = self.criterion(preds, output)

        # Backward pass
        self.opt_detective.zero_grad(set_to_none=True)
        loss_detective.backward()
        self.opt_detective.step()

        return loss_detective.item()
    
    def train(self, steps=1000, detective_updates_per_step=1, brain_updates_per_step=10, log_every=50, verbose=True):
        logs = []

        # Training loop
        for step in range(1, steps + 1):

            # loop over detective steps (brain is not updated here)
            ld_sum = 0.0
            for _ in range(detective_updates_per_step):
                ld_sum += self.detective_step()
            ld = ld_sum / detective_updates_per_step

            # loop over brain steps (detective is not updated here)
            lb_sum = 0.0
            for _ in range(brain_updates_per_step):
                lb_sum += self.brain_step()
            lb = lb_sum / brain_updates_per_step

            logs.append((step, ld, lb))
            if verbose and step % log_every == 0:
                print(f"Step {step}: LD = {ld:.4f}, LB = {lb:.4f}")
        return logs
    

    
    def plot_loss(self, logs):

        steps = [log[0] for log in logs]
        ld_values = [log[1] for log in logs]
        lb_values = [log[2] for log in logs]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sns.lineplot(x=steps, y=ld_values, ax=axs[0], color='blue')
        axs[0].set_title('Detective Loss over Training Steps')
        axs[0].set_ylabel('Detective Loss (LD)')

        sns.lineplot(x=steps, y=lb_values, ax=axs[1], color='orange')
        axs[1].set_title('Brain Loss over Training Steps')
        axs[1].set_ylabel('Brain Loss (LB)')
        axs[1].set_xlabel('Training Steps')

        plt.tight_layout()
        plt.show()



