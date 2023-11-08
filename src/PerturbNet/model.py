import torch
import torch.nn as nn

class PerturbNet(torch.nn.Module):
    def __init__(self, 
                 n_input: int=148,
                 n_hiddens: int=2048,
                 dropout: float=0.5):
        super().__init__()
        
        self.n_input = n_input
        self.n_hiddens = n_hiddens
        self.dropout = dropout

        self.model = nn.Sequential(nn.Linear(self.n_input, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))
        
    def forward(self, x):
        x = self.model(x)
        
        return x