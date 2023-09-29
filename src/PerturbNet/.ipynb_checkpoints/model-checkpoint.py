from typing import List

import torch
import torch.nn as nn

class DeepTensorFactorization(torch.nn.Module):
    def __init__(self, 
                 n_input: int=128,
                 n_hiddens: int=2048,
                 dropout: float=0.5):
        super().__init__()
        
        self.n_hiddens = n_hiddens
        self.dropout = dropout

        
        self.model = nn.Sequential(nn.Linear(self.n_factors, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))
        
    def forward(self, cell_type_indices, compound_indices, gene_indices):
        cell_type_vec = self.cell_type_embedding(cell_type_indices)
        compound_vec = self.compound_embedding(compound_indices)
        gene_vec = self.gene_embedding(gene_indices)
        
        x = torch.concat([cell_type_vec, compound_vec, gene_vec], dim=1)
        x = self.model(x)
        
        return x
    
    @property
    def get_cell_type_embedding(self):
        return self.cell_type_embedding.weight.detach().numpy()
    
    @property
    def get_compound_embedding(self):
        return self.compound_embedding.weight.detach().numpy()
    
    @property
    def get_gene_embedding(self):
        return self.gene_embedding.weight.detach().numpy()