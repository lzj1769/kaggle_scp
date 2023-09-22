import torch
import torch.nn as nn

class DeepTensorFactorization(torch.nn.Module):
    def __init__(self, cell_types, compounds, genes, 
                 n_cell_type_factors=4, 
                 n_compounds_factors=16, 
                 n_gene_factors=32):
        super().__init__()
        
        self.cell_types = cell_types
        self.compounds = compounds
        self.genes = genes
        
        self.n_cell_types = len(cell_types)
        self.n_compounds = len(compounds)
        self.n_genes = len(genes)
        
        self.n_cell_type_factors = n_cell_type_factors
        self.n_compounds_factors = n_compounds_factors
        self.n_gene_factors = n_gene_factors
        
        self.cell_type_embedding = torch.nn.Embedding(self.n_cell_types, self.n_cell_type_factors)
        self.sm_embedding = torch.nn.Embedding(self.n_compounds, self.n_compounds_factors)
        self.gene_embedding = torch.nn.Embedding(self.n_genes, self.n_gene_factors)
        
        self.n_factors = n_cell_type_factors + n_compounds_factors + n_gene_factors
        
        self.model = nn.Sequential(nn.Linear(self.n_factors, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Linear(128, 1))
        
    def forward(self, cell_type_indices, sm_indices, gene_indices):
        cell_type_vec = self.cell_type_embedding(cell_type_indices)
        sm_vec = self.sm_embedding(sm_indices)
        gene_vec = self.gene_embedding(gene_indices)
        
        x = torch.concat([cell_type_vec, sm_vec, gene_vec], dim=1)
        x = self.model(x)
        
        return x
    
    
    def get_cell_type_embedding():
        return NotImplementedError
    
    def get_compound_embedding():
        return NotImplementedError
    
    def get_gene_embedding():
        return NotImplementedError
        