import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SingleCellPerturbationDataset(Dataset):
    def __init__(self, df, cell_types, compounds, genes, train):
        self.df = df
        
        self.cell_types = cell_types
        self.compounds = compounds
        self.genes = genes
        self.train = train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cell_type = self.df['cell_type'].values[idx]
        sm_name = self.df['sm_name'].values[idx]
        gene = self.df['gene'].values[idx]
        
        cell_type_index = self.cell_types.index(cell_type)
        compound_index = self.compounds.index(sm_name)
        gene_index = self.genes.index(gene)
        
        if self.train:
            target = self.df['p_value'].values[idx]
            return (cell_type_index, compound_index, gene_index, target)
        else:
            return (cell_type_index, compound_index, gene_index)
        
        
def get_dataloader(data, fold, batch_size, num_workers):
    df_train = pd.read_csv()