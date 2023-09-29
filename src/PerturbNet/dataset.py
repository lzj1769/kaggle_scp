import os
import pandas as pd
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config

class SingleCellPerturbationDataset(Dataset):
    def __init__(self, 
                 x: np.array, 
                 y: np.array=None,
                 cell_types: np.array=None,
                  compounds: np.array=None,
                  genes: np.array=None,
                 train: bool=True):
        
        self.x = x.astype(np.float32)
        
        if y is not None:
            self.y = y.astype(np.float32)
            
        self.cell_types = cell_types.astype(np.int32)
        self.compounds = compounds.astype(np.int32)
        self.genes = genes.astype(np.int32)
        self.train = train
        
        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.train:
            return self.x[idx], self.y[idx], self.cell_types[idx], self.compounds[idx], self.genes[idx]
        else:
            return self.x[idx]
        

def get_dataloader(x: np.array,
                   y: Optional[np.array]=None,
                   cell_types: np.array=None,
                   compounds: np.array=None,
                   genes: np.array=None,
                   batch_size: int=50000, 
                   num_workers: int=2,
                   drop_last: bool=False, 
                   shuffle: bool=False, 
                   train: bool=True) -> DataLoader:

    dataset = SingleCellPerturbationDataset(x=x, 
                                            y=y, 
                                            cell_types=cell_types, 
                                            compounds=compounds, 
                                            genes=genes, 
                                            train=train)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            persistent_workers=True)
    
    return dataloader
    
    