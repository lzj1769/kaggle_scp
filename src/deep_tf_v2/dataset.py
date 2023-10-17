import os
import pandas as pd
from tqdm import tqdm
import numpy as np
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
        
        # get sign and -log10(p-value)
        target = self.df['target'].values[idx]
        sign = 1 if target > 0 else 0

        return (cell_type_index, compound_index, gene_index, np.abs(target), np.float32(sign))
        

def get_dataloader(df: pd.DataFrame, cell_types: list, 
                   compounds: list, genes: list, batch_size: int, num_workers: int,
                   drop_last: bool, shuffle: bool, train: bool) -> DataLoader:
    """
    Create a dataloader based on the input dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should be formated as cell_type | compound | gene
    cell_types : list
        Cell type list
    compounds : list
        Compound list
    genes : list
        Gene list
    drop_last : bool
        _description_
    shuffle : bool
        _description_

    Returns
    -------
    DataLoader
        _description_
    """

    dataset = SingleCellPerturbationDataset(df=df,
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
    
    