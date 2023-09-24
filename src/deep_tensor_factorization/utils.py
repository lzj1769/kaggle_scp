import os
import random
import numpy as np
import torch
import pandas as pd

import config

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def get_cell_type_compound_gene():
    """
    Get all cell types, compounds, and genes.
    """
    df = pd.read_parquet(config.DE_TRAIN)
    
    cell_types = df['cell_type'].unique().tolist()
    compounds = df['sm_name'].unique().tolist()
    genes = df.columns.values.tolist()[5:]
    
    return cell_types, compounds, genes


def compute_mrrmse(df: pd.DataFrame) -> np.float32:
    """
    Compute Mean Rowwise Root Mean Squared Error

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Should be formated as cell_type|sm_name|genetarget|predict

    Returns
    -------
    np.float32
        mmrmse
    """
    
    df['diff'] = (df['target'] - df['predict']) ** 2
    df = df.drop(['gene', 'target', 'predict'], axis=1)
    grouped = df.groupby(['cell_type', 'sm_name'])
    df = grouped.mean()
    mrrmse = np.mean(np.sqrt(df['diff']))
    
    return mrrmse

def get_submission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe from long to wide format for submission

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe formated as cell_type|sm_name|predict

    Returns
    -------
    pd.DataFrame
        Output dataframe for submission
    """
    df.sort_values(['cell_type', 'sm_name'], inplace=True)
    df = df.pivot(index=['cell_type', 'sm_name'], columns='gene', values='predict')
    
    df.reset_index(inplace=True)
    
    df.drop(['cell_type', 'sm_name'], axis=1, inplace=True)
    df.index.names = ['id']
    
    return df