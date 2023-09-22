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


def compute_mean_row_wise_root_mse(df):
    df['diff'] = (df['target'] - df['predict']) ** 2
    df = df.drop(['gene', 'target', 'predict'], axis=1)
    grouped = df.groupby(['cell_type', 'sm_name'])
    df = grouped.mean()
    mrrmse = np.mean(np.sqrt(df['diff']))
    
    return mrrmse

def get_submission(df_pred):
    df_pred.sort_values(['cell_type', 'sm_name'], inplace=True)
    df_pred = df_pred.pivot(index=['cell_type', 'sm_name'], columns='gene', values='predict')
    
    df_pred.reset_index(inplace=True)
    
    df_pred.drop(['cell_type', 'sm_name'], axis=1, inplace=True)
    df_pred.index.names = ['id']
    
    return df_pred