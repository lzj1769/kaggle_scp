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