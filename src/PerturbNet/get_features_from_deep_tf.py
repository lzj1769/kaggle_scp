from utils import get_cell_type_compound_gene
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_data(model_path, df, cell_types, compounds, genes, get_target=True):
    state_dict = torch.load(model_path)

    # get embedding for cell type, compound, and gene
    cell_type_embedding = state_dict['state_dict']['cell_type_embedding.weight'].cpu(
    )
    compound_embedding = state_dict['state_dict']['compound_embedding.weight'].cpu(
    )
    gene_embedding = state_dict['state_dict']['gene_embedding.weight'].cpu()

    n_samples = len(df)
    n_features = cell_type_embedding.shape[1] + \
        compound_embedding.shape[1] + gene_embedding.shape[1]

    # get features
    x = np.zeros((n_samples, n_features))
    cell_type_idxs = np.zeros(n_samples)
    compound_idxs = np.zeros(n_samples)
    gene_idxs = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        cell_type_idx = cell_types.index(df['cell_type'][i])
        compound_idx = compounds.index(df['sm_name'][i])
        gene_idx = genes.index(df['gene'][i])

        cell_type_vec = cell_type_embedding[cell_type_idx]
        compound_vec = compound_embedding[compound_idx]
        gene_vec = gene_embedding[gene_idx]

        x[i] = torch.concat([cell_type_vec, compound_vec, gene_vec])
        cell_type_idxs[i] = cell_type_idx
        compound_idxs[i] = compound_idx
        gene_idxs[i] = gene_idx

    if get_target:
        y = np.zeros(n_samples)
        for i in range(n_samples):
            y[i] = df['target'][i]

        return x, y, cell_type_idxs, compound_idxs, gene_idxs
    else:
        return x, cell_type_idxs, compound_idxs, gene_idxs


def main():
    cell_types, compounds, genes = get_cell_type_compound_gene()

    for cell_type in ['nk', 't_cd4', 't_cd8', 't_reg']:
        logging.info(f'Extracting features using {cell_type} as validation')

        model_path = f'../../model/deep_tensor_factorization/valid_cell_type_{cell_type}.pth'

        # load training data
        logging.info(f'Loading training data')
        df_train = pd.read_csv(f'../../results/deep_tensor_factorization/df_train_{cell_type}.csv',
                               index_col=0)

        # remove control compound from training data
        df_train = df_train[~df_train['sm_name'].isin(
            ['Dabrafenib', 'Belinostat'])]
        df_train = df_train.reset_index(drop=True)

        x, y, cell_type_idxs, compound_idxs, gene_idxs = get_data(model_path=model_path,
                                                                  df=df_train,
                                                                  cell_types=cell_types,
                                                                  compounds=compounds,
                                                                  genes=genes)
        np.savez(f'../../results/PerturbNet/deep_tf/train_{cell_type}.npz',
                 x=x,
                 y=y,
                 cell_types=cell_type_idxs,
                 compounds=compound_idxs,
                 genes=gene_idxs)

        # load validation data
        logging.info(f'Loading validation data')
        df_valid = pd.read_csv(f'../../results/deep_tensor_factorization/df_valid_{cell_type}.csv',
                               index_col=0)
        
        # remove control compound from validation data
        df_valid = df_valid[~df_valid['sm_name'].isin(
            ['Dabrafenib', 'Belinostat'])]
        df_valid = df_valid.reset_index(drop=True)

        x, y, cell_type_idxs, compound_idxs, gene_idxs = get_data(model_path=model_path,
                                                                  df=df_valid,
                                                                  cell_types=cell_types,
                                                                  compounds=compounds,
                                                                  genes=genes)
        np.savez(f'../../results/PerturbNet/deep_tf/valid_{cell_type}.npz',
                 x=x, y=y,
                 cell_types=cell_type_idxs,
                 compounds=compound_idxs,
                 genes=gene_idxs)

        # test data
        # there are no control compounds in test data
        logging.info(f'Loading test data')
        df_test = f'../../results/deep_tensor_factorization/test.csv'
        x, cell_type_idxs, compound_idxs, gene_idxs = get_data(model_path=model_path,
                                                               df=df_test,
                                                               cell_types=cell_types,
                                                               compounds=compounds,
                                                               genes=genes,
                                                               get_target=False)

        np.savez(f'../../results/PerturbNet/deep_tf/test_{cell_type}.npz',
                 x=x, cell_types=cell_type_idxs, compounds=compound_idxs, genes=gene_idxs)


if __name__ == "__main__":
    main()
