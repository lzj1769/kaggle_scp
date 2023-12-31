import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from tqdm import tqdm

from model import DeepTensorFactorization
from dataset import get_dataloader
from utils import set_seed, get_cell_type_compound_gene, get_submission
import config

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def predict(model, dataloader, device):
    model.eval()

    preds = list()
    for (cell_type_indices, compound_indices, gene_indices) in tqdm(dataloader):
        cell_type_index = cell_type_indices.to(device)
        compound_indices = compound_indices.to(device)
        gene_index = gene_indices.to(device)

        pred = model(cell_type_index, compound_indices,
                     gene_index).detach().cpu().view(-1).tolist()
        preds.append(pred)

    preds = np.concatenate(preds)

    return preds


def main():
    # set random seed
    set_seed(42)

    device = torch.device("cuda")

    # Setup model
    cell_types, compounds, genes = get_cell_type_compound_gene()

    df_submission_list = []
    avg_train_loss, avg_train_mrrmse, avg_valid_loss, avg_valid_mrrmse = 0, 0, 0, 0
    for cell_type in ['nk', 't_cd4', 't_cd8', 't_reg']:
        logging.info(f"Predicting with cell type as validation: {cell_type}")

        # load data
        df_test = pd.read_csv(f"{config.RESULTS_DIR}/test.csv")
        test_loader = get_dataloader(df=df_test,
                                     cell_types=cell_types,
                                     compounds=compounds,
                                     genes=genes,
                                     batch_size=50000,
                                     num_workers=2,
                                     drop_last=False,
                                     shuffle=False,
                                     train=False)

        model = DeepTensorFactorization(cell_types=cell_types,
                                        compounds=compounds,
                                        genes=genes)

        model_path = os.path.join(config.MODEL_PATH,
                                  f'valid_cell_type_{cell_type}.pth')

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])

        train_loss = state_dict['train_loss']
        valid_loss = state_dict['valid_loss']
        train_mrrmse = state_dict['train_mrrmse']
        valid_mrrmse = state_dict['valid_mrrmse']

        model.to(device)

        df_test['predict'] = predict(
            model=model,
            dataloader=test_loader,
            device=device)

        df_submission = get_submission(df_test)

        filename = f'{cell_type}_train_loss_{train_loss:.03f}_train_mrrmse_{train_mrrmse:.03f}_valid_loss_{valid_loss:.03f}_valid_mrrmse_{valid_mrrmse:.03f}.csv'
        df_submission.to_csv(f"{config.SUBMISSION_PATH}/{filename}")

        df_submission_list.append(df_submission)

        avg_train_loss += train_loss / 4
        avg_train_mrrmse += train_mrrmse / 4
        avg_valid_loss += valid_loss / 4
        avg_valid_mrrmse += valid_mrrmse / 4

    # get average submission
    df_avg_submission = sum(
        df_submission_list) / len(df_submission_list)

    filename = f'avg_train_loss_{avg_train_loss:.03f}_train_mrrmse_{avg_train_mrrmse:.03f}_valid_loss_{avg_valid_loss:.03f}_valid_mrrmse_{avg_valid_mrrmse:.03f}.csv'
    df_avg_submission.to_csv(f"{config.SUBMISSION_PATH}/{filename}")


if __name__ == "__main__":
    main()
