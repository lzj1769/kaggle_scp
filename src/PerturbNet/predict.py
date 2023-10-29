import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from model import PerturbNet
from dataset import get_dataloader
from utils import set_seed, get_submission
import config

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--deep_tf', 
                        choices=['v1', 'v2', 'v3', 'v4'],
                        type=str, default='v1')
    parser.add_argument("--use_ChemBERTa", action="store_true", default=False,
                        help="If use features from ChemBERTa")
    parser.add_argument("--scale_feature", action="store_true", default=True,
                        help="If standardize the input features. Default: True")
    parser.add_argument("--use_cell_type_umap", action="store_true",
                        help="If use features from UMAP for cell types")
    parser.add_argument("--use_chemical_fingerprint", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    return parser.parse_args()


def predict(model, dataloader, device):
    model.eval()

    preds = list()
    for x in tqdm(dataloader):
        pred = model(x.to(device)).detach().cpu().view(-1).tolist()
        preds.append(pred)

    preds = np.concatenate(preds)

    return preds


def main():
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    device = torch.device("cuda")

    df_submission_list = []
    avg_train_loss, avg_train_mrrmse, avg_valid_loss, avg_valid_mrrmse = 0, 0, 0, 0
    for cell_type in ['nk', 't_cd4', 't_cd8', 't_reg']:
        logging.info(f"Predicting with cell type as validation: {cell_type}")

        train_deep_tf = np.load(
                f"{config.RESULTS_DIR}/deep_tf_{args.deep_tf}/train_{cell_type}.npz")
        valid_deep_tf = np.load(
                f"{config.RESULTS_DIR}/deep_tf_{args.deep_tf}/valid_{cell_type}.npz")
        test_deep_tf = np.load(
                f"{config.RESULTS_DIR}/deep_tf_{args.deep_tf}/test.npz")

        # load data
        train_x, valid_x, test_x = train_deep_tf['x'], valid_deep_tf['x'], test_deep_tf['x']

        # concatentate molecular features from ChemBERTa
        if args.use_ChemBERTa:
            logging.info(f'Loading molecular features from ChemBERTa')
            train_ChemBERTa = np.load(
                f"{config.RESULTS_DIR}/ChemBERTa/train_{cell_type}.npz")
            valid_ChemBERTa = np.load(
                f"{config.RESULTS_DIR}/ChemBERTa/valid_{cell_type}.npz")
            test_ChemBERTa = np.load(
                f"{config.RESULTS_DIR}/ChemBERTa/test.npz")

            train_x = np.concatenate([train_x, train_ChemBERTa['x']], axis=1)
            valid_x = np.concatenate([valid_x, valid_ChemBERTa['x']], axis=1)
            test_x = np.concatenate([test_x, test_ChemBERTa['x']], axis=1)
        
        if args.use_cell_type_umap:
            logging.info(f'Loading cell type UMAP features')
            train_cell_type = np.load(
                f"{config.RESULTS_DIR}/cell_type_umap/train_{cell_type}.npz")
            valid_cell_type = np.load(
                f"{config.RESULTS_DIR}/cell_type_umap/valid_{cell_type}.npz")
            test_cell_type = np.load(
                f"{config.RESULTS_DIR}/cell_type_umap/test.npz")

            train_x = np.concatenate([train_x, train_cell_type['x']], axis=1)
            valid_x = np.concatenate([valid_x, valid_cell_type['x']], axis=1)
            test_x = np.concatenate([test_x, test_cell_type['x']], axis=1)
            
        if args.use_chemical_fingerprint:
            logging.info(f'Loading chemical fingerprint')
            train_cell_type = np.load(
                f"{config.RESULTS_DIR}/chemical_fingerprint/train_{cell_type}.npz")
            valid_cell_type = np.load(
                f"{config.RESULTS_DIR}/chemical_fingerprint/valid_{cell_type}.npz")
            test_cell_type = np.load(
                f"{config.RESULTS_DIR}/chemical_fingerprint/test.npz")

            train_x = np.concatenate([train_x, train_cell_type['x']], axis=1)
            valid_x = np.concatenate([valid_x, valid_cell_type['x']], axis=1)
            test_x = np.concatenate([test_x, test_cell_type['x']], axis=1)

        logging.info('Standarizing the features')
        scaler = StandardScaler()
        scaler.fit(X=np.concatenate([train_x, valid_x, test_x], axis=0))
        test_x = scaler.transform(test_x)

        logging.info(
            f'Number of test samples: {test_x.shape[0]}; number of features: {test_x.shape[1]}')

        test_loader = get_dataloader(x=test_x,
                                     num_workers=2,
                                     drop_last=False,
                                     shuffle=False,
                                     train=False)

        # Setup model
        model = PerturbNet(n_input=test_x.shape[1])
        model_path = os.path.join(config.MODEL_PATH,
                                  f'{cell_type}.pth')

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])

        train_loss = state_dict['train_loss']
        valid_loss = state_dict['valid_loss']
        train_mrrmse = state_dict['train_mrrmse']
        valid_mrrmse = state_dict['valid_mrrmse']

        model.to(device)

        # predict target
        df_test = pd.read_csv(f"{config.RESULTS_DIR}/deep_tf/test.csv")
        model.eval()
        preds = list()
        for x in tqdm(test_loader):
            pred = model(x.to(device)).detach().cpu().view(-1).tolist()
            preds.append(pred)

        df_test['predict'] = np.concatenate(preds)
        df_submission = get_submission(df_test)

        filename = f'{cell_type}_valid_mrrmse_{valid_mrrmse:.03f}.csv'
        df_submission.to_csv(f"{config.SUBMISSION_PATH}/{filename}")

        df_submission_list.append(df_submission)

        avg_train_loss += train_loss / 4
        avg_train_mrrmse += train_mrrmse / 4
        avg_valid_loss += valid_loss / 4
        avg_valid_mrrmse += valid_mrrmse / 4

    # get average submission
    df_avg_submission = sum(
        df_submission_list) / len(df_submission_list)

    filename = f'avg_valid_mrrmse_{avg_valid_mrrmse:.03f}.csv'
    df_avg_submission.to_csv(f"{config.SUBMISSION_PATH}/{filename}")


if __name__ == "__main__":
    main()
