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
    parser.add_argument("--use_ChemBERTa", action="store_true", default=False,
                        help="If use features from ChemBERTa")
    parser.add_argument("--scale_feature", action="store_true", default=False,
                        help="If standardize the input features. Default: True")
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

        # load test data
        test_deep_tf = np.load(f"{config.RESULTS_DIR}/deep_tf/test.npz")
        test_x = test_deep_tf['x']
        
        train_deep_tf = np.load(
                f"{config.RESULTS_DIR}/deep_tf/train_{cell_type}.npz")
        valid_deep_tf = np.load(
                f"{config.RESULTS_DIR}/deep_tf/valid_{cell_type}.npz")
        
        train_x, valid_x = train_deep_tf['x'], valid_deep_tf['x']

        if args.scale_feature:
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
