import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from model import DeepTensorFactorization
from dataset import get_dataloader
from utils import set_seed, get_cell_type_compound_gene, compute_mrrmse

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--batch_size", default=1000, type=int,
                        help="Batch size. Default 1000")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform. Default: 100")
    parser.add_argument("--lr", default=1e-03, type=float,
                        help="Learning rate. Default: 0.001")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device):
    model.train()

    train_loss = 0.0
    for (cell_type_indices, compound_indices, gene_indices, targets) in dataloader:
        preds = model(cell_type_indices.to(device),
                      compound_indices.to(device),
                      gene_indices.to(device))

        loss = criterion(preds.view(-1), targets.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

    return train_loss


def main():
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    # Setup CUDA, GPU
    device = torch.device("cuda")

    # Setup model
    cell_types, compounds, genes = get_cell_type_compound_gene()

    model = DeepTensorFactorization(cell_types=cell_types,
                                    compounds=compounds,
                                    genes=genes)
    model_path = '../../model/deep_tf_v2/model.pth'
    model.to(device)

    # Setup data
    logging.info(f'Loading data')
    df_train = pd.read_csv(f"../../results/deep_tf/train.csv")
    df_train.target = df_train.target.astype(np.float32)

    train_loader = get_dataloader(df=df_train,
                                  cell_types=cell_types,
                                  compounds=compounds,
                                  genes=genes,
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  drop_last=False,
                                  shuffle=True,
                                  train=True)

    # Setup loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                                 lr=1e-3,
                                 weight_decay=1e-4)

    """ Train the model """
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    log_prefix = f'{dt_string}'
    log_dir = os.path.join('./log', log_prefix)
    tb_writer = SummaryWriter(log_dir=log_dir)

    logging.info(f'Training started')
    for epoch in range(args.epochs):
        loss = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device)

        tb_writer.add_scalar("Loss", loss, epoch)

        state = {'state_dict': model.state_dict(),
                 'loss': loss}
        torch.save(state, model_path)

    logging.info(f'Training finished')


if __name__ == "__main__":
    main()
