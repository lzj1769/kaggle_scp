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

from sklearn.preprocessing import StandardScaler

from model import PerturbNet
from dataset import get_dataloader
from utils import set_seed, compute_mrrmse
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
    parser.add_argument("--valid_cell_type", type=str, default='nk',
                        help="Which cell type used for validation. Available options are: nk,  t_cd4, t_cd8, t_reg")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--resume",
                        action="store_true",
                        help='training model from check point')
    parser.add_argument("--batch_size", default=10000, type=int,
                        help="Batch size. Default 5000")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform. Default: 100")
    parser.add_argument("--lr", default=3e-04, type=float,
                        help="Learning rate. Default: 0.001")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--use_ChemBERTa", action="store_true",
                        help="If use features from ChemBERTa")

    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device):
    model.train()

    train_loss = 0.0
    cell_types, compunds, genes, targets, preds = list(), list(), list(), list(), list()
    for (x, y, cell_type_indices, compound_indices, gene_indices) in dataloader:
        pred = model(x.to(device))
        loss = criterion(pred.view(-1), y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

        # collect data to compute mrrmse
        cell_types.append(cell_type_indices.view(-1).tolist())
        compunds.append(compound_indices.view(-1).tolist())
        genes.append(gene_indices.view(-1).tolist())
        targets.append(y.view(-1).tolist())
        preds.append(pred.detach().cpu().view(-1).tolist())

    df = pd.DataFrame(data={'cell_type': np.concatenate(cell_types),
                            'sm_name': np.concatenate(compunds),
                            'gene': np.concatenate(genes),
                            'target': np.concatenate(targets),
                            'predict': np.concatenate(preds)})

    mrrmse = compute_mrrmse(df)

    return train_loss, mrrmse


def valid(model, dataloader, criterion, device):
    model.eval()

    valid_loss = 0.0
    cell_types, compunds, genes, targets, preds = list(), list(), list(), list(), list()
    for (x, y, cell_type_indices, compound_indices, gene_indices) in dataloader:
        pred = model(x.to(device))

        loss = criterion(pred.view(-1), y.to(device))

        valid_loss += loss.item() / len(dataloader)

        # collect data to compute mrrmse
        cell_types.append(cell_type_indices.view(-1).tolist())
        compunds.append(compound_indices.view(-1).tolist())
        genes.append(gene_indices.view(-1).tolist())
        targets.append(y.view(-1).tolist())
        preds.append(pred.detach().cpu().view(-1).tolist())

    df = pd.DataFrame(data={'cell_type': np.concatenate(cell_types),
                            'sm_name': np.concatenate(compunds),
                            'gene': np.concatenate(genes),
                            'target': np.concatenate(targets),
                            'predict': np.concatenate(preds)})

    mrrmse = compute_mrrmse(df)

    return valid_loss, mrrmse


def main():
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    # Setup CUDA, GPU
    device = torch.device("cuda")
    logging.info(f'Validation cell type: {args.valid_cell_type}')

    # Setup data
    logging.info(f'Loading data')

    train_deep_tf = np.load(
        f"{config.RESULTS_DIR}/deep_tf/train_{args.valid_cell_type}.npz")
    valid_deep_tf = np.load(
        f"{config.RESULTS_DIR}/deep_tf/valid_{args.valid_cell_type}.npz")

    train_x, train_y = train_deep_tf['x'], train_deep_tf['y']
    valid_x, valid_y = valid_deep_tf['x'], valid_deep_tf['y']

    # concatentate molecular features from ChemBERTa
    if args.use_ChemBERTa:
        logging.info(f'Loading molecular features from ChemBERTa')
        train_ChemBERTa = np.load(
            f"{config.RESULTS_DIR}/ChemBERTa/train_{args.valid_cell_type}.npz")
        valid_ChemBERTa = np.load(
            f"{config.RESULTS_DIR}/ChemBERTa/valid_{args.valid_cell_type}.npz")

        train_x = np.concatenate([train_x, train_ChemBERTa['x']], axis=1)
        valid_x = np.concatenate([valid_x, valid_ChemBERTa['x']], axis=1)

    logging.info(
        f'Number of training samples: {train_x.shape[0]}; number of features: {train_x.shape[1]}')
    logging.info(
        f'Number of validation samples: {valid_x.shape[0]}; number of features: {valid_x.shape[1]}')

    # feature standarization
    # logging.info('Standarizing the features')
    # scaler = StandardScaler()
    # scaler.fit(X=np.concatenate([train_x, valid_x], axis=0))
    # train_x = scaler.transform(train_x)
    # valid_x = scaler.transform(valid_x)
    
    train_loader = get_dataloader(x=train_x,
                                  y=train_y,
                                  cell_types=train_deep_tf['cell_types'],
                                  compounds=train_deep_tf['compounds'],
                                  genes=train_deep_tf['genes'],
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  drop_last=False,
                                  shuffle=True,
                                  train=True)

    valid_loader = get_dataloader(x=valid_x,
                                  y=valid_y,
                                  cell_types=valid_deep_tf['cell_types'],
                                  compounds=valid_deep_tf['compounds'],
                                  genes=valid_deep_tf['genes'],
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  drop_last=False,
                                  shuffle=False,
                                  train=True)

    # Setup model
    model = PerturbNet(n_input=train_x.shape[1])
    model_path = os.path.join(config.MODEL_PATH,
                              f'valid_cell_type_{args.valid_cell_type}.pth')

    if args.resume:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])

        train_loss = state_dict['train_loss']
        valid_loss = state_dict['valid_loss']
        train_mrrmse = state_dict['train_mrrmse']
        valid_mrrmse = state_dict['valid_mrrmse']

        logging.info(f'Loading model from check point!')
        logging.info(
            f'Train loss: {train_loss: .03f}; valid loss: {valid_loss: .03f}; train mrrmse: {train_mrrmse: .03f}; valid mrrmse: {valid_mrrmse: .03f}')

    model.to(device)

    # Setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],
                                 lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', min_lr=1e-5)

    """ Train the model """
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_prefix = f'{args.valid_cell_type}_{dt_string}'
    log_dir = os.path.join(config.TRAINING_LOG_PATH,
                           log_prefix)

    tb_writer = SummaryWriter(log_dir=log_dir)

    logging.info(f'Training started')

    if args.resume:
        best_valid_mrrmse = valid_mrrmse
    else:
        best_valid_mrrmse = 100

    for epoch in range(args.epochs):
        train_loss, train_mrrmse = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device)

        valid_loss, valid_mrrmse = valid(
            dataloader=valid_loader,
            model=model,
            criterion=criterion,
            device=device)

        tb_writer.add_scalar("Training loss", train_loss, epoch)
        tb_writer.add_scalar("Valid loss", valid_loss, epoch)
        tb_writer.add_scalar("Trianing MRRMSE", train_mrrmse, epoch)
        tb_writer.add_scalar("Vliad MRRMSE", valid_mrrmse, epoch)

        if valid_mrrmse < best_valid_mrrmse:
            best_valid_mrrmse = valid_mrrmse
            state = {'state_dict': model.state_dict(),
                     'train_loss': train_loss,
                     'valid_loss': valid_loss,
                     'train_mrrmse': train_mrrmse,
                     'valid_mrrmse': valid_mrrmse}
            torch.save(state, model_path)

        scheduler.step(valid_loss)

    logging.info(f'Training finished')


if __name__ == "__main__":
    main()
