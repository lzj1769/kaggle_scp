import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from torch.utils.tensorboard import SummaryWriter

from model import DeepTensorFactorization
from dataset import get_dataloader
from utils import set_seed, get_cell_type_compound_gene, compute_mean_row_wise_root_mse
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
                        help="Which cell type for validation. Available options are: nk,  t_cd4, t_cd8, t_reg")
    parser.add_argument("--log",
                        action="store_true",
                        help='write training history')
    parser.add_argument("--resume",
                        action="store_true",
                        help='training model from check point')
    parser.add_argument("--batch_size", default=50000, type=int,
                        help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    
    train_loss = 0.0
    cell_types, compunds, genes, targets, preds = list(), list(), list(), list(), list()
    for (cell_type_indices, compound_indices, gene_indices, target) in dataloader:
        pred = model(cell_type_indices.to(device), 
                     compound_indices.to(device), 
                     gene_indices.to(device))
        loss = criterion(pred.view(-1), target.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() / len(dataloader)
        
        # collect data to compute mrrmse
        cell_types.append(cell_type_indices.view(-1).tolist())
        compunds.append(compound_indices.view(-1).tolist())
        genes.append(gene_indices.view(-1).tolist())
        targets.append(target.view(-1).tolist())
        preds.append(pred.detach().cpu().view(-1).tolist())
        
    df = pd.DataFrame(data={'cell_type': np.concatenate(cell_types),
                            'sm_name': np.concatenate(compunds),
                            'gene': np.concatenate(genes),
                            'target': np.concatenate(targets),
                            'predict': np.concatenate(preds)})
    
    mrrmse = compute_mean_row_wise_root_mse(df)
    
    return train_loss, mrrmse
        
     
def valid(model, dataloader, criterion, device):
    model.eval()
    
    valid_loss = 0.0
    cell_types, compunds, genes, targets, preds = list(), list(), list(), list(), list()
    for (cell_type_indices, compound_indices, gene_indices, target) in dataloader:
        pred = model(cell_type_indices.to(device), 
                     compound_indices.to(device), 
                     gene_indices.to(device))
        
        loss = criterion(pred.view(-1), target.to(device))
        
        valid_loss += loss.item() / len(dataloader)
        
        # collect data to compute mrrmse
        cell_types.append(cell_type_indices.view(-1).tolist())
        compunds.append(compound_indices.view(-1).tolist())
        genes.append(gene_indices.view(-1).tolist())
        targets.append(target.view(-1).tolist())
        preds.append(pred.detach().cpu().view(-1).tolist())
        

    df = pd.DataFrame(data={'cell_type': np.concatenate(cell_types),
                            'sm_name': np.concatenate(compunds),
                            'gene': np.concatenate(genes),
                            'target': np.concatenate(targets),
                            'predict': np.concatenate(preds)})
    
    mrrmse = compute_mean_row_wise_root_mse(df)
    
    return valid_loss, mrrmse      
        
def main():
    args = parse_args()

    # set random seed
    set_seed(args.seed)

    # Setup CUDA, GPU
    device = torch.device("cuda")
    logging.info(f'Validation cell type: {args.valid_cell_type}')
    
    # Setup model
    cell_types, compounds, genes = get_cell_type_compound_gene()
    
    model = DeepTensorFactorization(cell_types=cell_types,
                                    compounds=compounds,
                                    genes=genes)
    model.to(device)
    model_path = os.path.join(config.MODEL_PATH,
                              f'valid_cell_type_{args.valid_cell_type}.pth')
    
    # Setup data
    logging.info(f'Loading data')
    
    df_train = pd.read_csv(f"{config.RESULTS_DIR}/df_train_{args.valid_cell_type}.csv") 
    df_valid = pd.read_csv(f"{config.RESULTS_DIR}/df_valid_{args.valid_cell_type}.csv") 
    df_train.target = df_train.target.astype(np.float32)
    df_valid.target = df_valid.target.astype(np.float32)

    train_loader = get_dataloader(df=df_train, 
                                  cell_types=cell_types, 
                                  compounds=compounds,
                                  genes=genes,
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  drop_last=False,
                                  shuffle=True,
                                  train=True)
    
    valid_loader = get_dataloader(df=df_valid, 
                                  cell_types=cell_types, 
                                  compounds=compounds,
                                  genes=genes,
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  drop_last=False,
                                  shuffle=False,
                                  train=True)

    # Setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4,
                                 weight_decay=1e-5)
    
    """ Train the model """
    log_prefix = f'valid_cell_type_{args.valid_cell_type}'
    log_dir = os.path.join(config.TRAINING_LOG_PATH,
                           log_prefix)
    

    tb_writer = SummaryWriter(log_dir=log_dir)
        
    logging.info(f'Training started')
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
            torch.save(model.state_dict(), model_path)
            best_valid_mrrmse = valid_mrrmse
        
    logging.info(f'Training finished')
 
if __name__ == "__main__":
    main()