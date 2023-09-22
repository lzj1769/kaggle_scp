import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch
from datetime import datetime

from model import DeepTensorFactorization
from dataset import get_dataloader
from utils import set_seed, get_cell_type_compound_gene
import config

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
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def compute_mean_row_wise_root_mse(model, df, device, cell_types, compounds, genes):
    model.eval()
    dataloader = get_dataloader(df=df, 
                                cell_types=cell_types, 
                                compounds=compounds,
                                genes=genes,
                                batch_size=10000,
                                num_workers=2,
                                drop_last=False,
                                shuffle=False,
                                train=False)
    
    preds = list()
    for (cell_type_indices, compound_indices, gene_indices) in dataloader:
        cell_type_indices = cell_type_indices.to(device)
        compound_indices = compound_indices.to(device)
        gene_indices = gene_indices.to(device)

        pred = model(cell_type_indices, compound_indices, gene_indices).detach().cpu().view(-1).tolist()
        preds.append(pred)
        
    df['predict'] = np.concatenate(preds)
    df['diff'] = (df['target'] - df['predict']) ** 2
    df = df.drop(['gene', 'target', 'predict'], axis=1)
    grouped = df.groupby(['cell_type', 'sm_name'])
    df = grouped.mean()
    mrrmse = np.mean(np.sqrt(df['diff']))
    
    return mrrmse


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    
    train_loss = 0.0
    for (cell_type_indices, compound_indices, gene_indices, target) in dataloader:
        cell_type_indices = cell_type_indices.to(device)
        compound_indices = compound_indices.to(device)
        gene_indices = gene_indices.to(device)
        target = target.to(device)

        pred = model(cell_type_indices, compound_indices, gene_indices)
        loss = criterion(pred.view(-1), target)
        
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
    if not torch.cuda.is_available():
        print("cuda is not available")
        exit(0)
    else:
        device = torch.device("cuda")
    
    #
    print(f'Validation cell type: {args.valid_cell_type}')
    
    # Setup model
    cell_types, compounds, genes = get_cell_type_compound_gene()
    
    model = DeepTensorFactorization(cell_types=cell_types,
                                    compounds=compounds,
                                    genes=genes)
    model.to(device)
    model_path = os.path.join(config.MODEL_PATH,
                              f'valid_cell_type_{args.valid_cell_type}_.pth')
    
    # Setup data
    current_time = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    print(f"loading data: {current_time}")
    
    df_train = pd.read_csv(f"{config.RESULTS_DIR}/df_train_{args.valid_cell_type}.csv") 
    df_valid = pd.read_csv(f"{config.RESULTS_DIR}/df_valid_{args.valid_cell_type}.csv") 
    df_train.target = df_train.target.astype(np.float32)
    df_valid.target = df_valid.target.astype(np.float32)

    train_loader = get_dataloader(df=df_train, 
                                  cell_types=cell_types, 
                                  compounds=compounds,
                                  genes=genes,
                                  batch_size=10000,
                                  num_workers=2,
                                  drop_last=True,
                                  shuffle=True,
                                  train=True)

    # Setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4,
                                 weight_decay=1e-5)
    
    current_time = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
    print(f'training started: {current_time}')
    for epoch in range(args.epochs):
        train_loss = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device)
        
        train_mrrmse = compute_mean_row_wise_root_mse(model=model, 
                                                      df=df_train, 
                                                      device=device,
                                                      cell_types=cell_types,
                                                      compounds=compounds,
                                                      genes=genes)
        
        valid_mrrmse = compute_mean_row_wise_root_mse(model=model,
                                                      df=df_valid,
                                                      device=device,
                                                      cell_types=cell_types,
                                                      compounds=compounds,
                                                      genes=genes)
        
        current_time = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
        print(f"{current_time}; epoch: {epoch}; training loss: {train_loss: .2f}; training MRRMSE: {train_mrrmse: .2f}; validation MRRMSE: {valid_mrrmse: .2f}")
        
if __name__ == "__main__":
    main()