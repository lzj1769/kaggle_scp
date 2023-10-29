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
    parser.add_argument("--batch_size", default=50000, type=int,
                        help="Batch size. Default 5000")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of training epochs to perform. Default: 100")
    parser.add_argument("--lr", default=3e-04, type=float,
                        help="Learning rate. Default: 0.001")
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
    
    mrrmse = compute_mrrmse(df)
    
    return train_loss, mrrmse
        
   
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
    model_path = '../../model/deep_tf_v4/model.pth'
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
                                  num_workers=5,
                                  drop_last=False,
                                  shuffle=True,
                                  train=True)

    # Setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad == True],
                                 lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)

    """ Train the model """
    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    log_prefix = f'{dt_string}'
    log_dir = os.path.join('./log', log_prefix)
    tb_writer = SummaryWriter(log_dir=log_dir)
        
    logging.info(f'Training started')
    for epoch in range(args.epochs):
        train_loss, train_mrrmse = train(
            dataloader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device)
        
        tb_writer.add_scalar("Training loss", train_loss, epoch)
        tb_writer.add_scalar("Trianing mrrmse", train_mrrmse, epoch)
            
        state = {'state_dict': model.state_dict(),
                 'train_loss': train_loss,
                 'train_mrrmse': train_mrrmse}
        torch.save(state, model_path)
        
        scheduler.step()
        
    logging.info(f'Training finished')
 
if __name__ == "__main__":
    main()
