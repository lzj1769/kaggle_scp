import os
import sys
import warnings
import numpy as np
import lightgbm as lgb
import logging
import argparse
import optuna
from optuna import Trial

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
    return parser.parse_args()
    

def objective(trial: Trial, fast_check=True, ):
    
    
    valid_score = 0
    for cell_type in ['nk', 't_cd4', 't_cd8', 't_reg']:
        train_data = np.load(f'../../../results/PerturbNet/deep_tf/train_{cell_type}.npz')
        valid_data = np.load(f'../../../results/PerturbNet/deep_tf/valid_{cell_type}.npz')
        
        model, preds, log = fit_lightgbm(trial, train_data, valid_data)
    
    
    return NotImplemented


def fit_lightgbm(trial, train_data, valid_data):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 128, 1024),
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.1,
        'num_threads': 16,
        "boosting": "gbdt",
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        "bagging_freq": 5,
        "bagging_fraction": trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        "feature_fraction": trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        "metric": 'l2',
        "verbosity": -1,
    }
    
    lgb_train = lgb.Dataset(train_data['x'], label=train_data['y'])
    lgb_valid = lgb.Dataset(valid_data['x'], label=valid_data['y'])
    
    model = lgb.train(params,
                      lgb_train,
                      valid_names=['train', 'valid'],
                      valid_sets=[lgb_train, lgb_valid],
                      verbose_eval=20,
                      early_stopping_rounds=20)
    
    # predictions
    preds = model.predict(lgb_valid, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    
    log = {'train/l2': model.best_score['train']['l2'],
           'valid/l2': model.best_score['valid']['l2']}
    
    return model, preds, log
    


def main():
    args = parse_args()
    
    logging.info(f'Validation cell type: {args.valid_cell_type}')
    
    # Setup data
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

if __name__ == '__main__':
    main()