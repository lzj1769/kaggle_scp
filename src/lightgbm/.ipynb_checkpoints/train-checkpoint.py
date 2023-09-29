import numpy as np
import lightgbm as lgb

train_data = np.load('../../results/lightgbm/train_nk.npz')
valid_data = np.load('../../results/lightgbm/valid_nk.npz')

train_data = lgb.Dataset(train_data['x'], label=train_data['y'])

param = {
    'task': 'train',
    'objective': 'regression',
    'boosting': 'gbdt',
    'data_sample_strategy': 'bagging',
    'num_iterations': 10,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'num_threads': 32,
    'seed': 42,
    'metric': 'l2',
    'max_bin': 255
}

bst = lgb.train(param, train_data)