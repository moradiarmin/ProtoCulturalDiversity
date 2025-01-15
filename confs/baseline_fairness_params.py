import torch
from ray import tune


base_param = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 30,
    'eval_neg_strategy': 'uniform',
    'val_batch_size': 256,
    'rec_sys_param': {'use_bias': 0},
}

base_hyper_params = {
    **base_param,
    'neg_train': tune.randint(1, 50),
    'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'batch_size': tune.lograndint(64, 512, 2),
    'optim_param': {
        'optim': tune.choice(['adam', 'adagrad']),
        'wd': tune.loguniform(1e-4, 1e-2),
        'lr': tune.loguniform(1e-4, 1e-1)
    },
}


proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        'embedding_dim': 93,

    'item_ft_ext_param': {
        'cosine_type': 'shifted',
        'ft_type': 'prototypes',
        'n_prototypes': tune.randint(10, 100),
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_batch_weight': tune.loguniform(1e-3, 10),
        'sim_proto_weight': tune.loguniform(1e-3, 10),
        'use_weight_matrix': False,
        'sim_proto_vectors_weight': 0.0,
        'sim_fairness_regularizer_weight': tune.choice([0.003, 0.01, 1.0, 5.0, 20.0]),
        'reg_proto_vectors_type': 'zerosum', # none, zerosum, none
        'k': -1, # -1 = inclusive, # post-process
        'initialize': 'random' # init ['random', 'zero', 'on_init_points']
        },
        
    'user_ft_ext_param': {
        'cosine_type': 'shifted',
        'ft_type': 'prototypes',
        'n_prototypes': tune.randint(10, 100),
        'reg_batch_type': 'max',
        'reg_proto_type': 'max',
        'sim_proto_weight': tune.loguniform(1e-3, 10),
        'sim_batch_weight': tune.loguniform(1e-3, 10),
        'use_weight_matrix': False,
        'sim_proto_vectors_weight': 0.0,
        'sim_fairness_regularizer_weight': tune.choice([0.0, 0.003, 0.01, 1.0, 5.0]),
        'reg_proto_vectors_type': 'zerosum',
        'k': -1, # -1 = inclusive, # post-process
        'initialize': 'random'
        }
    },
}

# initialize (initialize)
# regularizer coefficients (sim proto vec, reg proto vec)
# choose k (k)