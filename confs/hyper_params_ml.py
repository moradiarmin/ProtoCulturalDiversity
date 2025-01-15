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
    
    # 'batch_size': tune.lograndint(64, 512, 2),
    'batch_size': 326,
    # 'neg_train': tune.randint(1, 50),
    'neg_train': 38,
    # 'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'train_neg_strategy': 'uniform',
    # 'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'loss_func_name': 'sampled_softmax',
    'optim_param': {
        # 'optim': tune.choice(['adam', 'adagrad']),
        'optim': 'adam',
        # 'wd': tune.loguniform(1e-4, 1e-2),
        'wd': 0.00016743350797948322,
        # 'lr': tune.loguniform(1e-4, 1e-1)
        'lr': 0.0030358481282326704
    },
}


proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        
        # 'embedding_dim': tune.randint(10, 100),
        'embedding_dim': 81,
        
        'item_ft_ext_param': {
            # 'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_proto_weight': 1.803667314461998,
            # 'sim_batch_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': 0.10505406949587856,
            # 'n_prototypes': tune.randint(10, 100),
            'n_prototypes': 43,
            
            "ft_type": "prototypes_double_tie",
            'initialize': 'random',
            'use_weight_matrix': False,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_proto_vectors_type': 'ortho',
            'reg_batch_type': 'max',
    
            'out_dimension': 89,
            'sim_proto_vectors_weight': tune.choice([0.01, 0.1, 1.0, 10.0]),
            'k': tune.choice([12, 25, -1]), # -1 = inclusive, # post-process
        },
        
               
        'user_ft_ext_param': {
            # 'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_proto_weight': 0.08608508477545103,
            # 'sim_batch_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': 0.049638023486463796,
            # 'n_prototypes': tune.randint(10, 100),
            'n_prototypes': 71,
            
            "ft_type": "prototypes_double_tie",
            'initialize': 'random',
            'use_weight_matrix': False,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_proto_vectors_type': 'ortho',
            'reg_batch_type': 'max',
            
            'out_dimension': 89,
            'sim_proto_vectors_weight': tune.choice([0.01, 0.1, 1.0, 10.0]),
            'k': tune.choice([12, 25, 40, 60, -1]) # -1 = inclusive, # post-process
        },
    },
}