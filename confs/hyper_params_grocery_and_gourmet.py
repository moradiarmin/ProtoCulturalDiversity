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
    'batch_size': 313,
    # 'neg_train': tune.randint(1, 50),
    'neg_train': 12,
    # 'train_neg_strategy': tune.choice(['popular', 'uniform']),
    'train_neg_strategy': 'uniform',
    # 'loss_func_name': tune.choice(['bce', 'bpr', 'sampled_softmax']),
    'loss_func_name': 'sampled_softmax',
    'optim_param': {
        # 'optim': tune.choice(['adam', 'adagrad']),
        'optim': 'adagrad',
        # 'wd': tune.loguniform(1e-4, 1e-2),
        'wd': 0.0007576695565756338,
        # 'lr': tune.loguniform(1e-4, 1e-1)
        'lr': 0.038545894730225266
    },
}


proto_double_tie_chose_original_hyper_params = {
    **base_hyper_params,
    'loss_func_aggr': 'mean',
    'ft_ext_param': {
        "ft_type": "prototypes_double_tie",
        
        # 'embedding_dim': tune.randint(10, 100),
        'embedding_dim': 93,
        
        'item_ft_ext_param': {
            # 'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_proto_weight': 1.2246719945476514,
            # 'sim_batch_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': 0.0016067743528386975,
            # 'n_prototypes': tune.randint(10, 100),
            'n_prototypes': 17,
            
            "ft_type": "prototypes_double_tie",
            'initialize': 'random',
            'use_weight_matrix': False,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_proto_vectors_type': 'ortho',
            'reg_batch_type': 'max',
    
            'out_dimension': 89,
            'sim_proto_vectors_weight': tune.choice([0.01, 0.1, 1.0, 10.0]),
            'k': tune.choice([12, -1]), # -1 = inclusive, # post-process
        },
        
               
        'user_ft_ext_param': {
            # 'sim_proto_weight': tune.loguniform(1e-3, 10),
            'sim_proto_weight': 0.003926141404662937,
            # 'sim_batch_weight': tune.loguniform(1e-3, 10),
            'sim_batch_weight': 0.04540597537543749,
            # 'n_prototypes': tune.randint(10, 100),
            'n_prototypes': 89,
            
            "ft_type": "prototypes_double_tie",
            'initialize': 'random',
            'use_weight_matrix': False,
            'cosine_type': 'shifted',
            'reg_proto_type': 'max',
            'reg_proto_vectors_type': 'ortho',
            'reg_batch_type': 'max',
            
            'out_dimension': 89,
            'sim_proto_vectors_weight': tune.choice([0.01, 0.1, 1.0, 10.0]),
            'k': tune.choice([12, 25, 40, 60, -1, -1]) # -1 = inclusive, # post-process
        },
    },
}