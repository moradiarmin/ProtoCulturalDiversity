import argparse
import os

from confs.hyper_params import mf_hyper_params, anchor_hyper_params, user_proto_chose_original_hyper_params, \
    item_proto_chose_original_hyper_params
    
# baseline protomf
# from confs.hyper_params import proto_double_tie_chose_original_hyper_params

#baseline fairness
# from confs.baseline_fairness_params import proto_double_tie_chose_original_hyper_params

# LFM-2b
# from confs.double_tie_optim_hyper_params import proto_double_tie_chose_original_hyper_params # could change to general tuning
# from confs.hyper_params_ml import proto_double_tie_chose_original_hyper_params

# from confs.hyper_params_beauty_and_personal_care import proto_double_tie_chose_original_hyper_params
# from confs.hyper_params_grocery_and_gourmet import proto_double_tie_chose_original_hyper_params
from confs.hyper_params_musical_instruments import proto_double_tie_chose_original_hyper_params
# from confs.hyper_params_video_games import proto_double_tie_chose_original_hyper_params

from experiment_helper import start_hyper, start_multiple_hyper, start_training, start_testing
from utilities.consts import SINGLE_SEED

os.environ['WANDB_START_METHOD'] = 'thread'
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='Start an experiment')

# `python start.py -m 'user_item_proto' -d 'amazon2014/processed_ratings_musical_instruments'
parser.add_argument('--model', '-m', type=str, help='Recommender System model',
                    choices=['mf', 'acf', 'user_proto', 'item_proto', 'user_item_proto'])

parser.add_argument('--dataset', '-d', type=str, help='Recommender System Dataset',
                    choices=['amazon2014', 'ml-1m', 'lfm2b-1mon', 'amazon2014/Video_Games', 'amazon2014/processed_ratings_musical_instruments', 'amazon2014/processed_ratings_video_games', 'amazon2014/processed_ratings_beauty_and_personal_care', 'amazon2014/processed_ratings_grocery_and_gourmet_food', 'amazon2014/processed_ratings_kindle_store'])

parser.add_argument('--multiple', '-mp',
                    help='Whether to run the experiment across all seeds (see utilities/consts.py)',
                    action='store_true', default=False, required=False)
parser.add_argument('--seed', '-s', help='Seed to set for the experiments', type=int, default=SINGLE_SEED,
                    required=False)

args = parser.parse_args()

model = args.model
dataset = args.dataset
multiple = args.multiple
seed = args.seed

conf_dict = None
if model == 'mf':
    conf_dict = mf_hyper_params
elif model == 'acf':
    conf_dict = anchor_hyper_params
elif model == 'user_proto':
    conf_dict = user_proto_chose_original_hyper_params
elif model == 'item_proto':
    conf_dict = item_proto_chose_original_hyper_params
elif model == 'user_item_proto':
    conf_dict = proto_double_tie_chose_original_hyper_params

if multiple:
    start_multiple_hyper(conf_dict, model, dataset)
else:
    start_hyper(conf_dict, model, dataset, seed)