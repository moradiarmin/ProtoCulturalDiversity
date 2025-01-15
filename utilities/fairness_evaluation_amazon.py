import ast
import pandas as pd
import os
from torch import nn
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np

from experiment_helper import load_data
from rec_sys.tester import Tester
from utilities.eval import Evaluator

def get_outputs_for_eval(model, test_loader):
    """
    Generate model outputs for evaluation.
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
    Returns:
        tuple: Outputs for items and users.
    """
    outs_array_items = []
    outs_array_users = []

    for u_idxs, i_idxs, labels in test_loader:
        out = model(u_idxs, i_idxs).detach()
        out = nn.Sigmoid()(out).cpu().numpy()
        outs_array_items.append(np.column_stack((i_idxs.numpy(), labels.numpy(), out)))
        outs_array_users.append(u_idxs.numpy())

    return outs_array_items, outs_array_users

def get_processed_outputs(items_output, countries_dic, items):
    """
    Process model outputs to extract metrics.
    Args:
        items_output (list): Model output for items.
        countries_dic (dict): Dictionary to store rankings by country.
        items (DataFrame): Metadata for items.
    Returns:
        tuple: Updated country dictionary and popularity-rank tuples.
    """
    pop_rank_tuples = []

    for epoch_recom_list in tqdm(items_output, desc="Processing Outputs"):
        for user_recom_list in epoch_recom_list:
            sorted_indices = np.argsort(user_recom_list[:, 2])[::-1]
            ranked_outputs = np.empty_like(user_recom_list[:, 2])
            ranked_outputs[sorted_indices] = np.arange(len(user_recom_list[:, 2]))
            user_recom_list = np.column_stack((user_recom_list, ranked_outputs))

            for item in user_recom_list:
                item_id = int(item[0])
                item_rank = int(item[3])

                item_pop = items.loc[items['item_id'] == item_id, 'num_interactions'].values[0]
                pop_rank_tuples.append((item_id, item_pop, item_rank))

                item_country = items.loc[items['item_id'] == item_id, 'country'].values[0]
                if item_country in countries_dic:
                    countries_dic[item_country].append(item_rank)

    return countries_dic, pop_rank_tuples

def calc_fairness_metrics(model_outputs, under_countries, over_countries, items):
    """
    Calculate fairness metrics.
    Args:
        model_outputs (tuple): Processed outputs (countries, popularity-rank tuples).
        under_countries (list): Underrepresented countries.
        over_countries (list): Overrepresented countries.
        items (DataFrame): Item metadata.
    Returns:
        dict: Fairness metrics.
    """
    under_ranks = [np.mean(model_outputs[0].get(country, [])) for country in under_countries]
    over_ranks = [np.mean(model_outputs[0].get(country, [])) for country in over_countries]

    lt_normal_threshold = items['num_interactions'].quantile(0.2)
    lt_log_threshold = np.log(items['num_interactions']).quantile(0.2) ** 2

    rank_tuples = model_outputs[1]
    lt_array = np.array([
        (rank, pop < lt_normal_threshold, pop < lt_log_threshold)
        for _, pop, rank in rank_tuples
    ])

    metrics = {
        'under_countries_avg_rank': np.nanmean(under_ranks),
        'over_countries_avg_rank': np.nanmean(over_ranks),
        'mean_rank_lt_normal': np.nanmean(lt_array[lt_array[:, 1], 0]),
        'mean_rank_lt_log': np.nanmean(lt_array[lt_array[:, 2], 0]),
        'unique_values_normal': len(np.unique(lt_array[lt_array[:, 1], 0])),
        'unique_values_log': len(np.unique(lt_array[lt_array[:, 2], 0])),
    }
    return metrics

# Paths and configurations
FAIRNESS_METRICS_PATH = "logs/fairness_metrics_eval.csv"
COUNTRY_GROUPS_PATH = "search_addresses/dataset_country_groups.json"
MODEL_PATHS_FILE = "configs/model_paths.txt"
CATEGORIES = ['MUSICAL INSTRUMENTS', 'GROCERY AND GOURMET', 'VIDEO GAMES', 'BEAUTY']
DATASET_PATHS = ['data/amazon2014/processed_ratings_musical_instruments',\
                 'data/amazon2014/processed_ratings_grocery_and_gourmet_food',\
                 'data/amazon2014/processed_ratings_video_games',\
                 'data/amazon2014/processed_ratings_beauty_and_personal_care']


for category, dataset_path in zip(CATEGORIES, DATASET_PATHS):
    users = pd.read_csv(os.path.join(dataset_path, 'user_ids.csv'))
    items = pd.read_csv(os.path.join(dataset_path, 'items_metadata.csv'))
    test_data = pd.read_csv(os.path.join(dataset_path, 'test_data.csv'))
    train_data = pd.read_csv(os.path.join(dataset_path, 'train_data.csv'))

    items['num_interactions'] = train_data.groupby('item_id').size()
    items['num_interactions'].fillna(items['num_interactions'].mean(), inplace=True)
    items['item_id'] = items.index

    with open(COUNTRY_GROUPS_PATH, 'r') as file:
        country_data = json.load(file)
        over_countries = country_data[category]['overrepresented_countries']
        under_countries = country_data[category]['underrepresented_countries']

    countries_dic = {country: [] for country in over_countries + under_countries}

    model_paths = []
    with open(MODEL_PATHS_FILE, 'r') as f:
        for line in f:
            if category in line:
                model_paths.append(line.split('=')[1].strip())

    for model_path in model_paths:
        model_checkpoint = os.path.join(model_path, 'checkpoint/best_model.pth')
        with open(os.path.join(model_path, 'params.json'), 'r') as f:
            config = json.load(f)

        config['device'] = 'cpu'
        config = argparse.Namespace(**config)
        data_loaders = load_data(config, is_train=False)
        tester = Tester(data_loaders['test_loader'], config, model_checkpoint)

        model = tester.model
        model.eval()

        item_outputs = get_outputs_for_eval(model, data_loaders['test_loader'])[0]
        processed_outputs = get_processed_outputs(item_outputs, countries_dic, items)
        fairness_metrics = calc_fairness_metrics(processed_outputs, under_countries, over_countries, items)

        df = pd.read_csv(FAIRNESS_METRICS_PATH)
        fairness_metrics.update({'model_path': model_path, 'dataset_path': dataset_path})
        df = pd.concat([df, pd.DataFrame([fairness_metrics])], ignore_index=True)
        df.to_csv(FAIRNESS_METRICS_PATH, index=False)

print("Evaluation completed.")