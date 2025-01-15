import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from rec_sys.tester import Tester
from experiment_helper import load_data
from utilities.eval import Evaluator


def calc_fairness_metrics(model_outputs, train_dataset_track_augmented):
    """
    Calculate fairness metrics based on model outputs.
    Args:
        model_outputs (tuple): Processed model outputs (countries, genders, pop_rank_tuples).
        train_dataset_track_augmented (DataFrame): Dataset with track augmentations.
    Returns:
        dict: Fairness metrics.
    """
    under_countries = ['IE', 'JP', 'KR', 'HK', 'CN']
    over_countries = ['US', 'GB', 'FR']
    genders = ['Female', 'Male']

    # Country-wise rankings
    under_ranks = [np.mean(model_outputs[0].get(country, [])) for country in under_countries]
    over_ranks = [np.mean(model_outputs[0].get(country, [])) for country in over_countries]

    # Gender-wise rankings
    female_ranks = np.mean(model_outputs[1].get('Female', []))
    male_ranks = np.mean(model_outputs[1].get('Male', []))

    # Popularity-based rankings
    lt_normal_threshold = train_dataset_track_augmented['item_num_interactions'].quantile(0.2)
    lt_log_threshold = np.log(train_dataset_track_augmented['item_num_interactions']).quantile(0.2) ** 2

    rank_tuples = model_outputs[2]
    lt_array = np.array([
        (rank, pop < lt_normal_threshold, pop < lt_log_threshold) 
        for _, pop, rank in rank_tuples
    ])

    metrics = {
        'under_countries_avg_rank': np.mean(under_ranks),
        'over_countries_avg_rank': np.mean(over_ranks),
        'mean_rank_female': female_ranks,
        'mean_rank_male': male_ranks,
        'mean_rank_lt_normal': np.mean(lt_array[lt_array[:, 1], 0]),
        'mean_rank_lt_log': np.mean(lt_array[lt_array[:, 2], 0]),
        'unique_values_normal': len(np.unique(lt_array[lt_array[:, 1], 0])),
        'unique_values_log': len(np.unique(lt_array[lt_array[:, 2], 0])),
    }
    return metrics


def get_outputs_for_eval(model, test_loader):
    """
    Generate model outputs for evaluation.
    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
    Returns:
        tuple: Outputs for items and users.
    """
    outs_items, outs_users = [], []

    for u_idxs, i_idxs, labels in test_loader:
        outputs = model(u_idxs, i_idxs).detach()
        outputs = nn.Sigmoid()(outputs).cpu().numpy()
        loss = model.loss_func(outputs, labels).item()
        
        outs_items.append(np.column_stack((i_idxs, labels, outputs)))
        outs_users.append(u_idxs.numpy())

    return outs_items, outs_users


def get_processed_outputs(outs_items, train_dataset_track_augmented):
    """
    Process model outputs to extract relevant metrics.
    Args:
        outs_items (list): Model outputs for items.
        train_dataset_track_augmented (DataFrame): Dataset with track augmentations.
    Returns:
        tuple: Processed countries, genders, and popularity-rank tuples.
    """
    countries = {code: [] for code in ['US', 'GB', 'FR', 'IE', 'JP', 'KR', 'HK', 'CN']}
    genders = {'Male': [], 'Female': []}
    pop_rank_tuples = []

    for epoch_outputs in tqdm(outs_items, desc="Processing Outputs"):
        for user_outputs in epoch_outputs:
            sorted_indices = np.argsort(user_outputs[:, 2])[::-1]
            ranked_outputs = np.empty_like(user_outputs[:, 2])
            ranked_outputs[sorted_indices] = np.arange(len(user_outputs[:, 2]))
            user_outputs = np.column_stack((user_outputs, ranked_outputs))

            for item in user_outputs:
                item_id, _, _, item_rank = map(int, item)
                try:
                    item_pop = train_dataset_track_augmented.loc[
                        train_dataset_track_augmented['item_id'] == item_id, 'item_num_interactions'
                    ].iloc[0]
                    pop_rank_tuples.append((item_id, item_pop, item_rank))
                except IndexError:
                    continue

                item_country = train_dataset_track_augmented.loc[
                    train_dataset_track_augmented['item_id'] == item_id, 'country'
                ].iloc[0]
                if item_country in countries:
                    countries[item_country].append(item_rank)

                item_gender = train_dataset_track_augmented.loc[
                    train_dataset_track_augmented['item_id'] == item_id, 'gender'
                ].iloc[0]
                if item_gender in genders:
                    genders[item_gender].append(item_rank)

    return countries, genders, pop_rank_tuples


# Main execution
if __name__ == "__main__":
    print("Starting evaluations...")

    dataset_path = 'data/lfm2b-1mon'
    users = pd.read_csv(os.path.join(dataset_path, 'users.tsv'), sep='\t')
    tracks = pd.read_csv(os.path.join(dataset_path, 'tracks_augmented.csv'))

    # Process dataset
    train_dataset = pd.read_csv(os.path.join(dataset_path, 'listening_history_train.csv'))
    train_dataset_track_augmented = train_dataset.merge(tracks, on='old_item_id')

    # Model evaluation
    model_paths = []  # Add model paths here
    for model_path in model_paths:
        with open(os.path.join(model_path, 'params.json'), 'r') as f:
            config = argparse.Namespace(**json.load(f))

        test_loader = load_data(config, is_train=False)['test_loader']
        tester = Tester(test_loader, config, os.path.join(model_path, 'checkpoint_000000/best_model.pth'))
        model = tester.model

        outs_items, _ = get_outputs_for_eval(model, test_loader)
        processed_outputs = get_processed_outputs(outs_items, train_dataset_track_augmented)
        metrics = calc_fairness_metrics(processed_outputs, train_dataset_track_augmented)

        with open("all_fairness_evaluations_LFM.json", "a") as json_file:
            json.dump(metrics, json_file)

    print("Evaluation complete.")
