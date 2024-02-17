import json

import torch
from torch.utils.data import random_split
from dp.preprocess import preprocess
from dp.train import train
DATA_PATH = "Data/DataSet.json"


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:  # Open the file in read mode ,with utf-8 for arabic
        try:
            json_data = json.load(file)
            key_value_pairs = list(json_data.items())  # Create a list of key-value pairs
            dataset = [('ar', word, pronunciation) for word, pronunciation in key_value_pairs]
            return dataset
        except json.JSONDecodeError as e:
            print(f"Error loading JSON: {e}")


def GetDataSet():
    return load_data(DATA_PATH)


if __name__ == '__main__':
    dataset_raw = GetDataSet()
    train_dataset_raw_size = int(0.9 * len(dataset_raw))
    val_dataset_raw_size = len(dataset_raw) - train_dataset_raw_size
    train_data, val_data = random_split(dataset_raw, [train_dataset_raw_size, val_dataset_raw_size])
    config_file = 'autoreg_config.yaml'
    preprocess(config_file=config_file,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=False,)
    num_gpus = torch.cuda.device_count()

    train(rank=0, num_gpus=num_gpus, config_file=config_file,checkpoint_file='checkpoints/best_model.pt')
