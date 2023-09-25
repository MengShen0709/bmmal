import yaml
import argparse
import os
import torch


def concat_tensors_from_prediction(predict_epoch_outputs):
    item_numbers = len(predict_epoch_outputs[0])
    concat_tensors = []
    for i in range(item_numbers):

        concat_tensor = [each_batch[i] for each_batch in predict_epoch_outputs]
        if concat_tensor[0] is None:
            concat_tensor = None
        else:
            if concat_tensor[0].dim() == 3:
                # size [mc_itration, num_data, feature_dim]
                concat_tensor = torch.cat(concat_tensor, dim=1)
            elif concat_tensor[0].dim() == 2:
                # size [num_data, feature_dim]
                concat_tensor = torch.cat(concat_tensor, dim=0)
            elif concat_tensor[0].dim() == 1:
                # size [num_data]
                concat_tensor = torch.cat(concat_tensor, dim=0)
            else:
                raise "don't support more than 3 dim"

        concat_tensors.append(concat_tensor)
    return concat_tensors


def load_config(config_file_path):
    # load config file
    with open(config_file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def parse_args(config):
    """
    Usage: python arrange_by_classes.py <path to downloaded dataset>
    """
    argparser = argparse.ArgumentParser('Run Active Learning on mmimdb dataset')
    argparser.add_argument('-d', '--devices', type=str, help='select gpu devices, example --devices 0,1')
    argparser.add_argument('-s', '--strategy', type=str, help='select active learning strategy')
    argparser.add_argument('--seed', type=int, help='choose global seed for pytorch, numpy and random')
    argparser.add_argument('-r', '--round', type=int, required=True, help='current round number')
    args = argparser.parse_args()
    if args.devices:
        print(config["trainer_params"]["devices"])
        config["trainer_params"]["devices"] = [int(d) for d in args.devices.split(',')]
        print(config["trainer_params"]["devices"])
    if args.strategy:
        config["query_strategy"] = args.strategy
    if args.seed:
        config["logging_params"]["manual_seed"] = args.seed
    config["current_round"] = args.round
    return config


def save_label_ids(label_ids, path):
    ids = label_ids.copy()
    ids.sort()
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "label.txt"), "w") as file:
        for id in ids:
            file.write(str(id) + "\n")
