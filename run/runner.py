import argparse

from exp.exp_pl import MMClassification
import torch
import pytorch_lightning as pl
import pathlib
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from dataset import *
from model import *
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from strategy import *
import os
import pandas as pd
from collections import OrderedDict

from utils import load_config, save_label_ids, concat_tensors_from_prediction
import time
import shutil
import pprint
import torch.nn as nn

AL_METRICS = OrderedDict()

MODEL = {
    "bert_resnet": BertResnet,
    "avmodel": AVModel,
    "avmodel_nl_gate": AVModelNLGate,
}


def load_previous_labeled_ids():
    # read labeled ids
    if current_round == 0:
        return []
    else:
        with open(last_round_logging_path / "label.txt", "r") as file:
            lines = file.readlines()
            labeled_ids = [str(line).rstrip() for line in lines]
        return labeled_ids


def query(query_budget):
    unlabeled_ids = datamodule.unlabeled_ids
    all_ids = datamodule.all_ids
    query_budget = query_budget if len(unlabeled_ids) >= query_budget else len(unlabeled_ids)
    print(f"query budget set to {query_budget}")
    if query_budget == 0:
        raise ValueError("All unlabeled datas are labeled")
    # query
    if current_round == 0 or len(unlabeled_ids) <= query_budget:
        query_start_time = time.time()
        sampling_method = RandomSampling(datamodule.unlabeled_ids)
        query_ids = sampling_method.query(initial_query_budget)
    elif strategy == "random":
        query_start_time = time.time()
        sampling_method = RandomSampling(datamodule.unlabeled_ids)
        query_ids = sampling_method.query(query_budget)
    else:

        # create Pytorch Lightning Trainer
        prediction_trainer = Trainer(
            enable_checkpointing=False,
            default_root_dir=config['logging_params']['save_dir'],
            max_epochs=1,
            devices=[config["trainer_params"]["devices"][0]],
            accelerator="gpu",
            logger=False,
            deterministic=True,
        )  # only use one GPU

        # create dataloader for querying
        if strategy in ["gcn", "uncertain_gcn", "kgreedycenter"]:
            query_dataloader = datamodule.whole_dataloader()
        elif strategy == "bald":
            experiment.mc_iteration = 10
            query_dataloader = datamodule.unlabeled_dataloader()
        else:
            query_dataloader = datamodule.unlabeled_dataloader()

        # predict features and logits for querying
        predict_results = prediction_trainer.predict(
            experiment,
            dataloaders=query_dataloader,
            ckpt_path=str(task_model_logging_path)
        )

        m1_probs, m2_probs, mm_probs, z1, z2, zm, contribution_m1, contribution_m2, \
            delta_m1, delta_m2, delta_m1_logits, delta_m2_logits = concat_tensors_from_prediction(predict_results)

        # querying
        query_start_time = time.time()

        if strategy == "bmmal":

            contribution_m1 = contribution_m1.gather(1, torch.argmax(mm_probs, dim=-1, keepdim=True))
            contribution_m2 = contribution_m2.gather(1, torch.argmax(mm_probs, dim=-1, keepdim=True))

            m1_strong_mask = contribution_m1 > contribution_m2
            m2_strong_mask = contribution_m2 >= contribution_m1

            scale_m1 = torch.empty_like(contribution_m1)
            scale_m2 = torch.empty_like(contribution_m2)

            scale_m1[m1_strong_mask] = 1
            scale_m1[m2_strong_mask] = 1 - (contribution_m2[m2_strong_mask] - contribution_m1[m2_strong_mask])

            scale_m2[m2_strong_mask] = 1
            scale_m2[m1_strong_mask] = 1 - (contribution_m1[m1_strong_mask] - contribution_m2[m1_strong_mask])

            tb_logger.experiment.add_scalar(f"Scale/{m1_name}_al_m1_strong",
                                            scale_m1[m1_strong_mask].mean(), current_round)
            tb_logger.experiment.add_scalar(f"Scale/{m1_name}_al_m2_strong",
                                            scale_m1[m2_strong_mask].mean(), current_round)

            tb_logger.experiment.add_scalar(f"Scale/{m2_name}_al_m1_strong",
                                            scale_m2[m1_strong_mask].mean(), current_round)
            tb_logger.experiment.add_scalar(f"Scale/{m2_name}_al_m2_strong",
                                            scale_m2[m2_strong_mask].mean(), current_round)

            sampling_method = BMMAL(unlabeled_ids,
                                    device=f"cuda:{config['trainer_params']['devices'][0]}")
            query_ids = sampling_method.query(
                n=query_budget, unimodal_z=[z1, z2],
                unimodal_probs=[mm_probs, mm_probs],
                unimodal_contributions=[scale_m1, scale_m2],
                num_classes=class_num,
                mm_probs=mm_probs,
                multilabel=multi_label)

        elif strategy == "entropy":
            sampling_method = EntropySampling(unlabeled_ids, multilabel=multi_label)
            query_ids = sampling_method.query(query_budget, mm_probs)
        elif strategy == "gcn":
            sampling_method = GCNSampling(unlabeled_sample_ids=unlabeled_ids, all_sample_ids=all_ids,
                                          device=f"cuda:{config['trainer_params']['devices'][0]}", method="CoreGCN")
            query_ids = sampling_method.query(query_budget, zm)
        elif strategy == "uncertain_gcn":
            sampling_method = GCNSampling(unlabeled_sample_ids=unlabeled_ids, all_sample_ids=all_ids,
                                          device=f"cuda:{config['trainer_params']['devices'][0]}",
                                          method="UncertainGCN")
            query_ids = sampling_method.query(query_budget, zm)
        elif strategy == "deepfool":
            experiment.load_state_dict(torch.load(task_model_logging_path)["state_dict"])
            if experiment.model.mm_classifier.__class__.__name__ == "ClassifierSum":
                clf = nn.Linear(zm.size(1), class_num)
                clf.weight.data[:, :z1.size(1)] = experiment.model.m1_classifiter.fc.weight.data
                clf.weight.data[:, z1.size(1):] = experiment.model.m2_classifiter.fc.weight.data
                clf.bias.data.fill_(0.)
                clf.bias.data += experiment.model.m1_classifiter.fc.bias.data
                clf.bias.data += experiment.model.m2_classifiter.fc.bias.data
                sampling_method = AdversarialDeepFool(unlabeled_ids, clf, max_iter=1)
            else:
                sampling_method = AdversarialDeepFool(unlabeled_ids, experiment.model.mm_classifier, max_iter=1)
            query_ids = sampling_method.query(query_budget, zm)
        elif strategy == "bald":
            sampling_method = BALD(unlabeled_ids)
            query_ids = sampling_method.query(query_budget, mm_probs, multilabel=multi_label)
        elif strategy == "badge":
            sampling_method = BADGE(unlabeled_ids,
                                    device=f"cuda:{config['trainer_params']['devices'][0]}")
            query_ids = sampling_method.query(query_budget, zm, mm_probs, class_num, multilabel=multi_label)
        elif strategy == "coreset":
            device = f"cpu"
            sampling_method = KCenterGreedy(unlabeled_ids, all_ids, splits=mini_batch)
            query_ids = sampling_method.query(query_budget, zm.to(device))
        elif strategy == "kmeans":
            sampling_method = KMeansSampling(unlabeled_ids)
            query_ids = sampling_method.query(query_budget, query_budget, zm)
        else:
            raise NotImplemented

    query_end_time = time.time()

    # add queried ids to labeled pool
    datamodule.query_for_label(query_ids)

    labeled_ids = datamodule.labeled_ids

    save_label_ids(labeled_ids, path=logging_path)

    return query_end_time - query_start_time


def fit():
    # train
    print(">>> unlabel pool size>>>", len(datamodule.unlabeled_ids))
    print(">>> labeled pool size>>>", len(datamodule.labeled_ids))

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if "early_stop" in config.keys() and config["early_stop"]:
        early_stop_callback = EarlyStopping(**config["early_stop_params"])
        callbacks = [lr_monitor, early_stop_callback]
    else:
        callbacks = [lr_monitor]

    # Init model weights
    experiment.model.__init__(**config["model_params"])

    fitting_trainer = Trainer(
        enable_checkpointing=False,
        default_root_dir=config['logging_params']['save_dir'],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        **config["trainer_params"]
    )

    fitting_trainer.fit(
        experiment,
        datamodule=datamodule
    )

    print(f"saving model to {task_model_logging_path}")
    fitting_trainer.save_checkpoint(
        filepath=task_model_logging_path,
        weights_only=True
    )

    if config["save_each_round_model"]:
        print(f"saving model to {each_round_task_model_logging_path}")
        fitting_trainer.save_checkpoint(
            filepath=each_round_task_model_logging_path,
            weights_only=True
        )

    test_metrics_dict = fitting_trainer.test(
        experiment,
        datamodule=datamodule,
        ckpt_path=str(task_model_logging_path),
        verbose=False
    )

    return test_metrics_dict


def test():
    # train
    print(">>> unlabel pool size>>>", len(datamodule.unlabeled_ids))
    print(">>> labeled pool size>>>", len(datamodule.labeled_ids))

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if "early_stop" in config.keys() and config["early_stop"]:
        early_stop_callback = EarlyStopping(**config["early_stop_params"])
        callbacks = [lr_monitor, early_stop_callback]
    else:
        callbacks = [lr_monitor]

    # Init model weights
    experiment.model.__init__(**config["model_params"])

    fitting_trainer = Trainer(
        enable_checkpointing=False,
        default_root_dir=config['logging_params']['save_dir'],
        logger=[csv_logger, tb_logger],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        **config["trainer_params"]
    )

    test_metrics_dict = fitting_trainer.test(
        experiment,
        datamodule=datamodule,
        ckpt_path=str(each_round_task_model_logging_path),
        verbose=False
    )

    return test_metrics_dict


def save_metrics(metrics_dict, query_time):
    # save all rounds of results into csv
    metrics_names = list(metrics_dict[0].keys()) + ['query_time']
    AL_METRICS[f'round {current_round}'] = list(metrics_dict[0].values()) + [query_time]
    if current_round == 0:
        metrics_dataframe = pd.DataFrame.from_dict(AL_METRICS, orient='index', columns=metrics_names)
    else:
        metrics_dataframe = pd.read_csv(last_round_logging_path / "al_metrics.csv", index_col=0)
        metrics_dataframe = pd.concat(
            [metrics_dataframe, pd.DataFrame.from_dict(AL_METRICS, orient='index', columns=metrics_names)])
    metrics_dataframe.to_csv(logging_path / "al_metrics.csv")


def parse_args():
    """
    Usage: python arrange_by_classes.py <path to downloaded dataset>
    """
    argparser = argparse.ArgumentParser('Run Active Learning on mmimdb dataset')
    argparser.add_argument('-c', '--config', type=str, required=True, help='path to config.yml')
    argparser.add_argument('-d', '--devices', type=str, help='select gpu devices, example --devices 0,1')
    argparser.add_argument('-s', '--strategy', type=str, help='select active learning strategy')
    argparser.add_argument('--seed', type=int, help='choose global seed for pytorch, numpy and random')
    argparser.add_argument('-r', '--round', type=int, required=True, help='current round number')
    argparser.add_argument('--dataset_dir', type=str, help='path to dataset directory')
    argparser.add_argument('--test', action="store_true", help='test model')
    args = argparser.parse_args()

    # load pre-set config file
    config = load_config(args.config)

    # update config according to args
    config["current_round"] = args.round
    if args.devices:
        config["trainer_params"]["devices"] = [int(d) for d in args.devices.split(',')]
    if args.strategy:
        config["query_strategy"] = args.strategy
    if args.seed:
        config['manual_seed'] = args.seed
    if args.dataset_dir:
        config['dataset_params']['root_dir'] = args.dataset_dir
    if "embedding_file" in config["model_params"].keys():
        config["model_params"]["embedding_file"] = \
            os.path.join(config["dataset_params"]["root_dir"], config["model_params"]["embedding_file"])

    if "initial_query_budget" not in config.keys():
        config["initial_query_budget"] = config["query_budget"]

    config["test"] = args.test

    return config


if __name__ == "__main__":
    # For reproduction
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    config = parse_args()

    assert config["query_strategy"] in \
           ["gcn", "uncertain_gcn", "bmmal", "random", "coreset",
            "entropy", "deepfool", "bald", "badge", "kmeans"]
    pprint.pprint(config, indent=4)

    strategy = config["query_strategy"]
    query_budget = config['query_budget']
    initial_query_budget = config['initial_query_budget']
    seed = config['manual_seed']
    current_round = config["current_round"]
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    multi_label = bool(config["multi_label"])
    class_num = config["model_params"]["class_num"]
    logging_root_dir = pathlib.Path(config["logging_params"]["save_dir"])
    mini_batch = config["mini_batch"]
    save_each_round_model = config["save_each_round_model"]
    logging_path = logging_root_dir / f"{dataset_name}-{strategy}-{seed}" / f"version_{current_round}"
    last_round_logging_path = logging_root_dir / f"{dataset_name}-{strategy}-{seed}" / f"version_{current_round - 1}"
    task_model_logging_path = logging_root_dir / f"{dataset_name}-{strategy}-{seed}" / "task_model.ckpt"
    each_round_task_model_logging_path = logging_path / "checkpoints" / "task_model.ckpt"
    m1_name = config["m1"]
    m2_name = config["m2"]

    # Datamodule setup
    datamodule = MultiModalDataModule(dataset_name=dataset_name, **config["dataset_params"])
    previous_labeled_ids = load_previous_labeled_ids()
    if current_round != 0:
        assert len(previous_labeled_ids) == query_budget * (
                current_round - 1) + initial_query_budget, "Error when loading labeled ids"
    datamodule.train_dataset.query_for_label(previous_labeled_ids)  # add previous labeled ids into datapool

    # Model setup
    model = MODEL[model_name](**config["model_params"])

    # query_budget
    if isinstance(query_budget, float):
        query_budget = int(query_budget * len(datamodule.train_dataset.all_ids))
    elif isinstance(query_budget, int):
        query_budget = int(query_budget)
    else:
        raise "Please check query budget"

    # Seed everything
    pl.seed_everything(seed, workers=True)

    # Loggers setup
    csv_logger = CSVLogger(
        save_dir=str(logging_root_dir),
        name=f"{dataset_name}-{strategy}-{seed}",
        version=current_round
    )
    tf_logger_path = logging_path / "tf"
    if os.path.exists(tf_logger_path):
        shutil.rmtree(tf_logger_path)
    tb_logger = TensorBoardLogger(
        save_dir=str(logging_root_dir),
        sub_dir='tf',
        name=f"{dataset_name}-{strategy}-{seed}",
        version=current_round
    )

    # log hyperparams
    csv_logger.log_hyperparams(config)

    # Experiment setup
    experiment = MMClassification(
        model,
        config,
        config["exp_params"],
        tb_logger
    )

    if config["test"]:
        metrics_dict = test()
        save_metrics(metrics_dict, 0)
    else:
        query_time = query(query_budget)

        metrics_dict = fit()

        save_metrics(metrics_dict, query_time)
