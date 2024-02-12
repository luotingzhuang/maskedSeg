#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils.train_model import Trainer
from dataset.dataloader import TaskDataset
from torch.utils.data import DataLoader
import json
from utils.train_utils import set_seed, seed_worker


def argParser():
    parser = ArgumentParser()
    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to data directory",
    )
    parser.add_argument("--csv_file", type=str, default="", help="name of csv file")
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="path to output directory"
    )

    # training
    parser.add_argument(
        "--loss",
        type=str,
        help="loss function",
        default="",
    )

    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    parser.add_argument(
        "--es_criterion", type=str, default="total", help="early stopping criterion"
    )
    parser.add_argument(
        "--es_warmup", type=int, default=0, help="early stopping warmup"
    )
    parser.add_argument(
        "--es_patience", type=int, default=20, help="early stopping patience"
    )
    parser.add_argument(
        "--log", action="store_true", default=False, help="log training"
    )
    parser.add_argument(
        "--print_every", type=int, default=10, help="print every n batches"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # resume if your model is interrupted
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="continue training",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="path to checkpoint",
    )
    args = parser.parse_args()
    return args


def main(args):
    # set seed
    set_seed(args.seed)

    if args.es_criterion not in args.loss and args.es_criterion != "total":
        raise ValueError("Early stopping criterion not in loss!")

    # continue training on previous checkpoint
    if args.resume:
        # load args json
        with open(os.path.join(args.exp_dir, "args.json"), "r") as f:
            prev_args = json.load(f)

        # update args
        for key, value in prev_args.items():
            if key not in ["resume", "exp_dir", "epochs"]:
                setattr(args, key, value)

        # init model
        model = Trainer(args, mode="train")
        # load checkpoint
        model.load_prev()

    else:
        # create experiment folder
        exp_name = ""

        exp_dir = os.path.join(args.result_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError("Experiment folder already exists!")
        args.exp_dir = exp_dir

        # save args json
        with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        # init model
        model = Trainer(args, mode="train")

    # init dataset
    train_dataset = TaskDataset(
        data_dir=args.data_dir, csv_file=args.csv_file, mode="train"
    )
    val_dataset = TaskDataset(
        data_dir=args.data_dir, csv_file=args.csv_file, mode="val"
    )

    # init dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train
    results = model.train(train_loader, val_loader)

    results.to_csv(os.path.join(args.exp_dir, "results.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)