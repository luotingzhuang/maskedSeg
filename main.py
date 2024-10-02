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
    #parser.add_argument(
    #    "--data_dir",
    #    type=str,
    #    default="",
    #    help="path to data directory",
    #)
    parser.add_argument("--csv_file", type=str, default="", help="name of csv file")
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="path to output directory"
    )

    # training
    parser.add_argument('--loss',  nargs='+', default=["DiceCELoss"], 
                        help = "loss functions: DiceCELoss, HausdorffDTLoss"
    )

    parser.add_argument(
        "--prior", action="store_true", default=False, help="include prior"
    )
    parser.add_argument("--seg_type", type=str, default="lung", help="segmentation type", choices=['lung','left_right_lung'])
    parser.add_argument("--mask", action="store_true", default=False, help="mask image")
    # parser.add_argument("--mask_size", type=int,  nargs='+', default=[7], help="mask size")
    # parser.add_argument("--offset", action="store_true", default=False, help="mask image offset")
    # parser.add_argument("--mask_percent", type=int,  nargs='+', default=[70], help="mask percent")
    parser.add_argument("--mask_dir", type=str, default="", help="path to mask directory")
    parser.add_argument("--prior_type", type=str, default="seg", help="prior type", choices=['seg','img','both'])
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze encoder")
    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--update_size", type=int, default=1, help="update size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    #parser.add_argument(
    #    "--es_criterion", type=str, default="total", help="early stopping criterion"
    #)
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

    # scheduler
    parser.add_argument("--sche", type=str, default=None, help="scheduler", choices=['cosine'])
    parser.add_argument(
        "--max_epoch",
        type=float,
        default=50,
        help="maximum number of epochs for scheduler",
    )

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
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint file"
    )
    args = parser.parse_args()
    return args


def main(args):
    # set seed
    set_seed(args.seed)


    # continue training on previous checkpoint
    if args.resume:
        # load args json
        with open(os.path.join(args.exp_dir, "args.json"), "r") as f:
            prev_args = json.load(f)

        # update args
        for key, value in prev_args.items():
            if key not in ["resume", "exp_dir", "epochs"]:
                setattr(args, key, value)
        # if isinstance(args.mask_size, int):
        #     args.mask_size = [args.mask_size]
        # if isinstance(args.mask_percent, int):
        #     args.mask_percent = [args.mask_percent]
        # init model
        model = Trainer(args)
        # load checkpoint
        model.ckpt_path = os.path.join(args.exp_dir, args.ckpt)
        model.load_prev()
        

    else:
        # create experiment folder
        exp_name = f"exp_{args.opt}_lr{args.lr}_bs{args.batch_size}_us{args.update_size}_seed{args.seed}"
        for loss in args.loss:
            exp_name += f"_{loss}"        
        if args.prior:
            exp_name += f"_prior{args.prior}_{args.prior_type}"
        #if args.mask:
            # exp_name += f"_mask{args.mask_size}_percent{args.mask_percent}"
            # if args.offset:
            #     exp_name += '_offset'
            
        if args.freeze:
            exp_name += '_freeze'

        if args.sche:
            exp_name += f"_{args.sche}_max{args.max_epoch}"


        exp_dir = os.path.join(args.result_dir, args.mask_dir.split('/')[-1] , exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError("Experiment folder already exists!")
        args.exp_dir = exp_dir

        # save args json
        with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        # init model
        model = Trainer(args)

    # init dataset
    train_dataset = TaskDataset(
        csv_file=args.csv_file,
        mode="train", 
        mask=args.mask, 
        mask_dir=args.mask_dir,
        # mask_size=args.mask_size, 
        # mask_percent=args.mask_percent,
        # offset=args.offset,
        seg_type=args.seg_type
    )
    val_dataset = TaskDataset(
        csv_file=args.csv_file, 
        mode="val",
        mask=args.mask,
        mask_dir=args.mask_dir,
        # mask_size=args.mask_size,
        # mask_percent=args.mask_percent,
        # offset=args.offset,
        #seg_type=args.seg_type
    )

    test_dataset = TaskDataset(
        csv_file=args.csv_file, 
        mode="test",
        mask=args.mask,
        mask_dir=args.mask_dir,
        # mask_size=args.mask_size,
        # mask_percent=args.mask_percent,
        # offset=args.offset,
        #seg_type=args.seg_type
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train
    results_val, results_test = model.train(train_loader, val_loader, test_loader)

    results_val.to_csv(os.path.join(args.exp_dir, "results_val.csv"))
    results_test.to_csv(os.path.join(args.exp_dir, "results_test.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
