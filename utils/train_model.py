import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import time
import math
from argparse import ArgumentParser
from model.model import Model
from utils.loss_utils import loss1
from utils.train_utils import EarlyStopping
from utils.metric_utils import metric1
from tensorboardX import SummaryWriter


class baseTrainer:
    def __init__(self, args: ArgumentParser) -> None:
        self.args = args
        self.opt = args.opt
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.loss = args.loss
        self.es = args.es
        self.es_warmup = args.es_warmup
        self.es_patience = args.es_patience

        self.exp_dir = args.exp_dir
        self.log = args.log
        self.print_every = args.print_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__init_model()
        self.__init_loss()
        self.__init_optimizer()
        self.__init_scheduler()
        self.__init_scaler()
        self.__init_logger()
        self.__init_es()

    def __init_optimizer(self):
        print(f"Initiate {self.opt} optimizer", end=" ")

        if self.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        print("...done")

    def __init_loss(self):
        print(f"Initiate {self.loss} loss", end=" ")
        self.loss_fn = loss1()
        print("...done")

    def __init_model(self):
        print(f"Initiate  model", end=" ")
        self.model = Model()
        print("...done")

        if self.freeze:
            print(f"Freezing {self.freeze}")
            for name, param in self.model.named_parameters():
                if self.freeze in name:
                    param.requires_grad = False

        self.model.to(self.device)

    def __init_logger(self):
        if self.log:
            print("Initiate tensorboard logger", end=" ")
            self.log_dir = os.path.join(self.exp_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print("...done")
        else:
            self.writer = None

    def __init_es(self):
        if self.es:
            print(
                "Initiate early stopping with warmup: {} and patience: {}".format(
                    self.es_warmup, self.es_patience
                ),
                end=" ",
            )
            self.early_stopping = EarlyStopping(
                warmup=self.es_warmup, patience=self.es_patience, verbose=True
            )
            print("...done")
        else:
            self.early_stopping = None


class Trainer(baseTrainer):
    def __init__(self, args, mode: str = "train") -> None:
        super(Trainer, self).__init__(args, mode=mode)
        self.start_epoch = 0
        self.epochs = args.epochs
        self.ckpt_path = os.path.join(args.exp_dir, "checkpoint.pth.tar")
        self.es_ckpt_path = os.path.join(args.exp_dir, "es_checkpoint.pth.tar")

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> pd.DataFrame:
        # loop through epochs
        for i in range(self.start_epoch, self.epochs):
            self.model.train()
            print(f"------------Epoch {i+1}/{self.epochs}------------")
            # training
            train_loss_sum = 0
            train_metric = []
            for batch_idx, (input_data, label) in enumerate(train_loader):
                input_data = input_data.to(self.device)

                # run model
                model_out = self.model(input_data)

                # compute metric
                batch_metric = self.__compute_metric(model_out, label)
                train_metric.extend(batch_metric)

                # compute loss
                train_loss = self.__compute_loss(self.loss_fn, model_out, label)

                # backprop
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss_sum += train_loss.item()

                # training progess batch and loss
                if batch_idx % self.print_every == 0:
                    print(
                        "Batch {} - Train Loss: {:.6f}".format(
                            batch_idx, train_loss.item()
                        )
                    )

            train_loss_mean = train_loss_sum / len(train_loader)
            train_metric_mean = np.mean(train_metric)
            print(
                "Epoch {} - Train Loss: {:.6f}; Metric: {:.6f}".format(
                    i, train_loss_mean, train_metric_mean
                )
            )

            # log training
            if self.writer:
                self.writer.add_scalar("train/Total_loss", train_loss_mean, i)
                self.writer.add_scalar("train/Metric", train_metric_mean, i)

            # validation
            earlystop = self.__eval(i, val_loader)

            # save model for each epoch
            state = {
                "epoch": i + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "early_stopping": self.early_stopping,
            }

            torch.save(state, self.ckpt_path)

            if earlystop:
                break

        print("Finished Training...")
        results = self.predict(val_loader)

        if self.writer:
            self.writer.close()

        return results

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader) -> bool:
        val_loss_sum = 0
        val_metric = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (input_data, label) in enumerate(val_loader):
                input_data = input_data.to(self.device)
                # run model
                model_out = self.model(input_data)

                # compute metric
                batch_metric = self.__compute_metric(model_out, label)
                val_metric.extend(batch_metric)

                # compute loss
                val_loss = self.__compute_loss(self.loss_fn, model_out, label)
                val_loss_sum += val_loss.item()

        val_loss_mean = val_loss_sum / len(val_loader)
        val_metric_mean = np.mean(val_metric)
        print(
            "Epoch {} - Validation Loss : {:.6f}; Metric: {:.6f} ".format(
                cur, val_loss_mean, val_metric_mean
            )
        )

        if self.writer:
            self.writer.add_scalar("val/Total_loss", val_loss_mean, cur)
            self.writer.add_scalar("val/Metric", val_metric_mean, cur)

        # early stopping
        if self.es:
            self.early_stopping(
                epoch=cur,
                val_loss=val_loss_mean,
                model=self.model,
                ckpt_path=self.es_ckpt_path,
            )
            return self.early_stopping.early_stop
        else:
            return False

    def predict(self, val_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
        # load final model
        if self.es:
            ckpt_path = self.es_ckpt_path
        else:
            ckpt_path = self.ckpt_path

        print("=== Load model from {} ===".format(ckpt_path))
        self.model.load_state_dict(torch.load(ckpt_path)["model"])
        self.model.eval()

        val_loss_sum = 0
        val_metric = []

        with torch.no_grad():
            for batch_idx, (input_data, label) in enumerate(val_loader):
                input_data = input_data.to(self.device)
                # run model
                model_out = self.model(input_data)

                # compute metric
                batch_metric = self.__compute_metric(model_out, label)
                val_metric.extend(batch_metric)

                # compute loss
                val_loss = self.__compute_loss(self.loss_fn, model_out, label)
                val_loss_sum += val_loss.item()

        val_loss_mean = val_loss_sum / len(val_loader)
        val_metric_mean = np.mean(val_metric)
        print(
            "Final Validation Loss : {:.6f}; Metric: {:.6f} ".format(
                val_loss_mean, val_metric_mean
            )
        )

        # save results
        results = pd.DataFrame(
            {
                "val": [
                    val_loss_mean,
                    val_metric_mean,
                ],
            },
            index=["loss", "metric"],
        )
        results = pd.concat([results], axis=0)

        return results

    @staticmethod
    def __compute_metric(model_out, label):
        return batch_metric

    @staticmethod
    def __compute_loss(model_out, label):
        return loss

    def load_prev(self):
        print("Continue training from checkpoint: {}".format(self.ckpt_path))
        if os.path.isfile(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path)
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.early_stopping = ckpt["early_stopping"]
        else:
            raise ValueError("No checkpoint found at '{}'".format(self.ckpt_path))
