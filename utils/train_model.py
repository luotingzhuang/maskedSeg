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
from model.unet import PlainConvUNet
from model.initialization.weight_init import InitWeights_He
from utils.loss_utils import BoundaryLoss, BoundaryDoULoss
#from utils.loss_utils import loss1
from utils.train_utils import EarlyStopping
from utils.metric_utils import DiceCoefficient
import logging
from tensorboardX import SummaryWriter
import monai
import sys
sys.path.insert(0,'/workspace/radraid/projects/seg_lung/masked_seg/totalsegmentator/nnUNet')
sys.path.insert(0,'/workspace/radraid/projects/seg_lung/masked_seg/totalsegmentator/dynamic-network-architectures')

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import  determine_num_input_channels
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from tqdm import tqdm

import matplotlib.pyplot as plt

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
        self.update_size = args.update_size
        self.prior = args.prior
        self.prior_type = args.prior_type
        self.freeze = args.freeze
        self.sche = args.sche
        if self.sche:
            self.sche_param = {
                "cosine": {"max_epoch": args.max_epoch},
                #"lambdacosine": {"max_epoch": args.max_epoch, "lr_factor": args.lrf},
            }
        self.exp_dir = args.exp_dir
        self.log = args.log
        self.print_every = args.print_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seg_type = args.seg_type

        self.__init_model()
        self.__init_loss()
        self.__init_optimizer()
        self.__init_scheduler()
        #self.__init_scaler()
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
        self.DiceCE_loss = monai.losses.DiceCELoss(include_background = False,softmax=True, squared_pred=True, reduction='mean', to_onehot_y = True)
        self.HausdorffDT_loss = monai.losses.HausdorffDTLoss(softmax=True, reduction='mean')
        #self.boundary_loss = BoundaryLoss(1)
        print("...done")

    def __init_model(self):
        print(f"Initiate  model", end=" ")

        total_weights = torch.load('/workspace/radraid/projects/seg_lung/masked_seg/totalsegmentator/checkpoint_final.pth')
        plans_manager = PlansManager(total_weights['init_args']['plans'])
        configuration_manager = plans_manager.get_configuration(total_weights['init_args']['configuration'])
        dataset_json = total_weights['init_args']['dataset_json']
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager,dataset_json)
        label_manager = plans_manager.get_label_manager(dataset_json)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)
        deep_supervision = True

        conv_or_blocks_per_stage = {
            'n_conv_per_stage'
            if True else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
            'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
        }
        segmentation_network_class_name = configuration_manager.UNet_class_name
        num_stages = len(configuration_manager.conv_kernel_sizes)
        kwargs = {
            'PlainConvUNet': {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            },
            'ResidualEncoderUNet': {
                'conv_bias': True,
                'norm_op': get_matching_instancenorm(conv_op),
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }

        if self.seg_type == 'lung':
            num_classes = 2
        elif self.seg_type == 'left_right_lung':
            num_classes = 3

        self.model = PlainConvUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                    configuration_manager.unet_max_num_features) for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            prior= self.prior,
            prior_type= self.prior_type,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name]
        )
        self.model.apply(InitWeights_He(1e-2))
        
        #self.model.load_state_dict({i:j for i,j in total_weights['network_weights'].items() if 'decoder.seg_layers' not in i}, 
        #              strict = False)

        lung_weight = torch.load('/workspace/radraid/projects/seg_lung/masked_seg/maskedSeg/results_update/exp_adam_lr0.001_bs1_us64_seed0_DiceCELoss_mask[4, 7, 14, 28]_70_freeze/es_checkpoint.pth.tar')
        self.model.load_state_dict({i:j for i,j in lung_weight.items() if 'decoder.seg_layers' not in i}, 
                      strict = False)
        print("...done")

        if self.freeze:
            print(f"Freezing encoder")
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
        self.model = nn.DataParallel(self.model)
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

    def __init_scheduler(self):
        if self.sche != None:
            print(f"Initiate {self.sche} scheduler", end=" ")
            if self.sche == "cosine":
                sche_param = self.sche_param["cosine"]
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=sche_param["max_epoch"], eta_min=0
                )
            else:
                raise NotImplementedError
            print("...done")
        else:
            self.scheduler = None

class Trainer(baseTrainer):
    def __init__(self, args) -> None:
        super(Trainer, self).__init__(args)
        self.start_epoch = 0
        self.epochs = args.epochs
        #self.ckpt_path = os.path.join(args.exp_dir, "checkpoint.pth.tar")
        self.es_ckpt_path = os.path.join(self.exp_dir, "es_checkpoint.pth.tar")

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> pd.DataFrame:

        # loop through epochs
        for i in range(self.start_epoch, self.epochs):
            self.model.train()
            print(f"------------Epoch {i+1}/{self.epochs}------------")
            # training
            train_loss_sum = 0
            train_metric = []
            train_metric_2 = []
            for batch_idx, (y0_img, y0_seg, y1_img, y1_seg) in enumerate(tqdm(train_loader)):
                #print(y0_img.shape, y0_seg.shape, y1_img.shape, y1_seg.shape)
                y0_img = y0_img.to(self.device)
                y0_seg = y0_seg.to(self.device)
                y1_img = y1_img.to(self.device)
                y1_seg = y1_seg.to(self.device)

                model_input = (y0_img, y0_seg, y1_img)
                
                # run model
                model_out = self.model(model_input)
                model_out_prob = torch.nn.functional.softmax(model_out[0], dim =1)#torch.sigmoid(model_out[0])
                mean_pred = torch.zeros([1,1,224,224,224]).to(self.device)
                mean_pred[model_out_prob[:,[1],:,:,:]>0.6] = 1
                mean_pred[model_out_prob[:,[2],:,:,:]>0.6] = 2
                # compute metric
                if self.seg_type == 'left_right_lung':
                    batch_metric = self.__compute_metric( mean_pred == 1, y1_seg==1)
                    train_metric.extend([batch_metric.cpu().item()])
                    batch_metric_2 = self.__compute_metric( mean_pred == 2, y1_seg==2)
                    train_metric_2.extend([batch_metric_2.cpu().item()])
                #batch_metric = self.__compute_metric( torch.argmax(model_out_prob, dim=1).unsqueeze(0), y1_seg)
                #train_metric.extend([batch_metric.cpu().item()])

                # compute loss
                both = True if batch_idx > 100 else False
                train_loss = self.__compute_loss( model_out[0], y1_seg , both) / self.update_size
                train_loss = train_loss / self.update_size #normalize loss

                # backprop
                train_loss.backward()

                # update weights
                if (batch_idx + 1) % self.update_size == 0 or (batch_idx + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss_sum += train_loss.item()

                # training progess batch and loss
                if (batch_idx + 1) % self.print_every == 0:
                    print(
                        "Batch {} - Train Loss: {:.6f}; Dice left:{:.6f}; Dice right{:.6f}".format(
                            batch_idx, train_loss.item() * self.update_size, batch_metric.cpu().item(), batch_metric_2.cpu().item()
                        )
                    )
                    #plot img and seg
                    for img_idx in [60,120,180]:
                        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
                        ax[0].imshow(y1_img[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[1].imshow(y1_seg[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[2].imshow(torch.argmax(model_out_prob, dim=1).unsqueeze(0).float()[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[3].imshow(model_out_prob[0, 1, :, :, img_idx].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
                        ax[4].imshow(model_out_prob[0, 2, :, :, img_idx].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)



                        #ax[2].imshow((model_out_sigmoid > 0.5).float()[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        #ax[3].imshow(model_out_sigmoid[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
                        
                        #save fig
                        os.makedirs(os.path.join(self.exp_dir, 'figure', f'train_{i}'), exist_ok=True)
                        plt.savefig(os.path.join(self.exp_dir, 'figure', f'train_{i}',f"{batch_idx}_{img_idx}.png"))
                        plt.close()


            # update scheduler
            if self.scheduler:
                self.scheduler.step()
                scheduler_state_dict = self.scheduler.state_dict()
            else:
                scheduler_state_dict = {}

            train_loss_mean = train_loss_sum / len(train_loader)
            train_metric_mean = np.mean(train_metric)
            train_metric_mean_2 = np.mean(train_metric_2)
            print(
                "Epoch {} - Train Loss: {:.6f}; Dice left: {:.6f}; Dice right: {:.6f}".format(
                    i, train_loss_mean, train_metric_mean, train_metric_mean_2
                )
            )

            # log training
            if self.writer:
                self.writer.add_scalar("train/Total_loss", train_loss_mean, i)
                self.writer.add_scalar("train/Dice left", train_metric_mean, i)
                self.writer.add_scalar("train/Dice right", train_metric_mean_2, i)

            # validation
            earlystop = self.__eval(i, val_loader)

            # save model for each epoch
            state = {
                "epoch": i + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                'scheduler': scheduler_state_dict,
                "early_stopping": self.early_stopping,
            }
            self.ckpt_path = os.path.join(self.exp_dir, f"checkpoint_{i}.pth.tar")
            torch.save(state, self.ckpt_path)

            if earlystop:
                break

        print("Finished Training...")
        results = self.predict(val_loader, split = 'val')
        results_test = self.predict(test_loader, split = 'test')

        if self.writer:
            self.writer.close()

        return results, results_test

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader) -> bool:
        val_loss_sum = 0
        val_metric = []
        val_metric_2 = []

        pids = val_loader.dataset.data.pid.values
        priorna = val_loader.dataset.data.y0_seg.isna().values

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (y0_img, y0_seg, y1_img, y1_seg) in enumerate(tqdm(val_loader)):
                y0_img = y0_img.to(self.device)
                y0_seg = y0_seg.to(self.device)
                y1_img = y1_img.to(self.device)
                y1_seg = y1_seg.to(self.device)

                model_input = (y0_img, y0_seg, y1_img)

                # run model
                model_out = self.model(model_input)        

                model_out_prob = torch.nn.functional.softmax(model_out[0], dim =1)#torch.sigmoid(model_out[0])
                mean_pred = torch.zeros([1,1,224,224,224]).to(self.device)
                mean_pred[model_out_prob[:,[1],:,:,:]>0.6] = 1
                mean_pred[model_out_prob[:,[2],:,:,:]>0.6] = 2
                # compute metric
                if self.seg_type == 'left_right_lung':
                    batch_metric = self.__compute_metric( mean_pred == 1, y1_seg==1)
                    val_metric.extend([batch_metric.cpu().item()])
                    batch_metric_2 = self.__compute_metric( mean_pred == 2, y1_seg==2)
                    val_metric_2.extend([batch_metric_2.cpu().item()])
                #model_out_sigmoid = torch.sigmoid(model_out[0])


                # compute metric
                #batch_metric = self.__compute_metric( (model_out_sigmoid > 0.5).float(), y1_seg)
                #val_metric.extend([batch_metric.cpu().item()])

                # compute loss
                val_loss = self.__compute_loss( model_out[0], y1_seg)
                val_loss_sum += val_loss.item()


                if batch_idx % self.print_every == 0:
                    print(
                        "Batch {} - Train Loss: {:.6f}; Dice left:{:.6f}; Dice right{:.6f}".format(
                            batch_idx, val_loss.item(), batch_metric.cpu().item(), batch_metric_2.cpu().item()
                        )
                    )
                    #plot img and seg
                    for img_idx in [50,120,180]:
                        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
                        ax[0].imshow(y1_img[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[1].imshow(y1_seg[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[2].imshow(torch.argmax(model_out_prob, dim=1).unsqueeze(0).float()[0, 0, :, :, img_idx].cpu().detach().numpy(), cmap="gray")
                        ax[3].imshow(model_out_prob[0, 1, :, :, img_idx].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)
                        ax[4].imshow(model_out_prob[0, 2, :, :, img_idx].cpu().detach().numpy(), cmap="gray", vmin=0, vmax=1)


            
                        fig.suptitle(f"pid: {pids[batch_idx]}")

                        #save fig
                        os.makedirs(os.path.join(self.exp_dir, 'figure',f"val_{cur}"), exist_ok=True)
                        plt.savefig(os.path.join(self.exp_dir, 'figure',f"val_{cur}",f"{batch_idx}_{img_idx}.png"))
                        plt.close()


        val_loss_mean = val_loss_sum / len(val_loader)
        val_metric_mean = np.mean(val_metric)
        val_metric_mean_2 = np.mean(val_metric_2)
        print(
            "Epoch {} - Validation Loss : {:.6f}; Dice left: {:.6f} ; Dice right: {:.6f}".format(
                cur, val_loss_mean, val_metric_mean, val_metric_mean_2
            )
        )

        if self.writer:
            self.writer.add_scalar("val/Total_loss", val_loss_mean, cur)
            self.writer.add_scalar("val/Dice left", val_metric_mean, cur)
            self.writer.add_scalar("val/Dice right", val_metric_mean_2, cur)

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

    def predict(self, val_loader: torch.utils.data.DataLoader, split = 'val') -> pd.DataFrame:
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
        val_metric_2 = []
        os.makedirs(os.path.join(self.exp_dir, 'pred_scores', split), exist_ok=True)
        pids = val_loader.dataset.data.pid.values

        with torch.no_grad():
            for batch_idx, (y0_img, y0_seg, y1_img, y1_seg) in enumerate(tqdm(val_loader)):
                y0_img = y0_img.to(self.device)
                y0_seg = y0_seg.to(self.device)
                y1_img = y1_img.to(self.device)
                y1_seg = y1_seg.to(self.device)

                model_input = (y0_img, y0_seg, y1_img)

                # run model
                model_out = self.model(model_input)

                model_out_prob = torch.nn.functional.softmax(model_out[0], dim =1)#torch.sigmoid(model_out[0])
                mean_pred = torch.zeros([1,1,224,224,224]).to(self.device)
                mean_pred[model_out_prob[:,[1],:,:,:]>0.6] = 1
                mean_pred[model_out_prob[:,[2],:,:,:]>0.6] = 2
                # compute metric
                if self.seg_type == 'left_right_lung':
                    batch_metric = self.__compute_metric( mean_pred == 1, y1_seg==1)
                    val_metric.extend([batch_metric.cpu().item()])
                    batch_metric_2 = self.__compute_metric( mean_pred == 2, y1_seg==2)
                    val_metric_2.extend([batch_metric_2.cpu().item()])

                #model_out_sigmoid = torch.sigmoid(model_out[0])

                #save sigmoid into pkl files
                #torch.save(model_out_sigmoid, os.path.join(self.exp_dir, 'pred_scores', split, f"{pids[batch_idx]}.pkl"))

                ## compute metric
                #batch_metric = self.__compute_metric( (model_out_sigmoid > 0.5).float(), y1_seg)
                #val_metric.extend([batch_metric.cpu().item()])

                # compute loss
                val_loss = self.__compute_loss( model_out[0], y1_seg)
                val_loss_sum += val_loss.item()

        val_loss_mean = val_loss_sum / len(val_loader)
        val_metric_mean = np.mean(val_metric)
        val_metric_mean_2 = np.mean(val_metric_2)
        print(
            "Final {} Loss : {:.6f}; Dice left: {:.6f}; Dice right: {:.6f} ".format( split,
                val_loss_mean, val_metric_mean, val_metric_mean_2
            )
        )

        # save results
        results = pd.DataFrame(
            {
                split: [
                    val_loss_mean,
                    val_metric_mean,
                    val_metric_mean_2
                ],
            },
            index=["loss", "dice_left", "dice_right"],
        )
        results = pd.concat([results], axis=0)



        return results

    @staticmethod
    def __compute_metric(model_out, label):
        dice = DiceCoefficient()
        return dice(model_out, label)

    def __compute_loss(self, model_out, label, both = False):

        loss = []
        if 'DiceCELoss' in self.loss:
            dice_ce_loss = self.DiceCE_loss(model_out, label)
            loss.append(dice_ce_loss)
        loss = torch.stack(loss).sum()

        return loss

    def load_prev(self):
        print("Continue training from checkpoint: {}".format(self.ckpt_path))
        if os.path.isfile(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path)
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.early_stopping = ckpt["early_stopping"]
            self.early_stopping.early_stop = False
            if self.sche:
                self.scheduler.load_state_dict(ckpt["scheduler"])
        else:
            raise ValueError("No checkpoint found at '{}'".format(self.ckpt_path))
