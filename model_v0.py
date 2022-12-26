import copy
import math
import os
import pickle
import shutil
import time
from collections import OrderedDict
from re import S

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from rich import print
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

from configs.config_v0 import (
    DataConfig,
    NetConfig,
    TestingConfig,
    TrainingConfig,
    ValidationConfig,
)
from dataloader_v0 import UniDataloader
from model_v0_utils import (
    get_deeplabv3plus_backbone,
    get_deeplabv3plus_heads,
    get_deeplabv3plus_model,
    get_segmentation_metrics,
)
from utils import console_logger_start


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay=1
    ):
        super().__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        self.decay = decay
        self.initial_lrs = self.base_lrs

    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError(
                    "Expected non-negative epoch, but got {}".format(epoch)
                )
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
            else:
                n = 0

            self.base_lrs = [
                initial_lrs * (self.decay ** n) for initial_lrs in self.initial_lrs
            ]

        super().step(epoch)


class Model(pl.LightningModule):
    def __init__(self, batch_size, lr, logger=None):
        super(Model, self).__init__()

        self.n_logger = logger
        self.console_logger = console_logger_start()

        self.current_iter = 1
        self.total_iters = TrainingConfig.batch_size * TrainingConfig.max_epochs

        self.learning_rate = lr
        self.batch_size = batch_size
        self.UsedUniDataloader = UniDataloader()
        # self.save_hyperparameters()
        self.val_scales = 1
        self.test_scales = 1

        if TrainingConfig.ckpt_path == "none":
            print("No Checkpoint is detecte, Do training ... ")

        self.ignore = DataConfig.class_ignore

        if self.ignore:
            self.criterion_seg = nn.CrossEntropyLoss(
                ignore_index=DataConfig.num_classes
            )
        else:
            self.criterion_seg = nn.CrossEntropyLoss()

        self.norm_cfg = dict(type="SyncBN", requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

        (
            self.train_iou,
            self.val_iou,
            self.test_iou,
            self.train_precision,
            self.val_precision,
            self.test_precision,
            self.val_recall,
            self.test_recall,
        ) = get_segmentation_metrics(DataConfig.num_classes, self.ignore)

        self.model = get_deeplabv3plus_model()
        self.backbone = get_deeplabv3plus_backbone(self.model)
        # print("==>> self.backbone: ", self.backbone)
        self.decode_head, self.auxiliary_head = get_deeplabv3plus_heads(self.model)

        del self.model

    def forward(self, images, labels=None, epoch=None, batch_idx=None):

        x = self.backbone.stem(images)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        feats_1 = x

        x = self.backbone.layer2(x)
        feats_2 = x

        x = self.backbone.layer3(x)
        feats_3 = x

        aux_x = []
        aux_x.append(x)
        aux = self.auxiliary_head(aux_x)

        x = self.backbone.layer4(x)
        feats_4 = x

        out = self.decode_head([feats_1, feats_2, feats_3, feats_4])

        return out, aux

    def train_dataloader(self):

        return self.UsedUniDataloader.get_train_dataloader()

    def val_dataloader(self):

        return self.UsedUniDataloader.get_val_dataloader()

    def training_step(self, batch, batch_idx):
        self.current_iter += 1

        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups
        for param_group_idx in range(len(param_groups)):

            sub_param_group = param_groups[param_group_idx]

            sub_lr_name = "lr_" + str(param_group_idx)

            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=TrainingConfig.batch_size,
            )

        self.input_batch_time = time.time()

        data_dict = batch
        images, masks = (
            data_dict["img"]._data[0].to(self.device),
            data_dict["gt_semantic_seg"]._data[0].to(self.device),
        )

        if len(masks.shape) == 4:
            masks = masks.squeeze(1)

        if self.ignore:
            masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
                DataConfig.num_classes
            ).to(self.device)

        model_output, model_aux = self.forward(images, masks)
        model_output = F.interpolate(
            model_output,
            [masks.shape[-2], masks.shape[-1]],
            mode="bilinear",
            align_corners=False,
        )
        model_aux = F.interpolate(
            model_aux,
            [masks.shape[-2], masks.shape[-1]],
            mode="bilinear",
            align_corners=False,
        )
        model_predictions_out = model_output.argmax(dim=1)
        train_loss_out = self.criterion_seg(model_output, masks)
        train_loss_aux = self.criterion_seg(model_aux, masks)

        train_loss = train_loss_out + 0.4 * train_loss_aux

        losses = {"loss": train_loss}

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=TrainingConfig.batch_size,
        )

        self.log(
            "train_loss_out",
            train_loss_out,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=TrainingConfig.batch_size,
        )

        self.log(
            "train_loss_aux",
            train_loss_aux,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=TrainingConfig.batch_size,
        )

        self.train_iou.update(model_predictions_out, masks)

        self.batch_training_time = time.time() - self.input_batch_time

        if batch_idx % 10 == 0:
            self.console_logger.info(
                "epoch: {0:04d} | loss_train: {1:.4f} | b_time: {2:.4f}".format(
                    self.current_epoch, losses["loss"], self.batch_training_time
                )
            )

        return {"loss": losses["loss"]}

    def training_epoch_end(self, outputs):

        cwd = os.getcwd()
        print("==>> Expriment Folder: ", cwd)

        train_iou = self.train_iou.compute()

        if self.ignore:
            train_iou_mean = torch.mean(train_iou[:-1]).item()
        else:
            train_iou_mean = torch.mean(train_iou).item()

        self.log(
            "train_iou_epoch",
            train_iou_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=TrainingConfig.batch_size,
        )

        self.train_iou.reset()
        self.train_precision.reset()

    def validation_step(self, batch, batch_idx):

        data_dict = batch
        self.val_scales = len(data_dict["img"])
        val_loss_ms = 0
        for scale_idx in range(self.val_scales):
            images, masks = (
                data_dict["img"][scale_idx].to(self.device),
                data_dict["gt_semantic_seg"][scale_idx].long().to(self.device),
            )
            if len(masks.shape) == 4:
                masks = masks.squeeze(1)
            if self.ignore:
                masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
                    DataConfig.num_classes
                ).to(self.device)

            model_output, _ = self.forward(images, masks)
            model_output = F.interpolate(
                model_output,
                [masks.shape[-2], masks.shape[-1]],
                mode="bilinear",
                align_corners=False,
            )
            model_predictions = model_output.argmax(dim=1)
            val_loss_out = self.criterion_seg(model_output, masks)
            val_loss = val_loss_out

            val_loss_ms += val_loss

            self.val_iou.update(model_predictions, masks)
            self.val_precision.update(model_predictions, masks)
            self.val_recall.update(model_predictions, masks)

        self.log(
            "val_loss",
            val_loss_ms,
            on_step=True,
            on_epoch=True,
            batch_size=ValidationConfig.batch_size * self.val_scales,
        )

        return val_loss

    def validation_epoch_end(self, outputs):

        val_epoch_iou = self.val_iou.compute()
        val_epoch_precision = self.val_precision.compute()
        val_epoch_recall = self.val_recall.compute()

        if self.ignore:
            val_epoch_precision_mean = torch.mean(val_epoch_precision[:-1]).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall[:-1]).item()
            val_epoch_iou_mean = torch.mean(val_epoch_iou[:-1]).item()
        else:
            val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall).item()
            val_epoch_iou_mean = torch.mean(val_epoch_iou).item()

        self.log(
            "val_iou_epoch",
            val_epoch_iou_mean,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=ValidationConfig.batch_size * self.val_scales,
        )

        user_metric = val_epoch_iou_mean
        self.log(
            "user_metric",
            user_metric,
            on_step=False,
            on_epoch=True,
            batch_size=ValidationConfig.batch_size * self.val_scales,
        )

        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            # assert (
            #     DataConfig.num_classes == val_epoch_iou.shape[0]
            # ), "Something is wrong."

            for i in range(DataConfig.num_classes):
                self.console_logger.info(
                    "{0: <15}, iou: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        DataConfig.classes[i],
                        val_epoch_iou[i].item(),
                        val_epoch_precision[i].item(),
                        val_epoch_recall[i].item(),
                    )
                )
            self.console_logger.info("iou_mean: {0:.4f} ".format(val_epoch_iou_mean))

            self.console_logger.info(
                "precision_mean: {0:.4f} ".format(val_epoch_precision_mean)
            )
            self.console_logger.info(
                "recall_mean: {0:.4f} ".format(val_epoch_recall_mean)
            )

        self.val_iou.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        data_dict = batch
        self.test_scales = len(data_dict["img"])
        test_loss_ms = 0
        for scale_idx in range(self.test_scales):
            images, masks = (
                data_dict["img"][scale_idx].to(self.device),
                data_dict["gt_semantic_seg"][scale_idx].long().to(self.device),
            )
            if len(masks.shape) == 4:
                masks = masks.squeeze(1)
            if self.ignore:
                masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
                    DataConfig.num_classes
                ).to(self.device)

            model_output, _ = self.forward(images, masks)
            model_output = F.interpolate(
                model_output,
                [masks.shape[-2], masks.shape[-1]],
                mode="bilinear",
                align_corners=False,
            )
            model_predictions = model_output.argmax(dim=1)
            test_loss_out = self.criterion_seg(model_output, masks)
            test_loss = test_loss_out

            test_loss_ms += test_loss

            self.test_iou.update(model_predictions, masks)
            self.test_precision.update(model_predictions, masks)
            self.test_recall.update(model_predictions, masks)

        self.log(
            "test_loss",
            test_loss_ms,
            on_step=True,
            on_epoch=True,
            batch_size=ValidationConfig.batch_size * self.test_scales,
        )

    def test_epoch_end(self, outputs):

        test_epoch_iou = self.test_iou.compute()
        test_epoch_precision = self.test_precision.compute()
        test_epoch_recall = self.test_recall.compute()

        if self.ignore:
            test_epoch_precision_mean = torch.mean(test_epoch_precision[:-1]).item()
            test_epoch_recall_mean = torch.mean(test_epoch_recall[:-1]).item()
            test_epoch_iou_mean = torch.mean(test_epoch_iou[:-1]).item()
        else:
            test_epoch_precision_mean = torch.mean(test_epoch_precision).item()
            test_epoch_recall_mean = torch.mean(test_epoch_recall).item()
            test_epoch_iou_mean = torch.mean(test_epoch_iou).item()

        self.log(
            "test_iou_epoch",
            test_epoch_iou_mean,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=ValidationConfig.batch_size * self.test_scales,
        )

        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            # assert (
            #     DataConfig.num_classes == test_epoch_iou.shape[0]
            # ), "Something is wrong."

            for i in range(DataConfig.num_classes):
                self.console_logger.info(
                    "{0: <15}, iou: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        DataConfig.classes[i],
                        test_epoch_iou[i].item(),
                        test_epoch_precision[i].item(),
                        test_epoch_recall[i].item(),
                    )
                )
            self.console_logger.info("iou_mean: {0:.4f} ".format(test_epoch_iou_mean))

            self.console_logger.info(
                "precision_mean: {0:.4f} ".format(test_epoch_precision_mean)
            )
            self.console_logger.info(
                "recall_mean: {0:.4f} ".format(test_epoch_recall_mean)
            )

        self.test_iou.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def poly_lr_scheduler(
        self, optimizer, init_lr, iter, lr_decay_iter=1, max_iter=1000, power=0.9
    ):
        """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr * (1 - iter / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def configure_optimizers(self):

        optimizer = self.get_optim()

        if TrainingConfig.pl_resume:
            max_epochs = TrainingConfig.pl_resume_max_epoch
        elif TrainingConfig.pretrained_weights:
            max_epochs = TrainingConfig.pretrained_weights_max_epoch
        else:
            max_epochs = TrainingConfig.max_epochs

        if TrainingConfig.scheduler == "cosineAnn":
            eta_min = TrainingConfig.eta_min
            T_max = max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "cosineAnnWarm":

            last_epoch = -1
            eta_min = TrainingConfig.eta_min

            sch = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=TrainingConfig.T_0,
                T_mult=TrainingConfig.T_mult,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "cosineAnnWarmDecay":
            last_epoch = -1
            eta_min = TrainingConfig.eta_min
            decay = TrainingConfig.decay

            sch = CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=TrainingConfig.T_0,
                T_mult=TrainingConfig.T_mult,
                eta_min=eta_min,
                last_epoch=last_epoch,
                decay=decay,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }

        elif TrainingConfig.scheduler == "CosineAnnealingLR":
            steps = 10
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "step":

            step_size = int(TrainingConfig.step_ratio * max_epochs)
            gamma = TrainingConfig.gamma
            sch = StepLR(optimizer, step_size=step_size, gamma=gamma)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "none":
            return optimizer

    def get_optim(self):

        if not hasattr(torch.optim, NetConfig.opt):
            print("Optimiser {} not supported".format(NetConfig.opt))
            raise NotImplementedError

        optim = getattr(torch.optim, NetConfig.opt)

        if NetConfig.opt == "Adam":
            lr = NetConfig.lr
            betas = (0.9, 0.999)
            weight_decay = 0

            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        elif NetConfig.opt == "Lamb":
            lr = NetConfig.lr
            weight_decay = 0.02
            betas = (0.9, 0.999)

            optimizer = torch.optim.Lamb(
                self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
            )
        elif NetConfig.opt == "AdamW":
            lr = NetConfig.lr
            eps = 1e-8
            betas = (0.9, 0.999)
            weight_decay = 0.05

            optimizer = torch.optim.AdamW(
                [{"params": self.parameters()}],
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        elif NetConfig.opt == "SGD":

            if TrainingConfig.pl_resume:
                lr = TrainingConfig.pl_resume_lr
                backbone_lr = TrainingConfig.pl_resume_backbone_lr
            elif TrainingConfig.pretrained_weights:
                lr = TrainingConfig.pre_lr
                backbone_lr = TrainingConfig.pre_backbone_lr
            else:
                lr = NetConfig.lr
                backbone_lr = NetConfig.backbone_lr
            momentum = 0.9
            weight_decay = 0.0001

            if TrainingConfig.single_lr:
                print("Using a single learning rate for all parameters")
                optimizer = torch.optim.SGD(
                    [{"params": self.parameters()}],
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            else:
                print("Using different learning rates for all parameters")

                params = list(self.named_parameters())

                def is_backbone(n):
                    return "backbone" in n

                grouped_parameters = [
                    {
                        "params": [p for n, p in params if is_backbone(n)],
                        "lr": backbone_lr,
                    },
                    {"params": [p for n, p in params if not is_backbone(n)], "lr": lr},
                ]

                optimizer = torch.optim.SGD(
                    grouped_parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )

        else:
            optimizer = optim(self.parameters(), lr=NetConfig.lr)

        optimizer.zero_grad()

        return optimizer
