
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, StepLR)

from configs.config_v0 import (DataConfig, NetConfig, TestingConfig,
                               TrainingConfig, ValidationConfig)
from model_v0_utils import (get_deeplabv3plus_backbone,
                            get_deeplabv3plus_heads, get_deeplabv3plus_model,
                            get_segmentation_metrics)
from utils import console_logger_start


class Model(pl.LightningModule):
    def __init__(self, batch_size, lr, logger=None):
        super(Model, self).__init__()

        self.n_logger = logger
        self.console_logger = console_logger_start()

        self.current_iter = 1
        self.total_iters = TrainingConfig.batch_size * TrainingConfig.max_epochs

        self.learning_rate = lr
        self.batch_size = batch_size
        # self.save_hyperparameters()

        if TrainingConfig.ckpt_path == "none":
            print("No Checkpoint is detecte, Do training ... ")

        self.criterion_seg = nn.CrossEntropyLoss(ignore_index=DataConfig.num_classes)
        self.ignore = True
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
        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        aux_x = []
        aux_x.append(x)
        aux = self.auxiliary_head(aux_x)

        x = self.backbone.layer4(x)

        out_x = []
        out_x.append(x)
        out = self.decode_head(out_x)

        return out, aux

    # def train_dataloader(self, )

    def training_step(self, batch, batch_idx):
        self.current_iter += 1

        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups
        for param_group_idx in range(len(param_groups)):

            sub_param_group = param_groups[param_group_idx]
            """>>>
            print("==>> sub_param_group: ", sub_param_group.keys())
            # ==>> sub_param_group:  dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'initial_lr'])
            """

            sub_lr_name = "lr_" + str(param_group_idx)
            """>>>
            print("lr: {}, {}".format(sub_lr_name, sub_param_group["lr"]))
            # lr: lr_0, 0.001
            # lr: lr_1, 0.08
            """

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

        self.log(
            "train_iou_epoch",
            self.train_iou.compute(),
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
        images, masks = (
            data_dict["img"][0].to(self.device),
            data_dict["gt_semantic_seg"][0].long().to(self.device),
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

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            batch_size=ValidationConfig.batch_size,
        )

        self.val_iou.update(model_predictions, masks)
        self.val_precision.update(model_predictions, masks)
        self.val_recall.update(model_predictions, masks)

        return val_loss

    def validation_epoch_end(self, outputs):

        val_epoch_iou = self.val_iou.compute()
        val_epoch_precision = self.val_precision.compute()
        val_epoch_recall = self.val_recall.compute()

        self.log(
            "val_iou_epoch",
            val_epoch_iou,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=ValidationConfig.batch_size,
        )

        val_epoch_iou_mean = torch.mean(val_epoch_iou).item()
        user_metric = val_epoch_iou_mean
        self.log(
            "user_metric",
            user_metric,
            on_step=False,
            on_epoch=True,
            batch_size=ValidationConfig.batch_size,
        )

        if self.ignore:
            val_epoch_precision_mean = torch.mean(val_epoch_precision[:-1]).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall[:-1]).item()
        else:
            val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall).item()

        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            for i in range(val_epoch_iou.shape[0]):
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
        if TestingConfig.multiscale:
            image_mask_list, labels, paths = batch

            for i in range(len(image_mask_list)):
                images, masks = image_mask_list[i]

                if self.ignore:
                    masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
                        DataConfig.num_classes
                    ).to(self.device)

                losses_dict = self.forward(images, masks)
                test_loss = losses_dict["loss_ce"]

                model_output = losses_dict["outputs_seg"]

                model_output = F.interpolate(
                    model_output,
                    [masks.shape[-2], masks.shape[-1]],
                    mode="bilinear",
                    align_corners=False,
                )

                # model_output = F.softmax(model_output, dim=1)

                model_predictions = model_output.argmax(dim=1)

                # unique_prediction = torch.unique(model_predictions)
                # print("==>> unique_prediction: ", unique_prediction)

                test_loss_tm = self.criterion_seg(model_output, masks)

                self.test_iou.update(model_predictions, masks)
                self.test_precision.update(model_predictions, masks)
                self.test_recall.update(model_predictions, masks)

            self.log(
                "test_loss",
                test_loss,
                on_step=True,
                on_epoch=True,
                batch_size=TestingConfig.batch_size,
            )
            self.log(
                "test_loss_tm",
                test_loss_tm,
                on_step=True,
                on_epoch=True,
                batch_size=TestingConfig.batch_size,
            )

            return test_loss

        else:
            # images, masks, labels, paths = batch
            data_dict = batch

            images, masks = (
                data_dict["img"][0].to(self.device),
                data_dict["gt_semantic_seg"][0].long().to(self.device),
            )

            if len(masks.shape) == 4:
                masks = masks.squeeze(1)

            # print("==>> masks.shape: ", masks.shape)
            # unique_mask = torch.unique(masks)
            # print("==>> unique_mask: ", unique_mask)

            # mask_min = torch.min(masks)
            # print("==>> mask_min: ", mask_min)
            if self.ignore:
                masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
                    DataConfig.num_classes
                ).to(self.device)

            model_output, _ = self.forward(images, masks)
            # test_loss = losses_dict["loss_ce"]

            # model_output = losses_dict["outputs_seg"]

            model_output = F.interpolate(
                model_output,
                [masks.shape[-2], masks.shape[-1]],
                mode="bilinear",
                align_corners=False,
            )

            # model_output = F.softmax(model_output, dim=1)

            model_predictions = model_output.argmax(dim=1)

            # unique_prediction = torch.unique(model_predictions)
            # print("==>> unique_prediction: ", unique_prediction)

            test_loss_tm = self.criterion_seg(model_output, masks)

            test_loss = test_loss_tm

            self.test_iou.update(model_predictions, masks)
            self.test_precision.update(model_predictions, masks)
            self.test_recall.update(model_predictions, masks)

            self.log(
                "test_loss",
                test_loss,
                on_step=True,
                on_epoch=True,
                batch_size=TestingConfig.batch_size,
            )
            self.log(
                "test_loss_tm",
                test_loss_tm,
                on_step=True,
                on_epoch=True,
                batch_size=TestingConfig.batch_size,
            )

            return test_loss

    def test_epoch_end(self, outputs):

        test_loss = torch.tensor(outputs).mean()

        self.log(
            "test_iou_epoch",
            self.test_iou.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=TestingConfig.batch_size,
        )

        test_epoch_iou = self.test_iou.compute()
        test_epoch_precision = self.test_precision.compute()
        test_epoch_recall = self.test_recall.compute()

        test_epoch_iou_mean = torch.mean(test_epoch_iou).item()
        user_metric = test_epoch_iou_mean
        self.log(
            "user_metric",
            user_metric,
            on_step=False,
            on_epoch=True,
            batch_size=TestingConfig.batch_size,
        )

        if self.ignore:
            test_epoch_precision_mean = torch.mean(test_epoch_precision[:-1]).item()
            test_epoch_recall_mean = torch.mean(test_epoch_recall[:-1]).item()
        else:
            test_epoch_precision_mean = torch.mean(test_epoch_precision).item()
            test_epoch_recall_mean = torch.mean(test_epoch_recall).item()

        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))
            for i in range(test_epoch_iou.shape[0]):
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
            self.console_logger.info("test_loss: {0:.4f} ".format(test_loss))

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

        if TrainingConfig.scheduler == "cosineAnn":

            eta_min = 1.0e-6
            T_max = TrainingConfig.max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "cosineAnnWarm":
            sch = CosineAnnealingWarmRestarts(
                optimizer, T_0=TrainingConfig.T_0, T_mult=TrainingConfig.T_mult
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
            sch = StepLR(optimizer, step_size=10, gamma=0.1)

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
            lr = NetConfig.lr
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
                        "lr": NetConfig.backbone_lr,
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
