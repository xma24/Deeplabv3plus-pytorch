import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.onnx
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from rich import print

from expr_setting import ExprSetting

cudnn.benchmark = True


matplotlib.use("Agg")

plt.style.use("ggplot")

warnings.filterwarnings("ignore")

from configs.config_v0 import (
    DataConfig,
    DeeplabV3PlusConfig,
    NetConfig,
    TestingConfig,
    TrainingConfig,
    ValidationConfig,
)

if __name__ == "__main__":

    # train_args = parse_args()
    expr_setting = ExprSetting()

    lr_logger, model_checkpoint, early_stop, model_class, dataloader_class = (
        expr_setting.lr_logger,
        expr_setting.model_checkpoint,
        expr_setting.early_stop,
        expr_setting.model_class,
        expr_setting.dataloader_class,
    )

    os.makedirs(DataConfig.work_dirs, exist_ok=True)

    seed_everything(DataConfig.random_seed)

    if DataConfig.classes == "none":
        DataConfig.classes = np.arange(DataConfig.num_classes)

    if isinstance(TrainingConfig.num_gpus, int):
        num_gpus = TrainingConfig.num_gpus
    elif TrainingConfig.num_gpus == "autocount":
        TrainingConfig.num_gpus = torch.cuda.device_count()
        num_gpus = TrainingConfig.num_gpus
    else:
        gpu_list = TrainingConfig.num_gpus.split(",")
        num_gpus = len(gpu_list)

    if TrainingConfig.logger_name == "neptune":
        print("Not implemented")
        exit(0)
    elif TrainingConfig.logger_name == "csv":
        own_logger = CSVLogger(DataConfig.logger_root)
    elif TrainingConfig.logger_name == "wandb":
        own_logger = WandbLogger(
            project=TrainingConfig.wandb_name, settings=wandb.Settings(code_dir=".")
        )
    else:
        own_logger = CSVLogger(DataConfig.logger_root)

    print("num of gpus: {}".format(num_gpus))

    model = model_class(TrainingConfig.batch_size, NetConfig.lr, own_logger)

    if TrainingConfig.pretrained_weights:
        checkpoint = torch.load(TrainingConfig.pretrained_weights_path)
        print("init keys of checkpoint:{}".format(checkpoint.keys()))

        state_dict = checkpoint["state_dict"]
        print("init keys :{}".format(state_dict.keys()))

        updated_state_dict = {}
        for k in list(state_dict.keys()):
            updated_state_dict["model." + k] = state_dict[k]
        print("==>> updated_state_dict: ", updated_state_dict.keys())

        model.load_state_dict(state_dict, strict=True)

        TrainingConfig.max_epochs = TrainingConfig.pretrained_weights_max_epoch

        print("\n Using pretrained weights ... \n")

    if DeeplabV3PlusConfig.pretrained_cls_weights:
        state_dict = torch.load(DeeplabV3PlusConfig.pretrained_cls_weights_path)[
            "state_dict"
        ]

        backbone_state_dict = {}
        fc_state_dict = {}
        for k in list(state_dict.keys()):
            if k.startswith("fc."):
                fc_state_dict[k[len("fc.") :]] = state_dict[k]
            else:
                backbone_state_dict[k] = state_dict[k]
            del state_dict[k]
        model.backbone.load_state_dict(backbone_state_dict, strict=True)

        print("\n Using pretrained CLS BACKBONE weights ... \n")

    """>>>
        - The setting of pytorch lightning Trainer:
            (https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/trainer/trainer.py)
    """
    if TrainingConfig.cpus:
        print("using CPUs to do experiments ... ")
        trainer = pl.Trainer(
            num_nodes=TrainingConfig.num_nodes,
            # precision=config["TRAIN"]["PRECISION"],
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            profiler="pytorch",
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            replace_sampler_ddp=False,
        )
    elif TrainingConfig.lr_find:
        print("using GPUs and lr_find to do experiments ... ")
        trainer = pl.Trainer(
            # devices=TrainingConfig.num_gpus,
            devices=1,
            # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            # accelerator="gpu",
            # strategy=DDPStrategy(find_unused_parameters=True),
            # accelerator=TrainingConfig.accelerator,
            # strategy=DDPStrategy(find_unused_parameters=True),
            # (strategy="ddp", accelerator="gpu", devices=4);(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4);
            # (strategy="ddp_spawn", accelerator="auto", devices=4); (strategy="deepspeed", accelerator="gpu", devices="auto"); (strategy="ddp", accelerator="cpu", devices=3);
            # (strategy="ddp_spawn", accelerator="tpu", devices=8); (accelerator="ipu", devices=8);
            # strategy="ddp_spawn",
            # accelerator="auto",
            # profiler="pytorch",  # "simple", "advanced","pytorch"
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            max_epochs=TrainingConfig.max_epochs,
            # resume_from_checkpoint=config["TRAIN"]["CKPT_PATH"],
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            # replace_sampler_ddp=False,
            auto_lr_find=True,
        )
    elif (not TrainingConfig.lr_find) and TrainingConfig.pl_resume:
        print(
            "using GPUs to do experiments; Not using lr_find; Using resume setting... "
        )
        trainer = pl.Trainer(
            devices=TrainingConfig.num_gpus,
            # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            accelerator=TrainingConfig.accelerator,
            strategy=DDPStrategy(find_unused_parameters=True),
            # (strategy="ddp", accelerator="gpu", devices=4);(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4);
            # (strategy="ddp_spawn", accelerator="auto", devices=4); (strategy="deepspeed", accelerator="gpu", devices="auto"); (strategy="ddp", accelerator="cpu", devices=3);
            # (strategy="ddp_spawn", accelerator="tpu", devices=8); (accelerator="ipu", devices=8);
            # profiler="pytorch",  # "simple", "advanced","pytorch"
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            max_epochs=TrainingConfig.pl_resume_max_epoch,
            resume_from_checkpoint=TrainingConfig.pl_resume_path,
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            # replace_sampler_ddp=False,
        )
    else:
        print("using GPUs to do experiments; Not using lr_find; ")

        trainer = pl.Trainer(
            devices=TrainingConfig.num_gpus,
            # gpus=torch.cuda.device_count(),  ### let the code to detect the number of gpus to use
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            accelerator=TrainingConfig.accelerator,
            strategy=DDPStrategy(find_unused_parameters=True),
            # (strategy="ddp", accelerator="gpu", devices=4);(strategy=DDPStrategy(find_unused_parameters=False), accelerator="gpu", devices=4);
            # (strategy="ddp_spawn", accelerator="auto", devices=4); (strategy="deepspeed", accelerator="gpu", devices="auto"); (strategy="ddp", accelerator="cpu", devices=3);
            # (strategy="ddp_spawn", accelerator="tpu", devices=8); (accelerator="ipu", devices=8);
            # profiler="pytorch",  # "simple", "advanced","pytorch"
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint, early_stop],
            log_every_n_steps=1,
            # track_grad_norm=1,
            progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
            max_epochs=TrainingConfig.max_epochs,
            # resume_from_checkpoint="/home/xma24/vscode/openmmlab_projects/openmmlab_projects/DeepLabV3Plus-Pytorch-1205-2022-V5/pytorch_segmentation/36z9a2cl/checkpoints/epoch=90-val_loss=0.14-user_metric=0.73.ckpt",
            # sync_batchnorm=True if num_gpus > 1 else False,
            # plugins=DDPPlugin(find_unused_parameters=False),
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            # replace_sampler_ddp=False,
        )

    trainer.fit(model)
