
import sys

from termcolor import colored, cprint

from configs.config_v0 import (DataConfig, NetConfig, TestingConfig,
                               TrainingConfig, ValidationConfig)


class ExprSetting(object):
    def __init__(self):

        # self.config_file_path = config_file_path
        # with open(self.config_file_path) as file:
        #     self.config = yaml.safe_load(file)

        # self.config["DOT_ARGS"] = dotdict(self.config["EXTRA_ARGS"]["ARGS"])

        self.model_class = self.dynamic_models()
        self.dataloader_class = self.dynamic_dataloaders()
        self.lr_logger, self.model_checkpoint = self.checkpoint_setting()
        self.early_stop = self.earlystop_setting()

    def checkpoint_setting(self):
        from pytorch_lightning.callbacks import (LearningRateMonitor,
                                                 ModelCheckpoint)

        lr_logger = LearningRateMonitor(logging_interval="epoch")

        model_checkpoint = ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}-{user_metric:.2f}",
            save_last=True,
            save_weights_only=True,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        return lr_logger, model_checkpoint

    def earlystop_setting(self):
        # ## https://www.youtube.com/watch?v=vfB5Ax6ekHo
        from pytorch_lightning.callbacks import EarlyStopping

        early_stop = EarlyStopping(
            monitor="val_loss", patience=20000, strict=False, verbose=False, mode="min"
        )
        return early_stop

    def dynamic_dataloaders(self):

        if DataConfig.dataloader_name == "dataloader_v0":
            from dataloader_v0 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v1":
            from dataloader_v1 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v2":
            from dataloader_v2 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v3":
            from dataloader_v3 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v4":
            from dataloader_v4 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v5":
            from dataloader_v5 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v6":
            from dataloader_v6 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v7":
            from dataloader_v7 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v8":
            from dataloader_v8 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v9":
            from dataloader_v9 import UniDataloader
        elif DataConfig.dataloader_name == "dataloader_v10":
            from dataloader_v10 import UniDataloader
        else:
            sys.eixt("Please check your dataloader name in config file ... ")

        UsedUniDataloader = UniDataloader()
        return UsedUniDataloader

    def dynamic_models(self):
        if NetConfig.model_name == "model_v0":
            from model_v0 import Model
        elif NetConfig.model_name == "model_v1":
            from model_v1 import Model
        elif NetConfig.model_name == "model_v2":
            from model_v2 import Model
        elif NetConfig.model_name == "model_v3":
            from model_v3 import Model
        elif NetConfig.model_name == "model_v4":
            from model_v4 import Model
        elif NetConfig.model_name == "model_v5":
            from model_v5 import Model
        elif NetConfig.model_name == "model_v6":
            from model_v6 import Model
        elif NetConfig.model_name == "model_v7":
            from model_v7 import Model
        elif NetConfig.model_name == "model_v8":
            from model_v8 import Model
        elif NetConfig.model_name == "model_v9":
            from model_v9 import Model
        elif NetConfig.model_name == "model_v10":
            from model_v10 import Model
        else:
            sys.eixt("Please check your model name in config file ... ")

        UniModel = Model
        return UniModel
