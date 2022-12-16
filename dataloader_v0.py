

from configs.config_v0 import (DataConfig, TestingConfig, TrainingConfig,
                               ValidationConfig)


class ExtraDatasetConfig:
    # dataset settings
    dataset_type = "CityscapesDataset"
    data_root = DataConfig.data_root
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )
    crop_size = (512, 1024)
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(type="Resize", img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
        dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
        dict(type="Normalize", **img_norm_cfg),
        dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type="DefaultFormatBundle"),
        # dict(type="ImageToTensor", keys=["img", "gt_semantic_seg"]),
        dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    ]
    # val_pipeline = [
    #     dict(type="LoadImageFromFile"),
    #     dict(type="LoadAnnotations"),
    #     dict(type="Resize", img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    #     dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    #     dict(type="RandomFlip", prob=0.5),
    #     dict(type="PhotoMetricDistortion"),
    #     dict(type="Normalize", **img_norm_cfg),
    #     dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    #     dict(type="DefaultFormatBundle"),
    #     dict(type="Collect", keys=["img", "gt_semantic_seg"]),
    # ]
    val_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(2048, 1024),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Normalize", **img_norm_cfg),
                dict(type="ImageToTensor", keys=["img", "gt_semantic_seg"]),
                dict(type="Collect", keys=["img", "gt_semantic_seg"]),
            ],
        ),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(2048, 1024),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type="Resize", keep_ratio=True),
                dict(type="RandomFlip"),
                dict(type="Normalize", **img_norm_cfg),
                dict(type="ImageToTensor", keys=["img"]),
                dict(type="Collect", keys=["img", "gt_semantic_seg"]),
            ],
        ),
    ]
    dataset_config = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir="leftImg8bit/train",
            ann_dir="gtFine/train",
            pipeline=train_pipeline,
        ),
        val=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir="leftImg8bit/val",
            ann_dir="gtFine/val",
            pipeline=val_pipeline,
        ),
        test=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir="leftImg8bit/val",
            ann_dir="gtFine/val",
            pipeline=test_pipeline,
        ),
    )


class UniDataloader:
    def __init__(self):
        super(UniDataloader, self).__init__()

    def get_dataloader(self, split, num_gpus):
        print("==>> split: ", split)
        from datasets.datasets_builder import build_dataloader, build_dataset

        data_config = {}

        if split == "train":
            data_config = ExtraDatasetConfig.dataset_config["train"]
            shuffle = True
            drop_last = True
            batch_size = TrainingConfig.batch_size
        elif split == "val":
            data_config = ExtraDatasetConfig.dataset_config["val"]
            shuffle = False
            drop_last = False
            batch_size = ValidationConfig.batch_size
        elif split == "test":
            data_config = ExtraDatasetConfig.dataset_config["test"]
            shuffle = False
            drop_last = False
            batch_size = TestingConfig.batch_size
        else:
            print("\n The split is not valid.\n")

        dataset = build_dataset(data_config)
        print("==>> dataset: ", len(dataset))

        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=batch_size,
            workers_per_gpu=DataConfig.workers,
            # num_gpus=num_gpus,
            dist=False,
            shuffle=shuffle,
            seed=None,
            drop_last=drop_last,
            pin_memory=DataConfig.pin_memory,
            persistent_workers=True,
        )
        # dataloader = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     # sampler=sampler,
        #     num_workers=DataConfig.workers,
        #     # collate_fn=partial(collate, samples_per_gpu=batch_size),
        #     collate_fn=collate,
        #     pin_memory=DataConfig.pin_memory,
        #     shuffle=shuffle,
        #     # worker_init_fn=init_fn,
        #     drop_last=drop_last,
        #     persistent_workers=True,
        # )
        return dataloader

    def get_train_dataloader(self, num_gpus=1):

        self.train_loader = self.get_dataloader("train", num_gpus)

        return self.train_loader

    def get_val_dataloader(self, num_gpus=1):

        self.val_loader = self.get_dataloader("val", num_gpus)

        return self.val_loader

    def get_test_dataloader(self, num_gpus=1):

        self.test_loader = self.get_dataloader("test", num_gpus)

        return self.test_loader


if __name__ == "__main__":

    dataloader_class = UniDataloader()
    train_dataloader = dataloader_class.get_train_dataloader(num_gpus=4)
    """>>>
    print("train_dataloder: {}".format(len(train_dataloader)))
    # train_dataloder: 185
    """

    val_dataloader = dataloader_class.get_val_dataloader(num_gpus=4)
    """>>>
    print("val_dataloader: {}".format(len(val_dataloader)))
    # val_dataloader: 32
    """

    test_dataloader = dataloader_class.get_test_dataloader(num_gpus=4)
    """>>>
    print("test_dataloader: {}".format(len(test_dataloader)))
    # test_dataloader: 32
    """

    print("\n Checking Train Dataloader \n ")
    for batch_idx, data_dict in enumerate(train_dataloader):
        """>>>
        print("==>> data_dict: ", data_dict.keys())
        # ==>> data_dict:  dict_keys(['img_metas', 'img', 'gt_semantic_seg'])
        """

        image, mask = data_dict["img"]._data[0], data_dict["gt_semantic_seg"]._data[0]

        """>>>
        
        """
        print("==>> mask.shape: ", mask.shape)
        print("==>> image.shape: ", image.shape)
        # ==>> mask.shape:  torch.Size([8, 1, 512, 1024])
        # ==>> image.shape:  torch.Size([8, 3, 512, 1024])

        # break

    print("\n Checking Val Dataloader \n")
    for batch_idx, data_dict in enumerate(val_dataloader):

        """>>>
        print("==>> data_dict: ", data_dict.keys())
        # ==>> data_dict:  dict_keys(['img_metas', 'img', 'gt_semantic_seg'])
        """

        image, mask = data_dict["img"][0], data_dict["gt_semantic_seg"][0]
        """>>>
        
        """
        print("==>> mask.shape: ", mask.shape)
        print("==>> image.shape: ", image.shape)
        # print("==>> image: ", image)
        # ==>> mask.shape:  torch.Size([8, 1, 1024, 2048])
        # ==>> image.shape:  torch.Size([8, 3, 1024, 2048])

        # import sys

        # sys.exit()
