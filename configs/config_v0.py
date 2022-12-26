class DataConfig:
    data_root = "/data/SSD1/data/cityscapes-xin/"
    logger_root = "/data/SSD1/results/detection/main_csv_logs/"
    work_dirs = "/data/SSD1/results/semantic_segmentation/work_dirs/"
    fix_train_data = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # resize_size = [512, 1024]
    # crop_size = [512, 1024]

    workers = 16
    pin_memory = True
    random_seed = 42

    num_classes = 19
    class_ignore = True
    cls_names = "none"
    classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "ambiguous",
    ]

    class_idx = {
        "road": 0,
        "sidewalk": 1,
        "building": 2,
        "wall": 3,
        "fence": 4,
        "pole": 5,
        "traffic light": 6,
        "traffic sign": 7,
        "vegetation": 8,
        "terrain": 9,
        "sky": 10,
        "person": 11,
        "rider": 12,
        "car": 13,
        "truck": 14,
        "bus": 15,
        "train-plant": 16,
        "motorcycle": 17,
        "bicycle": 18,
        "ambiguous": 255,
    }
    dataset_name = "cityscapes"
    dataloader_name = "dataloader_v0"
    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    extra_args = {}


class NetConfig:
    model_name = "model_v0"
    backbone_name = "resnet101"
    model_real_name = "deeplabv3plus"
    lr = 0.08
    backbone_lr = 0.08
    opt = "SGD"  # AdamW, Adam, SGD
    # WEIGHT_DECAY = 0.0005
    # BETA = 0.5
    # MOMENTUM = 0.9
    # EPS = 0.00000001 # 1e-8
    # AMSGRAD = False0

    extra_args = {}


class TrainingConfig:
    interpolation: False
    logger_name = "wandb"  # "neptune", "csv", "wandb"
    cpus = False
    num_gpus = "autocount"
    num_nodes = 1
    max_epochs = 100
    wandb_name = (
        "pt-"
        + NetConfig.model_real_name
        + "-"
        + DataConfig.dataset_name
        + "-"
        + NetConfig.backbone_name
    )
    ckpt_path = "none"
    onnx_model = "./work_dirs/default.onnx"
    resume = "none"
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1

    batch_size = 8

    subtrain = False
    subtrain_ratio = 1
    precision = 16

    use_torchhub = False
    use_timm = False

    single_lr = True

    lr_find = False

    pl_resume = False
    pl_resume_lr = 0.008
    pl_resume_backbone_lr = 0.008
    pl_resume_max_epoch = 10
    pl_resume_path = ""

    pretrained_weights = False
    pre_backbone_lr = 0.08
    pre_lr = 0.08
    pretrained_weights_max_epoch = 100
    pretrained_weights_path = ""

    scheduler = (
        "cosineAnnWarm"
    )  # "step", "cosineAnnWarm", "poly", "cosineAnn", "cosineAnnWarmDecay"
    T_max = 100  # for cosineAnn; The same with max epoch
    eta_min = 1e-5  # for cosineAnn
    T_0 = 20  # for cosineAnnWarm; cosineAnnWarmDecay
    T_mult = 4  # for cosineAnnWarm; cosineAnnWarmDecay
    decay = 0.5  # for cosineAnnWarmDecay

    step_ratio = 0.3  # for StepLR
    gamma = 0.1  # for StepLR
    poly_lr: False
    extra_args = {}


class ValidationConfig:
    batch_size = 1
    val_interval = 1

    sub_val = False
    subval_ratio = 1
    extra_args = {}


class TestingConfig:
    batch_size = 1
    ckpt_path = "none"
    multiscale = False
    imageration = [1.0]
    slidingscale = False
    extra_args = {}


class DeeplabV3PlusConfig:
    """>>> C: ResNet50"""

    # pretrained = "open-mmlab://resnet50_v1c"
    # depth = 50
    # decode_head_c1_in_channels = 256
    # decode_head_c1_channels = 48
    # decode_head_in_channels = 2048
    # decode_head_channels = 512
    # auxiliary_head_in_channels = 1024
    # auxiliary_head_channels = 256

    """>>> C: ResNet101; training batch size < 16; """
    backbone_name = "resnet101"
    assert NetConfig.backbone_name == backbone_name, "Backbone name does not match."

    depth = 101
    decode_head_c1_in_channels = 256
    decode_head_c1_channels = 48
    decode_head_in_channels = 2048
    decode_head_channels = 512
    auxiliary_head_in_channels = 1024
    auxiliary_head_channels = 256

    pretrained_cls_weights = True
    pretrained_cls_weights_path = "/data/SSD1/data/weights/resnet101_v1c-e67eebb6.pth"

    """>>> C: ResNet18"""
    # backbone_name = "resnet18"
    # # pretrained = "open-mmlab://resnet18_v1c"
    # depth = 18
    # decode_head_c1_in_channels = 64
    # decode_head_c1_channels = 12
    # decode_head_in_channels = 512
    # decode_head_channels = 128
    # auxiliary_head_in_channels = 256
    # auxiliary_head_channels = 64
