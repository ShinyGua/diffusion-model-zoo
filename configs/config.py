import argparse
import os
from pathlib import Path

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.base = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.data = CN()
# Path to dataset base
_C.data.data_root_path = "datasets/"
# Dataset type
_C.data.dataset_name = "CELEBA-HQ"
#
_C.data.channels = 3
# Image patch size (default: 256)
_C.data.img_size = 256
# Mean pixel value of dataset
_C.data.mean = (0.5, 0.5, 0.5)
# Std deviation of dataset
_C.data.std = (0.5, 0.5, 0.5)

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.model = CN()

# Denoising Diffusion Probabilistic Models from https://arxiv.org/abs/2006.11239
_C.model.ddpm = CN()
_C.model.ddpm.target = "model.guided_diffusion.unet.UNetModel"
_C.model.ddpm.type = "simple"
_C.model.ddpm.in_channels = 3
_C.model.ddpm.out_ch = 3
_C.model.ddpm.ch = 128
_C.model.ddpm.ch_mult = [1, 2, 2, 2]
_C.model.ddpm.num_res_blocks = 2
_C.model.ddpm.attn_resolutions = [16, ]
_C.model.ddpm.dropout = 0.1
_C.model.ddpm.num_heads = 4
_C.model.ddpm.num_head_channels = 64
_C.model.ddpm.var_type = "fixedlarge"
_C.model.ddpm.ema_rate = 0.999
_C.model.ddpm.ema = True
_C.model.ddpm.resamp_with_conv = True
_C.model.ddpm.num_classes = None
_C.model.ddpm.use_checkpoint = False
_C.model.ddpm.num_heads_upsample = -1
_C.model.ddpm.use_scale_shift_norm = True
_C.model.ddpm.resblock_updown = True
_C.model.ddpm.use_new_attention_order = False
# DDPM pre-trained model
_C.model.ddpm.initial_checkpoint = ""

# Classifier model
_C.model.classifier = CN()
_C.model.classifier.pretrain = False  # pretrain classifier on Imagenet
_C.model.classifier.miniset = False
_C.model.classifier.initial_checkpoint = ""
_C.model.classifier.ch = 128
_C.model.classifier.out_ch = 1000
_C.model.classifier.in_channels = 3
_C.model.classifier.ch_mult = [1, 1, 2, 2, 4, 4]
_C.model.classifier.num_res_blocks = 2
_C.model.classifier.attn_resolutions = [32, 16, 8]
_C.model.classifier.dropout = 0.1
_C.model.classifier.resamp_with_conv = True

# Input image center crop percent (for validation only)
_C.model.crop_pct = 0.825

# Input batch size for training (default: 32)
_C.model.batch_size = 32

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.opt = CN()
# Optimizer (default: "sgd")
_C.opt.name = 'adamw'
# Optimizer Epsilon (default: 1e-8, use opt default)
_C.opt.eps = 1e-8
# Optimizer Betas (default: (0.5, 0.999), use opt default)
_C.opt.betas = (0.5, 0.999)
# Optimizer momentum (default: 0.9)
_C.opt.momentum = 0.9
# weight decay (default: 1e-3)
_C.opt.weight_decay = 1e-4
# Clip gradient norm (default: None, no clipping)
_C.opt.clip_grad = 25.0
# Gradient clipping mode. One of ("norm", "value", "agc")
_C.opt.clip_mode = 'norm'
# layer-wise learning rate decay (default: None)
_C.opt.layer_decay = None

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.train = CN()
# LR scheduler (default: "step")
_C.train.sched = 'cosine'
# learning rate (default: 0.0005)
_C.train.lr = 1e-3
# learning rate scale for Discriminator (default: 2.0)
_C.train.lr_s = 1.0
# learning rate noise on/off epoch percentages
_C.train.lr_noise = None
# learning rate noise limit percent (default: 0.67)
_C.train.lr_noise_pct = 0.67
# learning rate noise std-dev (default: 1.0)
_C.train.lr_noise_std = 1.0
# learning rate cycle len multiplier (default: 1.0)
_C.train.lr_cycle_mul = 1.0
# learning rate cycle limit, cycles enabled if > 1
_C.train.lr_cycle_limit = 1
# learning rate k-decay for cosine/poly (default: 1.0)
_C.train.lr_k_decay = 1.0
# warmup learning rate (default: 0.05)
_C.train.warmup_lr_rate = 0.05
# lower lr bound for cyclic schedulers that hit 0 (0.25)
_C.train.min_lr_rate = 0.25
# number of iteration to train (default: 10000)
_C.train.iteration = 2500
# epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
_C.train.epoch_repeats = 0.
# manual iteration number (useful on restarts)
_C.train.start_iteration = 0
# list of decay epoch indices for multistep lr. must be increasing
_C.train.decay_milestones = [0.3, 0.6]
# epoch interval to decay LR
_C.train.decay_epochs = 300
# iteration to warmup LR, if scheduler supports
_C.train.warmup_iteration_rate = 0.1
# epochs to cool down LR at min_lr, after cyclic schedule ends
_C.train.cooldown_epochs = 0
# patience epochs for Plateau LR scheduler (default: 10)
_C.train.patience_epochs = 10
# LR decay rate (default: 0.1)
_C.train.decay_rate = 0.1
# number of training steps for discriminator per iter
_C.train.n_critic = 3

# -----------------------------------------------------------------------------
# Augmentation and regularization settings
# -----------------------------------------------------------------------------
_C.aug = CN()
# Disable all training augmentation, override other train aug args
_C.aug.no_aug = False
# Random resize scale (default: 0.08 1.0)')
_C.aug.scale = [0.08, 1.0]
# Random resize aspect ratio (default: 0.75 1.33)
_C.aug.ratio = [3. / 4., 4. / 3.]
# Horizontal flip training aug probability
_C.aug.hflip = 0.5
# Vertical flip training aug probability
_C.aug.vflip = 0.
# Color jitter factor (default: 0.4)
_C.aug.color_jitter = 0.4
# Use AutoAugment policy. "v0" or "original". (default: None)
_C.aug.aa = 'rand-m9-mstd0.5-inc1'
# Number of augmentation repetitions (distributed training only) (default: 0)
_C.aug.aug_repeats = 0
# Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.aug.aug_splits = 0
# Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
_C.aug.jsd_loss = False
# Enable BCE loss w/ Mixup/CutMix use.
_C.aug.bce_loss = False
# Threshold for binarizing softened BCE targets (default: 0.2)
_C.aug.bce_target_thresh = 0.2
# Random erase prob (default: 0.)
_C.aug.reprob = 0.
# Random erase mode (default: "pixel")
_C.aug.remode = 'pixel'
# Random erase count (default: 1)
_C.aug.recount = 1
# Do not random erase first (clean) augmentation split
_C.aug.resplit = False
# mixup alpha, mixup enabled if > 0. (default: 0.)
_C.aug.mixup = 0.0
# cutmix alpha, cutmix enabled if > 0. (default: 0.)
_C.aug.cutmix = 0.0
# cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
_C.aug.cutmix_minmax = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.aug.mixup_prob = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.aug.mixup_switch_prob = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.aug.mixup_mode = 'batch'
# Turn off mixup after this epoch, disabled if 0 (default: 0)
_C.aug.mixup_off_epoch = 0
# Label smoothing (default: 0.1)
_C.aug.smoothing = 0.1
# Interpolation (random, bilinear, bicubic default: "random")
_C.aug.interpolation = "bicubic"
# Dropout rate (default: 0.)
_C.aug.drop = 0.1
# Drop connect rate, DEPRECATED, use drop-path (default: None)
_C.aug.drop_connect = None
# Drop path rate (default: None)
_C.aug.drop_path = None
# Drop block rate (default: None)
_C.aug.drop_block = None

# -----------------------------------------------------------------------------
# diffusion model settings
# -----------------------------------------------------------------------------
_C.dm = CN()
# schedule name of diffusion model ("linear" or "cosine")
_C.dm.schedule_name = "linear"
# number of timesteps diffusion model (default: 1000)
_C.dm.num_diffusion_timesteps = 1000
_C.dm.sample_timesteps = 1000
_C.dm.beta_start = 1000
_C.dm.beta_start = 0.0001
_C.dm.beta_end = 0.02
_C.dm.eta = 1.0
_C.dm.skip_type = "uniform"

# random seed (default: 42)
_C.seed = 42
# Frequency to logging info
_C.log_freq = 1
# Frequency to save checkpoint
_C.save_freq = 25
# how many training processes to use
_C.workers = 8
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.pin_mem = True
# path to output folder (default: /results)
_C.output = "results"
# local rank for DistributedDataParallel, given by command line argument
_C.local_rank = -1
# Tag of experiment, overwritten by command line argument
_C.tag = 'DDPM'
# Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
_C.dist_bn = 'reduce'
# Auto resume from latest checkpoint
_C.auto_resume = True

_C.reset_average_meter = 500

_C.ip = 500
_C.port = 81228

_C.eval = False
_C.throughput = False
_C.float16 = False


def parse_option():
    parser = argparse.ArgumentParser('adapter diffusion model training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="configs/FFHQ_Test.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--data-root-path', type=str, help='path to dataset root')
    parser.add_argument('--dataset-name', type=str, help='Dataset type')
    parser.add_argument('--float16', action='store_true', help='use torch.float16 or not')
    parser.add_argument('--num-workers', type=int, default=8, help='number of worker for dataloader')
    parser.add_argument('--tag', default="DDPM", help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def _update_config_from_file(config, cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    config.defrost()
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    if getattr(args, 'opts', None) is not None:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if getattr(args, 'batch_size', None) is not None:
        config.model.batch_size = args.batch_size
    if getattr(args, 'data_root_path', None) is not None:
        config.data.data_root_path = args.data_root_path
    if getattr(args, 'dataset_name', None) is not None:
        config.data.dataset_name = args.dataset_name
    if getattr(args, 'lr', None) is not None:
        config.train.lr = args.lr
    if getattr(args, 'num_workers', None) is not None:
        config.workers = args.num_workers
    if getattr(args, 'tag', None) is not None:
        config.tag = args.tag
    if getattr(args, 'float16', None) is not None:
        config.eval = True
    if getattr(args, 'eval', None) is not None:
        config.eval = True

    config.output = Path(config.output).joinpath(config.tag).__str__()
    config.data.target_data_root_path = Path(config.data.data_root_path).joinpath(
        config.data.target_dataset_name).__str__()
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config
