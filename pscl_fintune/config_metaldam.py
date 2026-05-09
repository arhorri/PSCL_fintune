"""
Configuration and runner for MetalDAM → PSCL finetuning.

Usage:
    import config_metaldam
    config_metaldam.run(method='self')   # self-supervised pretraining
    config_metaldam.run(method='fine')   # supervised finetuning
"""

import os
import random

import numpy as np
import torch

# PSCL model classes — imported via sys.path set in train_metaldam.py
from model import UNet, MoCo_DenseModel

from finetune import SelfSupervised_MetalDAM, Finetune_MetalDAM

METALDAM_PATCHES = '/mnt/e/Afile/1-Python/MetalDam/data/patches'


class MetalDAMConfig:
    """Configuration for MetalDAM PSCL pretraining and finetuning.

    All parameters mirror PSCL's config class so the training functions
    (SelfSupervised_MetalDAM, Finetune_MetalDAM) can use cfg.* directly.
    """

    def __init__(
        self,
        tt='metaldam',
        method='self',
        data_dir=METALDAM_PATCHES,
        # --- loss ---
        weight=None,
        dice_weight=None,
        dice_bce_ratio=0.5,
        moco_denseloss_ratio=0.7,
        global_weight=1,
        dense_weight=1,
        # --- training schedule ---
        self_max_epoch=200,
        self_save_epoch=10,
        fine_max_epoch=50,
        batch_size=6,
        self_batch_size=8,
        super_lr=1e-3,
        self_lr=1e-4,
        fineturn_lr_en=1e-4,
        fineturn_lr_de=1e-3,
        opti='adam',
        sche=False,
        adamw_decay=1e-3,
        # --- model architecture ---
        backbone=None,
        moco_mode=None,
        drop=0,
        channel=None,
        IncNorm=None,
        DownNorm=None,
        UpNorm=None,
        HeadNrom=None,
        confea_num=64,
        hidfea_num=128,
        multihead=4,
        # --- MoCo / contrastive ---
        selfmode='moco',
        simclrinner=True,
        queue_size=504,
        queue_momentum=0.99,
        temperature=0.07,
        top_k=30,
        patch_num=4,
        sample_ratio=2,
        sample_num=50,
        self_scale_weight=None,
        Patch_sup=True,
        lab_size_decay=0.2,
        patch_lab_decay=0.2,
        patch_size_decay=0.5,
        # --- data augmentation ---
        jitter_d=0.2,
        random_c=0.1,
        # --- checkpoint loading ---
        load_moco_ep='200',
        sup_num='f',
        epoch_map=None,
        # --- environment ---
        env='0',
        seed=42,
        verb=True,
        print_freq=100,
        mapping=None,
    ):
        self.tt      = tt
        self.method  = method
        self.data_dir = data_dir

        # loss
        self.weight           = weight if weight is not None else np.array([1, 1, 5, 5], dtype=np.float32)
        self.dice_weight      = dice_weight if dice_weight is not None else np.array([1, 1, 1, 1], dtype=np.float32)
        self.dice_bce_ratio   = dice_bce_ratio
        self.moco_denseloss_ratio = moco_denseloss_ratio
        self.global_weight    = global_weight
        self.dense_weight     = dense_weight

        # architecture
        self.backbone  = backbone  if backbone  is not None else UNet
        self.moco_mode = moco_mode if moco_mode is not None else MoCo_DenseModel
        self.drop      = drop
        self.channel   = channel   if channel   is not None else [32, 64, 128, 256]
        self.IncNorm   = IncNorm   if IncNorm   is not None else ['BN', 'BN']
        self.DownNorm  = DownNorm  if DownNorm  is not None else ['BN', 'BN']
        self.UpNorm    = UpNorm    if UpNorm    is not None else ['BN', 'LN']
        self.HeadNrom  = HeadNrom  if HeadNrom  is not None else ['LN', '']
        self.confea_num = confea_num
        self.hidfea_num = hidfea_num
        self.multihead  = multihead

        # MoCo / contrastive
        self.selfmode         = selfmode
        self.simclrinner      = simclrinner
        self.queue_size       = queue_size
        self.queue_momentum   = queue_momentum
        self.temperature      = temperature
        self.top_k            = top_k
        self.patch_num        = patch_num
        self.sample_ratio     = sample_ratio
        self.sample_num       = sample_num
        self.self_scale_weight = self_scale_weight if self_scale_weight is not None else [1, 1, 1, 1]
        self.Patch_sup        = Patch_sup
        self.lab_size_decay   = lab_size_decay
        self.patch_lab_decay  = patch_lab_decay
        self.patch_size_decay = patch_size_decay

        # augmentation
        self.jitter_d = jitter_d
        self.random_c = random_c

        # training schedule
        self.self_max_epoch  = self_max_epoch
        self.self_save_epoch = self_save_epoch
        self.fine_max_epoch  = fine_max_epoch
        self.batch_size      = batch_size
        self.self_batch_size = self_batch_size
        self.super_lr        = super_lr
        self.self_lr         = self_lr
        self.fineturn_lr_en  = fineturn_lr_en
        self.fineturn_lr_de  = fineturn_lr_de
        self.opti        = opti
        self.sche        = sche
        self.adamw_decay = adamw_decay

        # checkpoint
        self.load_moco_ep = load_moco_ep
        self.sup_num      = sup_num
        self.epoch_map    = epoch_map if epoch_map is not None else {'1': 0.5, '5': 1, 'f': 10}

        # output directory  (mirrors PSCL's tmp naming: method_backbone_tag/)
        self.tmp = f'{method}_{self.backbone.__name__}_{tt}/'
        os.makedirs(self.tmp, exist_ok=True)

        # eval frequency (every N epochs, same formula as PSCL)
        self.test_freq = int(100 / self.epoch_map[self.sup_num.split('_')[0]])

        # result colour mapping  (PSCL index -> display grey value)
        self.mapping = mapping if mapping is not None else {0: 0, 1: 64, 2: 128, 3: 255}

        # environment
        os.environ['CUDA_VISIBLE_DEVICES'] = env
        self.seed = seed
        self.verb = verb
        self.print_freq = print_freq


def _seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run(
    method: str = 'self',
    tt: str = 'metaldam',
    data_dir: str = METALDAM_PATCHES,
    load_moco_ep: str = '200',
    self_max_epoch: int = 200,
    **kwargs,
):
    """Build a MetalDAMConfig and launch the requested training stage.

    Args:
        method:         'self' (pretraining) or 'fine' (finetuning).
        tt:             Experiment tag — names output directories.
        data_dir:       Path to MetalDam/data/patches/ directory.
        load_moco_ep:   Checkpoint epoch to load when method='fine'.
        self_max_epoch: Number of pretraining epochs.
        **kwargs:       Any MetalDAMConfig parameter to override.
    """
    cfg = MetalDAMConfig(
        method=method,
        tt=tt,
        data_dir=data_dir,
        load_moco_ep=load_moco_ep,
        self_max_epoch=self_max_epoch,
        **kwargs,
    )
    _seed(cfg.seed)

    if method == 'self':
        SelfSupervised_MetalDAM(cfg)
    elif method == 'fine':
        Finetune_MetalDAM(cfg)
    else:
        raise ValueError(f"method must be 'self' or 'fine', got '{method}'")
