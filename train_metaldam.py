"""
MetalDAM → PSCL training entry point.

Run from Pscl_fintune/ directory:

    conda activate prep

    # Full pipeline: pretrain then finetune
    python train_metaldam.py

    # Pretraining only
    python train_metaldam.py --stage self

    # Finetuning only (requires a pretrain checkpoint at moco200.pt)
    python train_metaldam.py --stage fine

Checkpoints and logs are written inside this directory under:
    self_UNet_metaldam_Numf/f/   <- pretraining (moco{epoch}.pt, log_self.txt)
    fine_UNet_metaldam_Numf/f/   <- finetuning  (fine.pt, log_fine.txt)
"""

import argparse
import os
import sys

# Make PSCL source importable without modifying PSCL files
_PSCL_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'PSCL', 'PSCL')
sys.path.insert(0, _PSCL_SRC)

# Change working directory to Pscl_fintune so relative checkpoint paths are correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config_metaldam

METALDAM_PATCHES = '/mnt/e/Afile/1-Python/MetalDam/data/patches'
EXPERIMENT_TAG   = 'metaldam'
PRETRAIN_EPOCHS  = 200
FINETUNE_CKPT_EP = '200'   # which pretrain epoch to load for finetuning
GPU              = '0'     # CUDA_VISIBLE_DEVICES


def main():
    parser = argparse.ArgumentParser(description='MetalDAM PSCL training')
    parser.add_argument(
        '--stage', choices=['self', 'fine', 'both'], default='both',
        help='self = pretraining only | fine = finetuning only | both = full pipeline',
    )
    args = parser.parse_args()

    if args.stage in ('self', 'both'):
        print('=' * 60)
        print('Stage 1 — PSCL self-supervised pretraining on MetalDAM')
        print('=' * 60)
        config_metaldam.run(
            method='self',
            tt=EXPERIMENT_TAG,
            data_dir=METALDAM_PATCHES,
            self_max_epoch=PRETRAIN_EPOCHS,
            selfmode='moco',
            moco_denseloss_ratio=0.7,
            temperature=0.07,
            env=GPU,
        )

    if args.stage in ('fine', 'both'):
        print('=' * 60)
        print('Stage 2 — Supervised finetuning on MetalDAM')
        print('=' * 60)
        config_metaldam.run(
            method='fine',
            tt=EXPERIMENT_TAG,
            data_dir=METALDAM_PATCHES,
            load_moco_ep=FINETUNE_CKPT_EP,
            env=GPU,
        )


if __name__ == '__main__':
    main()
