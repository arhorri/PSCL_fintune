"""
Google Colab script-mode entry point for MetalDAM → PSCL finetuning.

Paste these 3 cells into a Colab notebook:

── Cell 1 (Config — edit this each session) ──────────────────────────────────
    DRIVE_DIR        = '/content/drive/MyDrive/PSCL_training'
    PSCL_REPO        = 'https://github.com/neulmc/PSCL'
    PRETRAIN_EPOCHS  = 200
    STAGE            = 'both'   # 'self' | 'fine' | 'both'
    SELF_LR          = 1e-3
    FINETUNE_LR_EN   = 1e-3
    FINETUNE_LR_DE   = 1e-3

── Cell 2 (Mount Drive + clone repo — run once per session) ──────────────────
    import os, subprocess, sys
    from google.colab import drive
    drive.mount('/content/drive')   # browser popup — click Allow

    repo = '/content/repo'
    if os.path.exists(repo):
        subprocess.run(['git', '-C', repo, 'pull'], check=True)
    else:
        subprocess.run(['git', 'clone', '--branch', 'kaggle',
                        'https://github.com/arhorri/PSCL_fintune.git', repo], check=True)
    print('Repo ready.')

── Cell 3 (Run training) ─────────────────────────────────────────────────────
    subprocess.run([
        sys.executable,
        f'{repo}/PSCL_fintune/colab_run.py',
        f'--drive-dir={DRIVE_DIR}',
        f'--pscl-repo={PSCL_REPO}',
        f'--pretrain-epochs={PRETRAIN_EPOCHS}',
        f'--stage={STAGE}',
        f'--self-lr={SELF_LR}',
        f'--finetune-lr-en={FINETUNE_LR_EN}',
        f'--finetune-lr-de={FINETUNE_LR_DE}',
    ], check=True)

Outputs are saved automatically to Google Drive at DRIVE_DIR/:
    self_UNet_metaldam/_Numf/f/moco{N}.pt   — Stage 1 checkpoints
    fine_UNet_metaldam/_Numf/f/fine.pt       — Stage 2 best model
    training_curves.png                      — loss + val metrics
"""

import argparse
import glob
import os
import re
import shutil
import sys


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='MetalDAM PSCL training — Colab script mode')
    p.add_argument('--drive-dir', default='',
                   dest='drive_dir',
                   help='Google Drive folder for checkpoint persistence '
                        '(e.g. /content/drive/MyDrive/PSCL_training). '
                        'Empty = skip Drive.')
    p.add_argument('--pscl-repo', default='https://github.com/neulmc/PSCL',
                   dest='pscl_repo',
                   help='GitHub URL of the PSCL repo to clone')
    p.add_argument('--data-dir', default='',
                   dest='data_dir',
                   help='Direct path to MetalDam/data/patches/. '
                        'Defaults to /content/repo/MetalDam/data/patches')
    p.add_argument('--pretrain-epochs', type=int, default=200,
                   dest='pretrain_epochs')
    p.add_argument('--self-lr', type=float, default=1e-3,
                   dest='self_lr',
                   help='Stage 1 pretraining LR (default 1e-3)')
    p.add_argument('--finetune-lr-en', type=float, default=1e-3,
                   dest='finetune_lr_en',
                   help='Stage 2 encoder LR (default 1e-3)')
    p.add_argument('--finetune-lr-de', type=float, default=1e-3,
                   dest='finetune_lr_de',
                   help='Stage 2 decoder LR (default 1e-3)')
    p.add_argument('--stage', choices=['self', 'fine', 'both'], default='both')
    p.add_argument('--gpu', default='0')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

def _mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('Google Drive mounted.')
    except ImportError:
        print('WARNING: not running in Colab — Drive mount skipped.')


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup(args):
    work     = '/content'
    code_dir = os.path.join(work, 'repo', 'PSCL_fintune')
    pscl_root = os.path.join(work, 'PSCL')
    data_dir = args.data_dir if args.data_dir else \
               os.path.join(work, 'repo', 'MetalDam', 'data', 'patches')

    # Clone PSCL source
    if not os.path.exists(pscl_root):
        import subprocess
        print(f'Cloning PSCL from {args.pscl_repo} ...')
        subprocess.run(
            ['git', 'clone', '--depth', '1', args.pscl_repo, pscl_root],
            check=True
        )
        print('PSCL cloned.')
    else:
        print('PSCL source already present.')

    # Auto-detect where model.py lives (root or PSCL/ subfolder)
    if os.path.isfile(os.path.join(pscl_root, 'PSCL', 'model.py')):
        pscl_src = os.path.join(pscl_root, 'PSCL')
    elif os.path.isfile(os.path.join(pscl_root, 'model.py')):
        pscl_src = pscl_root
    else:
        raise FileNotFoundError(
            f'model.py not found in {pscl_root} or {pscl_root}/PSCL/\n'
            'Check the PSCL repo structure.'
        )
    print(f'PSCL source dir: {pscl_src}')

    # Restore Stage 1 checkpoints from Drive
    if args.drive_dir:
        ckpt_src = os.path.join(args.drive_dir, 'self_UNet_metaldam', '_Numf', 'f')
        ckpt_dst = os.path.join(code_dir, 'self_UNet_metaldam', '_Numf', 'f')
        if os.path.exists(ckpt_dst):
            print(f'Checkpoints already present at {ckpt_dst}')
        elif os.path.exists(ckpt_src):
            shutil.copytree(ckpt_src, ckpt_dst)
            ckpts = sorted(f for f in os.listdir(ckpt_dst) if f.endswith('.pt'))
            print(f'Restored {len(ckpts)} checkpoint(s) from Drive: {ckpts}')
        else:
            print('No prior checkpoints found in Drive — starting fresh.')

    # Wire up imports
    for p in [pscl_src, code_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(code_dir)

    return code_dir, pscl_src, data_dir


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def _check_env():
    import torch
    print(f'\nPyTorch : {torch.__version__}')
    print(f'CUDA    : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'GPU     : {torch.cuda.get_device_name(0)}')
        print(f'VRAM    : {props.total_memory / 1e9:.1f} GB')
    else:
        print('WARNING: No GPU. Runtime → Change runtime type → GPU.')


# ---------------------------------------------------------------------------
# Data verification
# ---------------------------------------------------------------------------

def _verify_data(data_dir):
    splits = {'train': 788, 'val': 168, 'test': 192}
    all_ok = True
    print(f'\nVerifying MetalDAM patches at {data_dir} ...')
    for split, expected in splits.items():
        img_dir  = os.path.join(data_dir, split, 'images_norm')
        mask_dir = os.path.join(data_dir, split, 'masks')
        n_img  = len([f for f in os.listdir(img_dir)  if f.endswith('.png')]) if os.path.exists(img_dir)  else 0
        n_mask = len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) if os.path.exists(mask_dir) else 0
        ok = (n_img == expected and n_mask == expected)
        print(f'  [{"OK  " if ok else "FAIL"}] {split:5s}: {n_img} images, {n_mask} masks  (expected {expected})')
        if not ok:
            all_ok = False
    if not all_ok:
        raise SystemExit(
            '\nData verification FAILED.\n'
            f'Expected patches at: {data_dir}\n'
            'Make sure the repo was cloned and MetalDam/ is inside it.'
        )
    print('All splits verified.')


# ---------------------------------------------------------------------------
# Checkpoint detection
# ---------------------------------------------------------------------------

def _detect_checkpoint(code_dir):
    ckpt_dir = os.path.join(code_dir, 'self_UNet_metaldam', '_Numf', 'f')
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'moco*.pt')))
    if ckpts:
        latest = ckpts[-1]
        epoch = int(re.search(r'moco(\d+)\.pt', os.path.basename(latest)).group(1))
        return epoch, latest
    return 0, None


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def _run_stage1(cfg_mod, data_dir, pretrain_epochs, start_epoch, resume_ckpt, gpu,
                self_lr=1e-3):
    remaining = pretrain_epochs - start_epoch
    print(f'\n{"="*60}')
    print('Stage 1 — Self-supervised pretraining')
    print(f'  Epochs : {start_epoch} → {pretrain_epochs}  ({remaining} remaining)')
    print(f'  LR     : {self_lr}  (decays ×0.1 at 50% and 75%)')
    print('='*60)
    cfg_mod.run(
        method='self',
        tt='metaldam',
        data_dir=data_dir,
        self_max_epoch=pretrain_epochs,
        start_epoch=start_epoch,
        resume_ckpt=resume_ckpt,
        selfmode='moco',
        moco_denseloss_ratio=0.5,
        temperature=0.07,
        self_lr=self_lr,
        sche=True,
        env=gpu,
    )
    print('Stage 1 complete.')


def _run_stage2(cfg_mod, code_dir, data_dir, pretrain_epochs, gpu,
                finetune_lr_en=1e-3, finetune_lr_de=1e-3):
    fine_ckpt = os.path.join(code_dir, 'fine_UNet_metaldam', '_Numf', 'f', 'fine.pt')
    moco_ckpt = os.path.join(code_dir, 'self_UNet_metaldam', '_Numf', 'f',
                              f'moco{pretrain_epochs}.pt')
    print(f'\n{"="*60}')
    print('Stage 2 — Supervised finetuning')
    print(f'  encoder LR : {finetune_lr_en}')
    print(f'  decoder LR : {finetune_lr_de}')
    print('='*60)
    if os.path.exists(fine_ckpt):
        print('fine.pt already exists — skipping Stage 2.')
        return
    if not os.path.exists(moco_ckpt):
        raise FileNotFoundError(
            f'Stage 1 checkpoint not found: {moco_ckpt}\n'
            'Run Stage 1 first or restore checkpoints from Drive.'
        )
    print(f'Loading encoder from: {moco_ckpt}')
    cfg_mod.run(
        method='fine',
        tt='metaldam',
        data_dir=data_dir,
        load_moco_ep=str(pretrain_epochs),
        fineturn_lr_en=finetune_lr_en,
        fineturn_lr_de=finetune_lr_de,
        env=gpu,
    )
    print('Stage 2 complete.')


# ---------------------------------------------------------------------------
# Drive backup
# ---------------------------------------------------------------------------

def _save_to_drive(code_dir, drive_dir):
    if not drive_dir:
        return
    os.makedirs(drive_dir, exist_ok=True)
    saved = []
    for label, rel in [
        ('Stage 1 checkpoints', os.path.join('self_UNet_metaldam', '_Numf', 'f')),
        ('Stage 2 model',       os.path.join('fine_UNet_metaldam', '_Numf', 'f')),
    ]:
        src = os.path.join(code_dir, rel)
        dst = os.path.join(drive_dir, rel)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        files = [f for f in os.listdir(dst) if f.endswith('.pt')]
        saved.append(f'  {label}: {len(files)} file(s) → {dst}')

    # Also save training curves
    curves = '/content/training_curves.png'
    if os.path.exists(curves):
        shutil.copy(curves, os.path.join(drive_dir, 'training_curves.png'))
        saved.append(f'  training_curves.png → {drive_dir}')

    if saved:
        print(f'\nSaved to Drive ({drive_dir}):')
        for s in saved:
            print(s)
    else:
        print('\nNothing to save to Drive yet.')


# ---------------------------------------------------------------------------
# Output listing
# ---------------------------------------------------------------------------

def _list_outputs(code_dir):
    print('\n--- Output files ---')
    for label, rel in [
        ('Stage 1 checkpoints', os.path.join('self_UNet_metaldam', '_Numf', 'f')),
        ('Stage 2 model',       os.path.join('fine_UNet_metaldam', '_Numf', 'f')),
    ]:
        d = os.path.join(code_dir, rel)
        if not os.path.exists(d):
            print(f'  {label}: not found')
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith('.pt'))
        print(f'  {label}: {d}')
        for name in files:
            size_mb = os.path.getsize(os.path.join(d, name)) / 1e6
            print(f'    {name:<22s} {size_mb:.0f} MB')


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def _save_curves(code_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available — skipping training curves.')
        return

    def _parse_self(path):
        eps, loss, c_loss, d_loss = [], [], [], []
        if not os.path.exists(path):
            return eps, loss, c_loss, d_loss
        with open(path) as f:
            for line in f:
                m = re.search(r'Epoch \[(\d+)/\d+\].*Loss ([\d.]+).*C ([\d.]+).*Dense ([\d.]+)', line)
                if m:
                    ep = int(m.group(1))
                    if eps and eps[-1] == ep:
                        loss[-1], c_loss[-1], d_loss[-1] = float(m.group(2)), float(m.group(3)), float(m.group(4))
                    else:
                        eps.append(ep); loss.append(float(m.group(2)))
                        c_loss.append(float(m.group(3))); d_loss.append(float(m.group(4)))
        return eps, loss, c_loss, d_loss

    def _parse_fine(path):
        eps, loss, v_acc, v_miou = [], [], [], []
        if not os.path.exists(path):
            return eps, loss, v_acc, v_miou
        with open(path) as f:
            for line in f:
                m = re.search(r'Epoch \[(\d+)/\d+\] Loss ([\d.]+)', line)
                if m:
                    eps.append(int(m.group(1))); loss.append(float(m.group(2)))
                m2 = re.search(r'\[val\] ACC ([\d.]+).*mIoU ([\d.]+)', line)
                if m2:
                    v_acc.append(float(m2.group(1))); v_miou.append(float(m2.group(2)))
        return eps, loss, v_acc, v_miou

    self_log = os.path.join(code_dir, 'self_UNet_metaldam_Numf', 'f', 'log_self.txt')
    fine_log = os.path.join(code_dir, 'fine_UNet_metaldam_Numf', 'f', 'log_fine.txt')

    eps_s, loss_s, c_s, d_s   = _parse_self(self_log)
    eps_f, loss_f, v_acc, v_mi = _parse_fine(fine_log)

    if not eps_s and not eps_f:
        print('No log files found — skipping training curves.')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training curves', fontsize=13)

    if eps_s:
        axes[0].plot(eps_s, loss_s, 'b-',  lw=1.5, label='Total')
        axes[0].plot(eps_s, c_s,    'g--', lw=1,   label='Global InfoNCE')
        axes[0].plot(eps_s, d_s,    'r--', lw=1,   label='Dense patch')
        axes[0].set_title('Stage 1 — Pretraining loss')
        axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No Stage 1 log', ha='center', va='center',
                     transform=axes[0].transAxes)

    if eps_f:
        axes[1].plot(eps_f, loss_f, 'b-', lw=1.5)
        axes[1].set_title('Stage 2 — Finetuning loss')
        axes[1].set_xlabel('Epoch'); axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Stage 2 log', ha='center', va='center',
                     transform=axes[1].transAxes)

    if v_acc:
        x = list(range(len(v_acc)))
        axes[2].plot(x, v_acc, 'g-o', ms=5, label='Val Accuracy')
        axes[2].plot(x, v_mi,  'r-o', ms=5, label='Val mIoU')
        axes[2].set_title('Stage 2 — Validation metrics')
        axes[2].set_xlabel('Eval step'); axes[2].legend(); axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No val metrics', ha='center', va='center',
                     transform=axes[2].transAxes)

    plt.tight_layout()
    out = '/content/training_curves.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    print('MetalDAM → PSCL  |  Colab script mode')
    print(f'  drive_dir        : {args.drive_dir or "(none — Drive not used)"}')
    print(f'  pscl_repo        : {args.pscl_repo}')
    print(f'  pretrain_epochs  : {args.pretrain_epochs}')
    print(f'  stage            : {args.stage}')

    code_dir, _pscl_src, data_dir = _setup(args)
    _check_env()
    _verify_data(data_dir)

    import config_metaldam

    start_epoch, resume_ckpt = _detect_checkpoint(code_dir)
    if start_epoch:
        print(f'\nFound existing Stage 1 checkpoint: epoch {start_epoch}  ({resume_ckpt})')
    else:
        print('\nNo existing Stage 1 checkpoint — starting fresh.')

    if args.stage in ('self', 'both'):
        if start_epoch >= args.pretrain_epochs:
            print(f'\nStage 1 already complete at epoch {start_epoch}. Skipping.')
        else:
            _run_stage1(config_metaldam, data_dir, args.pretrain_epochs,
                        start_epoch, resume_ckpt, args.gpu,
                        self_lr=args.self_lr)

    if args.stage in ('fine', 'both'):
        _run_stage2(config_metaldam, code_dir, data_dir, args.pretrain_epochs, args.gpu,
                    finetune_lr_en=args.finetune_lr_en,
                    finetune_lr_de=args.finetune_lr_de)

    _list_outputs(code_dir)
    _save_curves(code_dir)
    _save_to_drive(code_dir, args.drive_dir)


if __name__ == '__main__':
    main()
