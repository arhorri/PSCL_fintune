"""
Training functions for MetalDAM → PSCL finetuning.

SelfSupervised_MetalDAM : PSCL contrastive pretraining on MetalDAM patches
Finetune_MetalDAM       : Supervised finetuning using the pretrained encoder
load_moco               : Loads encoder weights from a PSCL self-supervised checkpoint
"""

import itertools
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Pscl_fintune-local imports
from data_metaldam import MetalDAMDataset, MetalDAMMoCoDataset, MetalDAMMoCoDatasetSup

# PSCL source is added to sys.path by train_metaldam.py before this module is imported
from utils import Averagvalue, Logger


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_moco(base_encoder, checkpoint_path: str):
    """Load pretrained PSCL encoder weights into a UNet model.

    Strips the 'encoder_q.' prefix from MoCo checkpoint keys and applies
    them to the UNet with strict=False (projection head keys are dropped).
    """
    print(f"\nLoading pretrained encoder from: {checkpoint_path}\n")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt['moco']
    for k in list(state.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.decode'):
            state[k[len("encoder_q."):]] = state[k]
        del state[k]
    base_encoder.load_state_dict(state, strict=False)
    return base_encoder


# ---------------------------------------------------------------------------
# Self-supervised pretraining
# ---------------------------------------------------------------------------

def SelfSupervised_MetalDAM(cfg, start_epoch=0, resume_ckpt=None):
    """PSCL contrastive pretraining on MetalDAM patches.

    Unlabeled pool  : train split (788 patches)
    Labeled guidance: val split  (168 patches, for patch-sampling supervision)
    Checkpoint saved every cfg.self_save_epoch epochs as moco{epoch}.pt
    """
    if not 0 <= cfg.moco_denseloss_ratio <= 1:
        raise ValueError("moco_denseloss_ratio must be in [0, 1]")

    # --- directories & logging ---
    out_root = cfg.tmp + '_Num' + cfg.sup_num.split('_')[0]
    os.makedirs(out_root, exist_ok=True)
    cfg.tmp = os.path.join(out_root, cfg.sup_num)
    os.makedirs(cfg.tmp, exist_ok=True)

    shutil.copy(os.path.realpath(__file__), os.path.join(cfg.tmp, 'finetune_self_copy.py'))
    log = Logger(os.path.join(cfg.tmp, 'log_self.txt'))
    _stdout = sys.stdout
    sys.stdout = log

    # --- datasets ---
    train_loader = DataLoader(
        MetalDAMMoCoDataset(cfg.data_dir, jitter_d=cfg.jitter_d, random_c=cfg.random_c),
        batch_size=cfg.self_batch_size, drop_last=True, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    sup_loader = DataLoader(
        MetalDAMMoCoDatasetSup(cfg.data_dir, sup_split='val',
                               jitter_d=cfg.jitter_d, random_c=cfg.random_c),
        batch_size=1, drop_last=True, shuffle=True,
        num_workers=1, pin_memory=True,
    )
    # Cycle sup_loader so we don't recreate an iterator every batch
    sup_iter = itertools.cycle(sup_loader)

    # --- model ---
    MoComodel = cfg.moco_mode(
        backbone=cfg.backbone,
        queue_size=cfg.queue_size,
        momentum=cfg.queue_momentum,
        temperature=cfg.temperature,
        lab_size_decay=cfg.lab_size_decay,
        patch_num=cfg.patch_num,
        drop=cfg.drop,
        channel=cfg.channel,
        self_scale_weight=cfg.self_scale_weight,
        sample_ratio=cfg.sample_ratio,
        confea_num=cfg.confea_num,
        hidfea_num=cfg.hidfea_num,
        top_k=cfg.top_k,
        patch_lab_decay=cfg.patch_lab_decay,
        patch_size_decay=cfg.patch_size_decay,
        multihead=cfg.multihead,
        IncNorm=cfg.IncNorm,
        DownNorm=cfg.DownNorm,
        UpNorm=cfg.UpNorm,
        HeadNrom=cfg.HeadNrom,
        Patch_sup=cfg.Patch_sup,
        selfmode=cfg.selfmode,
    ).cuda()

    # --- optimizer & scheduler ---
    if cfg.opti == 'adam':
        optimiser = torch.optim.Adam(MoComodel.parameters(), lr=cfg.self_lr)
    else:
        optimiser = torch.optim.AdamW(MoComodel.parameters(), lr=cfg.self_lr,
                                      weight_decay=cfg.adamw_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser,
        milestones=[int(cfg.self_max_epoch * 0.5), int(cfg.self_max_epoch * 0.75)],
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    # --- resume from checkpoint ---
    if resume_ckpt and os.path.exists(resume_ckpt):
        state = torch.load(resume_ckpt, map_location='cpu')
        MoComodel.load_state_dict(state['moco'])
        if 'optim' in state:
            optimiser.load_state_dict(state['optim'])
        for _ in range(start_epoch):
            scheduler.step()
        print(f'Resumed from {resume_ckpt} (epoch {start_epoch})')
        sys.stdout.flush()

    # --- loss helpers ---
    def nt_xent_loss_d(z1, z2, temp, deep, bat):
        k = cfg.top_k * cfg.sample_ratio ** deep
        block1 = np.ones([k, k])
        block0 = np.zeros([k, k])
        rows = [np.hstack([block0 if r == c else block1 for c in range(4)]) for r in range(4)]
        base = np.vstack(rows)
        muban_new = np.tile(np.vstack([np.hstack([base, base]),
                                       np.hstack([base, base])]), (bat, bat))
        if not cfg.simclrinner:
            L = 4 * k
            for ii in range(bat * 2):
                muban_new[ii*L:(ii+1)*L, ii*L:(ii+1)*L] = 0
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        N = z1.shape[0]
        device = z1.device
        reps = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=-1)
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)]).view(2*N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        neg = sim[muban_new == 1].view(2*N, -1)
        logits = torch.cat([pos, neg], dim=1) / temp
        labels = torch.zeros(2*N, device=device, dtype=torch.long)
        return F.cross_entropy(logits, labels, reduction='sum') / (2*N)

    def nt_xent_loss(z1, z2, temp):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        N = z1.shape[0]
        device = z1.device
        reps = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=-1)
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)]).view(2*N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        neg = sim[~diag].view(2*N, -1)
        logits = torch.cat([pos, neg], dim=1) / temp
        labels = torch.zeros(2*N, device=device, dtype=torch.long)
        return F.cross_entropy(logits, labels, reduction='sum') / (2*N)

    n_batches = len(train_loader)
    print(f'Training: epoch {start_epoch} → {cfg.self_max_epoch}, {n_batches} batches/epoch')
    if start_epoch == 0:
        print('Note: first batch may take 3-5 min for CUDA kernel compilation.')
    sys.stdout.flush()

    # --- training loop ---
    for epoch in range(start_epoch, cfg.self_max_epoch + 1):
        MoComodel.train()
        losses = Averagvalue()
        losses_c = losses_d = losses_d1 = losses_d2 = losses_d3 = Averagvalue()

        if epoch % cfg.self_save_epoch == 0:
            torch.save({'moco': MoComodel.state_dict(),
                        'optim': optimiser.state_dict(),
                        'epoch': epoch},
                       os.path.join(cfg.tmp, f'moco{epoch}.pt'))

        t_epoch = time.time()
        for i, (inputs, ids, rot_k, filp, r1, r2, r3, r4) in enumerate(train_loader):
            t_batch = time.time()
            inputs = inputs.cuda(non_blocking=True)
            optimiser.zero_grad()
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            inp_s, ids_s, rk_s, fl_s, r1s, r2s, r3s, r4s = next(sup_iter)
            inp_s = inp_s.cuda(non_blocking=True)
            x_i_s, x_j_s, y_i_s = torch.split(inp_s, [3, 3, 4], dim=1)

            x_i = torch.cat([x_i_s, x_i], dim=0)
            x_j = torch.cat([x_j_s, x_j], dim=0)
            ids  = list(ids_s) + list(ids)
            rot_k = torch.cat([rk_s, rot_k], dim=0)
            filp  = torch.cat([fl_s,  filp],  dim=0)
            r1 = torch.cat([r1s, r1], dim=0)
            r2 = torch.cat([r2s, r2], dim=0)
            r3 = torch.cat([r3s, r3], dim=0)
            r4 = torch.cat([r4s, r4], dim=0)

            if cfg.selfmode == 'moco':
                logit, label, logits_d, labels_d = MoComodel(
                    x_i, x_j, y_i_s, ids, rot_k, filp, r1, r2, r3, r4)
                lc = criterion(logit, label)
                ld = criterion(logits_d, labels_d)
                b1 = cfg.top_k * 4
                b2 = b1 + cfg.top_k * 4 * cfg.sample_ratio
                b3 = b2 + cfg.top_k * 4 * cfg.sample_ratio ** 2
                ld1 = criterion(logits_d[:b1],   labels_d[:b1])
                ld2 = criterion(logits_d[b1:b2], labels_d[b1:b2])
                ld3 = criterion(logits_d[b2:b3], labels_d[b2:b3])
                loss = (lc * cfg.global_weight * (1 - cfg.moco_denseloss_ratio) +
                        ld * cfg.moco_denseloss_ratio * cfg.dense_weight)
            else:  # simclr
                q, k, qd1, kd1, qd2, kd2, qd3, kd3 = MoComodel(
                    x_i, x_j, y_i_s, ids, rot_k, filp, r1, r2, r3, r4)
                ld1 = nt_xent_loss_d(qd1, kd1, cfg.temperature, 0, q.shape[0])
                ld2 = nt_xent_loss_d(qd2, kd2, cfg.temperature, 1, q.shape[0])
                ld3 = nt_xent_loss_d(qd3, kd3, cfg.temperature, 2, q.shape[0])
                lc  = nt_xent_loss(q, k, cfg.temperature)
                ld  = (ld1 * cfg.self_scale_weight[0] +
                       ld2 * cfg.self_scale_weight[1] +
                       ld3 * cfg.self_scale_weight[2]) / 3
                loss = (lc * cfg.global_weight * (1 - cfg.moco_denseloss_ratio) +
                        ld * cfg.moco_denseloss_ratio * cfg.dense_weight)

            loss.backward()
            optimiser.step()
            losses.update(loss.item(), inputs.size(0))

            if i % cfg.print_freq == 0 or i == len(train_loader) - 1:
                lr = optimiser.param_groups[0]['lr']
                elapsed = time.time() - t_batch
                print(f'Epoch [{epoch}/{cfg.self_max_epoch}] '
                      f'[{i+1}/{len(train_loader)}] '
                      f'Loss {losses.avg:.4f}  C {lc.item():.4f}  '
                      f'Dense {ld.item():.4f}  lr {lr}  '
                      f'batch {elapsed:.1f}s')

        if cfg.sche:
            scheduler.step()

    log.close()
    sys.stdout = _stdout


# ---------------------------------------------------------------------------
# Supervised finetuning
# ---------------------------------------------------------------------------

def Finetune_MetalDAM(cfg):
    """Supervised finetuning on MetalDAM using a PSCL pretrained encoder.

    - Loads encoder from self-supervised checkpoint.
    - Trains on train split; evaluates on val split every test_freq epochs.
    - Saves best checkpoint (fine.pt) based on val pixel accuracy.
    - Prints final mIoU + per-class IoU on test split.
    """

    def _dice(pred, tgt):
        pred = pred.contiguous().view(pred.shape[0], -1)
        tgt  = tgt.contiguous().view(tgt.shape[0],  -1)
        num  = torch.sum(pred * tgt, dim=1) + 1e-4
        den  = torch.sum(pred + tgt, dim=1) + 1e-4
        return (1 - 2 * num / den).mean()

    def _multidice(pred, tgt):
        b, c, h, w = pred.shape
        pred = torch.softmax(pred.view(b, c, -1), dim=1)
        tgt  = tgt.view(b, c, -1)
        return [_dice(pred[:, i], tgt[:, i]) * cfg.dice_weight[i]
                for i in range(c)]

    def train_epoch(model, loader, optimizer, epoch):
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(cfg.weight).float().cuda())
        model.train()
        meter = Averagvalue()
        for image, label in loader:
            image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()
            pred   = model(image)
            l_bce  = criterion(pred, label)
            dices  = _multidice(pred, label)
            l_dice = (dices[1] + dices[2] + dices[3]) / 3
            loss   = l_bce * (1 - cfg.dice_bce_ratio) + l_dice * cfg.dice_bce_ratio
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), image.size(0))
        print(f'Epoch [{epoch}/{cfg.fine_max_epoch}] Loss {meter.avg:.4f}')

    def evaluate(model, loader, epoch, split_name):
        eval_dir = os.path.join(cfg.tmp, f'epoch_{epoch}_{split_name}')
        os.makedirs(eval_dir, exist_ok=True)
        model.eval()
        accs, i1s, i2s, i3s, o1s, o2s, o3s = [], [], [], [], [], [], []
        with torch.no_grad():
            for image, label, filename in loader:
                image, label = image.cuda(), label.cuda()
                pred_np  = torch.softmax(model(image), dim=1).cpu().numpy()[0]
                label_np = label.cpu().numpy()[0]
                pred_cls  = np.argmax(pred_np,  axis=0)
                label_cls = np.argmax(label_np, axis=0)
                # save coloured prediction
                out = pred_cls.copy()
                for c in range(4):
                    out[out == c] = cfg.mapping[c]
                cv2.imwrite(os.path.join(eval_dir, filename[0]), out)
                accs.append(np.mean(pred_cls == label_cls))
                for ci, (il, ol) in enumerate([(i1s, o1s), (i2s, o2s), (i3s, o3s)], 1):
                    pc = (pred_cls  == ci).astype(float)
                    lc = label_np[ci]
                    il.append(2 * np.logical_and(pc, lc))
                    ol.append(pc.sum() + lc.sum())
        iou = [np.array(il).sum() / np.array(ol).sum()
               for il, ol in [(i1s, o1s), (i2s, o2s), (i3s, o3s)]]
        miou = sum(iou) / 3
        acc  = np.mean(accs)
        print(f'[{split_name}] ACC {acc:.4f}  mIoU {miou:.4f}  '
              f'IoU1 {iou[0]:.4f}  IoU2 {iou[1]:.4f}  IoU3 {iou[2]:.4f}')
        return acc

    # --- directories & logging ---
    out_root = cfg.tmp + '_Num' + cfg.sup_num.split('_')[0]
    os.makedirs(out_root, exist_ok=True)
    cfg.tmp = os.path.join(out_root, cfg.sup_num)
    os.makedirs(cfg.tmp, exist_ok=True)

    shutil.copy(os.path.realpath(__file__), os.path.join(cfg.tmp, 'finetune_fine_copy.py'))
    log = Logger(os.path.join(cfg.tmp, 'log_fine.txt'))
    _stdout = sys.stdout
    sys.stdout = log

    # --- datasets ---
    train_loader = DataLoader(
        MetalDAMDataset(cfg.data_dir, split='train'),
        batch_size=cfg.batch_size, drop_last=True, shuffle=True,
    )
    val_loader = DataLoader(
        MetalDAMDataset(cfg.data_dir, split='val'),
        batch_size=1, drop_last=False, shuffle=False,
    )
    test_loader = DataLoader(
        MetalDAMDataset(cfg.data_dir, split='test'),
        batch_size=1, drop_last=False, shuffle=False,
    )

    # --- model ---
    model = cfg.backbone(cfg.drop, cfg.channel,
                         IncNorm=cfg.IncNorm, DownNorm=cfg.DownNorm,
                         UpNorm=cfg.UpNorm).cuda()

    # load pretrained encoder (path mirrors SelfSupervised_MetalDAM's save path)
    self_tmp = cfg.tmp.replace(
        'fine_' + cfg.backbone.__name__,
        'self_' + cfg.backbone.__name__,
    )
    ckpt_path = os.path.join(self_tmp, f'moco{cfg.load_moco_ep}.pt')
    model = load_moco(model, ckpt_path)

    # separate LRs: encoder layers lower, decoder layers higher
    params = [
        {'params': model.encode.inc.parameters(),   'lr': cfg.fineturn_lr_en},
        {'params': model.encode.down1.parameters(), 'lr': cfg.fineturn_lr_en},
        {'params': model.encode.down2.parameters(), 'lr': cfg.fineturn_lr_en},
        {'params': model.encode.down3.parameters(), 'lr': cfg.fineturn_lr_en},
        {'params': model.encode.up1.parameters(),   'lr': cfg.fineturn_lr_de},
        {'params': model.encode.up2.parameters(),   'lr': cfg.fineturn_lr_de},
        {'params': model.encode.up3.parameters(),   'lr': cfg.fineturn_lr_de},
        {'params': model.decode.parameters(),       'lr': cfg.fineturn_lr_de},
    ]
    optimizer = torch.optim.Adam(params)

    acc_best = 0
    for epoch in range(cfg.fine_max_epoch):
        train_epoch(model, train_loader, optimizer, epoch)
        if (epoch + 1) % cfg.test_freq == 0:
            acc = evaluate(model, val_loader, epoch, 'val')
            if acc > acc_best:
                acc_best = acc
                torch.save({'finetune': model.state_dict(), 'epoch': epoch, 'acc': acc},
                           os.path.join(cfg.tmp, 'fine.pt'))
                print(f'  -> saved best checkpoint (acc={acc:.4f})')

    print('\n--- Final evaluation on test split ---')
    evaluate(model, test_loader, cfg.fine_max_epoch, 'test')

    log.close()
    sys.stdout = _stdout
