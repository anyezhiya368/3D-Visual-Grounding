'''
PointGroup train.py
Written by Li Jiang
'''

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np

import torch.nn as nn
from util.config import cfg
from util.log import logger
import util.utils as utils
from spconv.pytorch.core import SparseConvTensor
from lib.pointgroup_ops.functions import pointgroup_ops

m = 0.95
num_hn_samples = 4096
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def random_flip( points, probability=0.5):
    x = np.random.choice(
         [False, True], replace=False, p=[1 - probability, probability])
    z =  np.random.choice(
         [False, True], replace=False, p=[1 - probability, probability])
    if z:
         points[:, 2] = -points[:, 2]
    if x:
        points[:, 0] = -points[:, 0]
    return  points


def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def gitter_feat(refer_mask, feats, v2p_map, mode, voxel_feats, voxel_coords):
    feat_dim = feats.shape[1] - 3
    feats = feats[:, 0:feat_dim]
    coords_float = feats[:, feat_dim:feat_dim + 3]
    refer_mask_feats = refer_mask.repeat(1, feat_dim)
    remain_mask_feats = refer_mask.repeat(1, feat_dim)
    feat_refer = torch.masked_select(feats, refer_mask_feats)
    feat_noise = torch.randn(feat_refer.shape[0], feat_refer.shape[1])
    feat_refer += feat_noise
    feat_remain = torch.masked_select(feats, remain_mask_feats)
    feats = torch.cat((feat_refer, feat_remain), 0)
    refer_mask_coord = refer_mask.repeat(1, 3)
    remain_mask_coord = refer_mask.repeat(1, 3)
    coord_refer = torch.masked_select(coords_float, refer_mask_coord)
    coord_remain = torch.masked_select(coords_float, remain_mask_coord)
    coords_float = torch.cat((coord_refer, coord_remain))
    feats_aug = torch.cat((feats, coords_float), 1)

    voxel_feats_aug = pointgroup_ops.voxelization(feats_aug, v2p_map, mode)
    voxel_coords_aug = voxel_coords.clone()

    return voxel_feats_aug, voxel_coords_aug

def train_epoch(train_loader, model, modelk, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}
    CELOSS = nn.CrossEntropyLoss().cuda()

    model.train()
    modelk.train()
    model.requires_grad_(True)
    modelk.requires_grad_(False)

    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)


        ##### prepare input
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ##### prepare augmentated input

        '''some data augmentation should be implemented here'''
        '''feature gitter for the cropped area'''

        #refer_mask = batch['refer_mask'] # (N) a boolean mask indicating whether the point is inside the referred objedt
        #voxel_feats_aug, voxel_coords_aug = gitter_feat(refer_mask, feats, v2p_map, cfg.mode, voxel_feats, voxel_coords)

        voxel_feats_aug = voxel_feats.clone()
        voxel_coords_aug = voxel_coords * 0.95
        input_aug = SparseConvTensor(voxel_feats_aug, voxel_coords_aug.int(), spatial_shape, cfg.batch_size)

        ##### momentum update
        for param_q, param_k in zip(model.parameters(), modelk.parameters()):
          param_k.data = param_k.data.to(device) * m + param_q.data.to(device) * (1.00 - m)

        ##### forward
        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        output_feats = ret['output_feats']
        output_feats = nn.functional.normalize(output_feats, dim=1)

        retk = modelk(input_aug, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        output_featsk = retk['output_feats']
        output_featsk = nn.functional.normalize(output_featsk, dim=1)

        ##### sample
        length = output_featsk.shape[0]
        sel = np.random.choice(length, min(length, num_hn_samples), replace=False)
        output_feats = output_feats[sel]
        output_featsk = output_featsk[sel]

        logits = torch.mm(output_feats, output_featsk.transpose(1, 0))
        logits = logits.type(torch.float32)
        labels = torch.arange(output_feats.size()[0])
        labels = labels.cuda()
        loss = CELOSS(logits/0.07, labels)


        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.4f} data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
            (epoch, cfg.epochs, i + 1, len(train_loader), loss.item(),
             data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        if (i == len(train_loader) - 1): print()


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, loss, time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)



if __name__ == '__main__':
    ##### init
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')

    if model_name == 'pointgroup':
        from model.pointgroup.pointgroup import PointGroup as Network
        from model.pointgroup.pointgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    model = Network(cfg)

    modelk = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()
    modelk = modelk.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset()
            dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda)      # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model,modelk, model_fn, optimizer, epoch)
