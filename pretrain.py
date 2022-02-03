'''
PointGroup train.py
Written by Li Jiang
'''

import random
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

def compute_bbox(coords):
    # :param coords (npoint, 3) float32 cuda
    # return bbox (6) float32 cuda
    bbox = torch.zeros(6).cuda()
    max_xyz = torch.max(coords, 0).values
    min_xyz = torch.min(coords, 0).values
    center_xyz = (max_xyz + min_xyz) / 2
    bbox[:3] = center_xyz
    scale = max_xyz - min_xyz
    bbox[3:6] = scale
    return bbox

def gitter_bbox(bbox):
    # :param bbox (6) float32 cuda
    # return bbox (6) float32 cuda
    gittered_bbox = bbox.clone()
    x_scale = bbox[3] / 20
    y_scale = bbox[4] / 20
    z_scale = bbox[5] / 20
    while True:
        x_offset = random.gauss(0, x_scale / 2)
        if x_offset < x_scale and x_offset > (-1) * x_scale:
            break
    while True:
        y_offset = random.gauss(0, y_scale / 2)
        if y_offset < y_scale and y_offset > (-1) * y_scale:
            break
    while True:
        z_offset = random.gauss(0, z_scale / 2)
        if z_offset < z_scale and z_offset > (-1) * z_scale:
            break
    gittered_bbox[0] += x_scale
    gittered_bbox[1] += y_scale
    gittered_bbox[2] += z_scale
    while True:
        scale_factor = random.gauss(0, 2.5)
        if scale_factor < 5 and scale_factor > -5:
            break
    gittered_bbox[3:6] *= (1 + scale_factor / 100)
    return gittered_bbox

def inside_bbox(coords, bbox, epsilon = 1e-6):
    # :param coords (N, 3) float32 cuda
    # :param bbox (6) float32 cuda
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    mask1 = torch.gt(x, bbox[0] - bbox[3] / 2 - epsilon)
    mask2 = torch.lt(x, bbox[0] + bbox[3] / 2 + epsilon)
    mask3 = torch.gt(y, bbox[1] - bbox[4] / 2 - epsilon)
    mask4 = torch.lt(y, bbox[1] + bbox[4] / 2 + epsilon)
    mask5 = torch.gt(z, bbox[2] - bbox[5] / 2 - epsilon)
    mask6 = torch.lt(z, bbox[2] + bbox[5] / 2 + epsilon)
    return mask1 & mask2 & mask3 & mask4 & mask5 & mask6

def compute_iou(mask1, mask2):
    # :param mask1 (N) boolean cuda
    # :param mask2 (N) boolean cuda
    i_mask = mask1 & mask2
    u_mask = mask1 | mask2
    i_len = list(torch.nonzero(i_mask).view(-1).shape)[0]
    u_len = list(torch.nonzero(u_mask).view(-1).shape)[0]
    return i_len / u_len

def generate_random_mask(coords):
    # :params coords(N, 3) float32 cuda
    # :params mask(N) boolean cuda
    while True:
        # generate a random bbox
        bbox = torch.zeros(6).cuda()
        max_xyz = torch.max(coords, 0).values
        min_xyz = torch.min(coords, 0).values
        center = (max_xyz + min_xyz) / 2
        scale = max_xyz - min_xyz
        for index in range(3):
            center_point = center[index].item()
            scale_point = scale[index].item() / 2
            while True:
                center_offset = random.gauss(0, scale_point / 2)
                if center_offset < scale_point and center_offset > (-1) * scale_point:
                    bbox[index] = center_point + center_offset
                    break
            while True:
                new_scale = random.gauss(0, scale_point / 2)
                if new_scale < scale_point and new_scale > 0:
                    bbox[index + 3] = new_scale
                    break
        point_mask = inside_bbox(coords, bbox)
        instance_mask = torch.nonzero(point_mask).view(-1)
        # print("INSTANCE_MASK:", instance_mask.shape)
        if list(instance_mask.shape)[0] > 1:
            break
    return point_mask

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
    loss_print = 0
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
        '''
        instance_labels_cpu = instance_labels.int().to('cpu')
        object_idxs = torch.nonzero(instance_labels_cpu > 1).view(-1)  # index for all the points contained inside an object

        batch_idxs_ = coords[:, 0].int()[object_idxs]
        batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input_.batch_size)
        coords_ = coords_float[object_idxs]
        idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, cfg.cluster_radius,
                                                          cfg.cluster_meanActive)
        proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(instance_labels_cpu, idx.cpu(), start_len.cpu(),
                                                                     cfg.cluster_npoint_thre)
        proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
        # print("PROPOSALS_IDX:", proposals_idx.shape)
        # print(proposals_idx)
        # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int

        instance_labels = torch.zeros(coords_float.shape[0]).long()
        cluster_idx = proposals_idx[:, 0].long() + 2
        point_idx = proposals_idx[:, 1].long()
        instance_labels[point_idx]  = cluster_idx
        instance_labels = instance_labels.cuda()
        # max = torch.max(instance_labels)
        # min = torch.min(instance_labels)
        # print("MAX:", max)
        # print("MIN:", min)
        # print("INSTANCE_LABELS:", instance_labels.shape)
        # print(instance_labels)
        mask_4_instance_labels = torch.nonzero(instance_labels > 1).view(-1)
        instance_labels_filtered = instance_labels[mask_4_instance_labels]
        
        instance_labels_unique = torch.unique(instance_labels_filtered, sorted=True)
        mask_dim = torch.max(instance_labels_unique).item() + 1
        instance_mask = torch.zeros(coords_float.shape[0], mask_dim).cuda()
        instance_mask_aug = torch.zeros(coords_float.shape[0], mask_dim).cuda()



        # instance_labels (N) long cuda
        # coords_float (N, 3) float32 cuda
        # print("NUM_POINTS:", coords_float.shape[0])


        '''
        filter_mask = torch.nonzero(instance_labels > 1).view(-1)
        instance_labels_filtered = instance_labels[filter_mask]
        instance_labels_unique = torch.unique(instance_labels_filtered, sorted=True)
        mask_dim = torch.max(instance_labels).item() + 1
        instance_mask = torch.zeros(coords_float.shape[0], mask_dim).cuda()
        instance_mask_aug = torch.zeros(coords_float.shape[0], mask_dim).cuda()


        instance_mask_aug2 = torch.zeros(coords_float.shape[0], mask_dim).cuda()

        ablation_mask = torch.zeros(len(instance_labels_unique)).long().cuda()
        label_num = 0
        for label in instance_labels_unique:
            instance_point_mask = torch.nonzero(instance_labels == label).view(-1)
            if list(instance_point_mask.shape)[0] < 10:
                instance_mask_aug[instance_point_mask, label] = 1
                instance_mask[instance_point_mask, label] = 1
                instance_mask_sp = generate_random_mask(coords_float)
                instance_mask_sp = torch.nonzero(instance_mask_sp).view(-1)
                instance_mask_aug2[instance_mask_sp, label] = 1
                print("LABEL:", label)
                print("INSTANCE_MASK:", instance_point_mask.shape)
            else:
                instance_coords = coords_float[instance_point_mask]
                instance_bbox = compute_bbox(instance_coords)
                instance_box_mask = inside_bbox(coords_float, instance_bbox)
                while True:
                    instance_bbox_gittered = gitter_bbox(instance_bbox)
                    instance_mask_gittered = inside_bbox(coords_float, instance_bbox_gittered)
                    if list(instance_mask_gittered.shape)[0] > 0:
                        break
                iou = compute_iou(instance_box_mask, instance_mask_gittered)
                if iou < 0.65:
                    ablation_mask[label_num] = 1
                label_num += 1
                instance_mask_gittered = torch.nonzero(instance_mask_gittered).view(-1)
                instance_mask_aug[instance_mask_gittered, label] = 1

                while True:
                    instance_mask_sp = generate_random_mask(coords_float)
                    if compute_iou(instance_mask_sp, instance_box_mask) < 0.3:
                        break
                instance_mask_sp = torch.nonzero(instance_mask_sp).view(-1)
                instance_mask_aug2[instance_mask_sp, label] = 1

                instance_box_mask = torch.nonzero(instance_box_mask).view(-1)
                instance_mask[instance_box_mask, label] = 1

                print("LABEL:", label)
                print("INSTANCE_MASK:", instance_box_mask.shape)
                print("INSTANCE_MASK_GITTERED:", instance_mask_gittered.shape)
                # print("IOU:", iou)
        ablation_mask = torch.nonzero(ablation_mask == 0).view(-1)
        if ablation_mask.shape == 0:
            continue
        # src = torch.ones(instance_labels.shape[0], 1).long()
        # instance_labels_2d = instance_labels.clone()
        # instance_labels_2d = instance_labels_2d.unsqueeze(-1).to('cpu')
        # instance_mask = torch.zeros(instance_labels.shape[0], mask_dim, dtype=src.dtype).scatter_(1, instance_labels_2d, src)
        voxel_feats_aug = voxel_feats.clone()
        voxel_coords_aug = voxel_coords.clone()
        input_aug = SparseConvTensor(voxel_feats_aug, voxel_coords_aug.int(), spatial_shape, cfg.batch_size)

        ##### momentum update
        for param_q, param_k in zip(model.parameters(), modelk.parameters()):
          param_k.data = param_k.data.to(device) * m + param_q.data.to(device) * (1.00 - m)

        ##### forward
        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, instance_labels, instance_mask)
        output_feats = ret['instance_feats']
        output_feats = output_feats[ablation_mask]
        output_feats = nn.functional.normalize(output_feats, dim=1)

        retk1 = modelk(input_aug, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, instance_labels, instance_mask_aug)
        output_featsk1 = retk1['instance_feats']
        output_featsk1 = output_featsk1[ablation_mask]
        output_featsk1 = nn.functional.normalize(output_featsk1, dim=1)

        retk2 = modelk(input_aug, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, instance_labels, instance_mask_aug2)
        output_featsk2 = retk2['instance_feats']
        output_featsk2 = nn.functional.normalize(output_featsk2, dim=1)

        output_featsk = torch.cat((output_featsk1, output_featsk2), 0)
        ##### sample
        # length = min(output_feats.shape[0], output_featsk.shape[0])
        # sel = np.random.choice(length, min(length, num_hn_samples), replace=False)
        # output_feats = output_feats[sel]
        # output_featsk = output_featsk[sel]

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
        loss_print += loss.item()


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, loss_print / len(train_loader), time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    writer.add_scalar('loss_pretrain', loss_print / len(train_loader), epoch)

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
