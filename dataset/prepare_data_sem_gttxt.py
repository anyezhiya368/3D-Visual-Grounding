'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import os

semantic_label_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
semantic_label_names = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


if __name__ == '__main__':
    split = 'val'
    files = sorted(glob.glob('{}/scene*_inst_nostuff.pth'.format(split)))
    rooms = [torch.load(i) for i in files]

    if not os.path.exists(split + '_sem_gt'):
        os.mkdir(split + '_sem_gt')

    for i in range(len(rooms)):
        xyz, rgb, label, instance_label = rooms[i]   # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = files[i].split('/')[-1][:12]
        print('{}/{} {}'.format(i + 1, len(rooms), scene_name))

        label_new = np.zeros(label.shape[0], dtype=np.int32)

        num_labels = 20
        for label_idx in range(num_labels):
            instance_mask = np.where(label == label_idx)[0]
            label_new[instance_mask] = semantic_label_idxs[label_idx]

        np.savetxt(os.path.join(split + '_sem_gt', scene_name + '.txt'), label_new, fmt='%d')





