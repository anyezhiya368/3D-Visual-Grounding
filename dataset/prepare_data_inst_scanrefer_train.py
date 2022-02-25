'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse, os, pickle

import scannet_util

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
if opt.data_split != 'test':
    files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
    files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
    files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), "{} {}".format(len(files), len(files4))

scanrefer_train = json.load(open(os.path.join('.', 'ScanRefer_filtered_train.json')))
train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
new_scanrefer_train = []
for data in scanrefer_train:
    if data["scene_id"] in train_scene_list:
        new_scanrefer_train.append(data)
scanrefer_train = new_scanrefer_train
MAX_DES_LEN = 126
GLOVE_PICKLE = os.path.join('.', 'glove.p')
with open(GLOVE_PICKLE, "rb") as f:
    glove = pickle.load(f)

def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), fn[:-15] + '_inst_nostuff.pth', _use_new_zipfile_serialization=False)
    print('Saving to ' + fn[:-15] + '_inst_nostuff.pth')


def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[x['label']] != 'floor':
                instance_segids.append(x['segments'])
                labels.append(x['label'])
                assert(x['label'] in scannet_util.g_raw2scannetv2.keys())
    if(fn == 'val/scene0217_00_vh_clean_2.ply' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(sem_labels[pointids])) == 1)

    utterances = []
    referred_ins_idx = []
    language_len = []
    for data in scanrefer_train:
        if data['scene_id'] == fn[7:-15]:
            utterances.append(data['token'])
            referred_ins_idx.append(data['object_id'])
    embeddings = np.zeros((len(utterances), MAX_DES_LEN, 300))
    for tokens_id in range(len(utterances)):
        tokens = utterances[tokens_id]
        language_len.append(len(tokens))
        for token_id in range(MAX_DES_LEN):
            if token_id < len(tokens):
                token = tokens[token_id]
                if token.isspace():
                    continue
                if token in glove:
                    embeddings[tokens_id, token_id, :] = glove[token]
                else:
                    embeddings[tokens_id, token_id, :] = glove["unk"]
            else:
                break

    referred_ins_idx = np.array(referred_ins_idx)
    language_len = np.array(language_len)

    torch.save((coords, colors, sem_labels, instance_labels, embeddings, referred_ins_idx, language_len), fn[:-15]+'_inst_scanrefer.pth', _use_new_zipfile_serialization=False)
    print('Saving to ' + fn[:-15]+'_inst_scanrefer.pth')

# for fn in files:
#     f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
