from tensorboardX import SummaryWriter
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')
opt = parser.parse_args()

exp_path = os.path.join('exp', 'scannetv2', 'pointgroup', opt.config.split('/')[-1][:-5])
writer = SummaryWriter(exp_path)


