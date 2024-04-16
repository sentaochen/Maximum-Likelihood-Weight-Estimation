# -*- coding: utf-8 -*-
import torch
import random
import os
import argparse

from loader.data_loader import load_data_for_PDA
from model.model import MODEL
from utils.train_PDA import train_for_PDA
from utils import globalvar as gl
# import dataloader as dir_dataloader
parser = argparse.ArgumentParser(description='PDA Classification')
parser.add_argument('--root_dir', type=str, default='./data/OfficeHome',
                    help='root dir of the dataset')     
parser.add_argument('--dataset', type=str, default='OfficeHome',
                    help='the name of dataset')
parser.add_argument('--source', type=str, default='Art',
                    help='source domain')
parser.add_argument('--target', type=str, default='Clipart',
                    help='target domain')
parser.add_argument('--net', type=str, default='resnet',
                    help='which network to use')
parser.add_argument('--phase', type=str, default='train',
                    help='the phase of training model')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_mult', type=float, nargs=4, default=[0.1, 0.1, 1, 1],
                    help='lr_mult (default: [0.1, 0.1, 1, 1])')
parser.add_argument('--steps', type=int, default=50000,
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--save_interval', type=int, default=2000,
                    help='how many batches to wait before saving a model(default: 2000)')
parser.add_argument('--update_weight', type=int, default=2000,
                    help='how many batches to update weight(default: 2000)')
parser.add_argument('--start_update', type=int, default=0,
                    help='how many batches to start update weight(default: 0)')
parser.add_argument('--save_check', type=bool, default=True,
                    help='save checkpoint or not(default: True)')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping to wait for improvment before terminating. (default: 10 (5000 iterations))')
parser.add_argument('--early', type=bool, default=True,
                    help='early stopping or not(default: True)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='MOMENTUM of SGD (default: 0.9)')
parser.add_argument('--decay', type=int, default=0.0005,
                    help='DECAY of SGD (default: 0.0005)')
parser.add_argument('--message', type=str, default='PDA by MLWE', help='the annotation')     
parser.add_argument("--batch_size", default=16, type=int, help="batch size") 
parser.add_argument("--seed", default=0, type=int, help="random seed") 
parser.add_argument("--class_num", default=65, type=int, help="class_num") 
parser.add_argument("--partial_classes_num", default=25, type=int, help="class_num") 


# ===========================================================================================================================================================================
args = parser.parse_args()
# args.root_dir = '/data' # YOUR PATH



DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
gl._init()
gl.set_value('DEVICE', DEVICE)

bottleneck_dim = 1024
if args.batch_size is None:
    batch_size = 32
else:
    batch_size = args.batch_size
print(args)
print('class_num: {}, bottleneck_dim: {}, batch_size:{}'.format(
    args.class_num, bottleneck_dim, batch_size))

seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

record_dir = './record_PDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
check_path = './save_model_PDA/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(check_path):
    os.makedirs(check_path)
record_file = os.path.join(record_dir, 'MLWE_{}_{}_to_{}.txt'.format(args.net, args.source, args.target))

gl.set_value('check_path', check_path)
gl.set_value('record_file', record_file)

if __name__ == '__main__':
    dataloaders = {}
    model = MODEL(args.net, args.class_num, bottleneck_dim).to(DEVICE)

    dataloaders['src_train_l'], dataloaders['tar_train_ul'] = load_data_for_PDA(
        args.root_dir, args.dataset, args.source, args.target, 'train', batch_size, args.net, args=args)
    dataloaders['src_test'], dataloaders['tar_test'] = load_data_for_PDA(
        args.root_dir, args.dataset, args.source, args.target, 'test', batch_size, args.net, args=args)

    print(len(dataloaders['src_train_l'].dataset), len(dataloaders['tar_train_ul'].dataset))
    train_for_PDA(args, model, dataloaders)
    