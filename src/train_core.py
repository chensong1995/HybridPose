import _init_paths
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from lib.datasets.linemod import LinemodDataset
from lib.datasets.occlusion_linemod import OcclusionLinemodDataset
from lib.datasets.blender_linemod import BlenderLinemodDataset
from lib.datasets.fuse_linemod import FuseLinemodDataset
from lib.datasets.sampler import ImageSizeBatchSampler
from lib.datasets.concat import ConcatDataset
from lib.model_repository import Resnet18_8s
from lib.utils import *
from trainers.coretrainer import CoreTrainer
import pdb

cuda = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda_sym_cor', type=float, default=0.1)
    parser.add_argument('--lambda_mask', type=float, default=1.0)
    parser.add_argument('--lambda_pts2d', type=float, default=10.0)
    parser.add_argument('--lambda_graph', type=float, default=0.1)
    parser.add_argument('--object_name', type=str, default='ape')
    parser.add_argument('--dataset', type=str, default='linemod', choices=['linemod', 'occlusion_linemod'])
    parser.add_argument('--save_dir', type=str, default='saved_weights/linemod/ape')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--test_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--num_keypoints', type=int, default=8)
    args = parser.parse_args()
    return args

def initialize(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_loaders(args):
    if args.dataset == 'linemod':
        linemod_train_set = LinemodDataset(object_name=args.object_name, split='train', augment=True, occlude=False)
        blender_set = BlenderLinemodDataset(object_name=args.object_name, augment=True, occlude=False)
        fuse_set = FuseLinemodDataset(object_name=args.object_name, augment=True)
        test_set = LinemodDataset(object_name=args.object_name, split='test')
        val_set = LinemodDataset(object_name=args.object_name, split='trainval', augment=False, occlude=False)
    elif args.dataset == 'occlusion_linemod':
        linemod_train_set = LinemodDataset(object_name=args.object_name, split='train', augment=True, occlude=False)
        blender_set = BlenderLinemodDataset(object_name=args.object_name, augment=True, occlude=False)
        fuse_set = FuseLinemodDataset(object_name=args.object_name, augment=True)
        test_set = OcclusionLinemodDataset(object_name=args.object_name)
        val_set = LinemodDataset(object_name=args.object_name, split='trainval', augment=False, occlude=False)
    else:
        raise ValueError('Invalid dataset {}'.format(args.dataset))
    train_set = ConcatDataset([linemod_train_set, blender_set, fuse_set])
    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_batch_sampler = ImageSizeBatchSampler(train_sampler, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               num_workers=8,
                                               batch_sampler=train_batch_sampler)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    return train_loader, test_loader, val_loader

def setup_model(args):
    model = Resnet18_8s(num_keypoints=args.num_keypoints)
    if cuda:
        model = nn.DataParallel(model).cuda()
    print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_session(model, optimizer, args)
    else:
        start_epoch = 0
    return model, optimizer, start_epoch

def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# main function
if __name__ == '__main__':
    args = parse_args()
    initialize(args)
    train_loader, test_loader, val_loader = setup_loaders(args)
    model, optimizer, start_epoch = setup_model(args)
    trainer = CoreTrainer(model,
                          optimizer,
                          train_loader,
                          test_loader,
                          args)
    for epoch in range(start_epoch, args.n_epochs):
        trainer.train(epoch)
        if (epoch + 1) % args.test_every == 0:
            trainer.test(epoch)
        if (epoch + 1) % args.save_every == 0:
            trainer.save_model(epoch)
    trainer.generate_data(val_loader)
