import PIL.Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset
import pdb

import sys
sys.path.insert(0, '.')
from lib.datasets.augmentation import rotate_instance, crop_resize_instance_v1
from lib.datasets.augmentation import crop_or_padding_to_fixed_size

cuda = torch.cuda.is_available()

class FuseLinemodDataset(Dataset):
    def __init__(self,
                 base_dir='data/fuse_linemod',
                 object_name='ape',
                 size=10000,
                 augment=True):
        self.base_dir = base_dir
        self.object_name = object_name
        linemod_objects = ['ape', 'benchviseblue', 'cam', 'can', 'cat',
                           'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                           'iron', 'lamp', 'phone']
        self.other_object_names = []
        for object_name in linemod_objects:
            if object_name == self.object_name:
                continue
            self.other_object_names.append(object_name)
        # compute length
        data_dir = os.path.join(base_dir, 'fuse')
        self.length = len(list(filter(lambda x: x.endswith('jpg'), os.listdir(data_dir))))
        self.size = size
        # pre-load data into memory
        label_dir = os.path.join(base_dir, '{}_fuse_labels'.format(self.object_name))
        self.pts2d = np.float32(np.load(os.path.join(label_dir, 'keypoints_2d.npy')))
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchviseblue',
                             'can', 'driller', 'eggbox', 'holepuncher', 'lamp']
        self.mask_idx = linemod_cls_names.index(self.object_name) + 1
        self.augment = augment
        if augment:
            self.rot_ang_min = -30
            self.rot_ang_max = 30

    def __len__(self):
        return self.size

    def keypoints_to_map(self, mask, pts2d, unit_vectors=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/datasets/linemod_dataset.py
        mask = mask[0]
        h, w = mask.shape
        n_pts = pts2d.shape[0]
        xy = np.argwhere(mask == 1.)[:, [1, 0]]
        xy = np.expand_dims(xy.transpose(0, 1), axis=1)
        pts_map = np.tile(xy, (1, n_pts, 1))
        pts_map = np.tile(np.expand_dims(pts2d, axis=0), (pts_map.shape[0], 1, 1)) - pts_map
        if unit_vectors:
            norm = np.linalg.norm(pts_map, axis=2, keepdims=True)
            norm[norm < 1e-3] += 1e-3
            pts_map = pts_map / norm
        pts_map_out = np.zeros((h, w, n_pts, 2), np.float32)
        pts_map_out[xy[:, 0, 1], xy[:, 0, 0]] = pts_map
        pts_map_out = np.reshape(pts_map_out, (h, w, n_pts * 2))
        pts_map_out = np.transpose(pts_map_out, (2, 0, 1))
        return pts_map_out

    def keypoints_to_graph(self, mask, pts2d):
        mask = mask[0]
        num_pts = pts2d.shape[0]
        num_edges = num_pts * (num_pts - 1) // 2
        graph = np.zeros((num_edges, 2, mask.shape[0], mask.shape[1]),
                         dtype=np.float32)
        edge_idx = 0
        for start_idx in range(0, num_pts - 1):
            start = pts2d[start_idx]
            for end_idx in range(start_idx + 1, num_pts):
                end = pts2d[end_idx]
                edge = end - start
                graph[edge_idx, 0][mask == 1.] = edge[0]
                graph[edge_idx, 1][mask == 1.] = edge[1]
                edge_idx += 1
        graph = graph.reshape((num_edges * 2, mask.shape[0], mask.shape[1]))
        return graph

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, height, width = idx
        else:
            assert not self.augment
        multiplier = np.random.randint(0, self.length // self.size)
        idx = multiplier * self.size + idx
        # image
        image_name = os.path.join(self.base_dir, 'fuse',
                                  '{}_rgb.jpg'.format(idx))
        image = cv2.imread(image_name)
        # mask
        mask_name = os.path.join(self.base_dir, 'fuse', '{}_mask.png'.format(idx))
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.
        mask = np.float32(mask)
        mask[mask != self.mask_idx] = 0.
        mask[mask == self.mask_idx] = 1.
        # keypoints
        pts2d = self.pts2d[idx]
        # symmetry correspondences
        sym_cor_name = os.path.join(self.base_dir, '{}_fuse_labels'.format(self.object_name),
                                    'cor{}.npy'.format(idx))
        sym_cor = np.float32(np.load(sym_cor_name))
        # data augmentation
        if self.augment:
            foreground = np.sum(mask)
            if foreground > 0:
                image, mask, pts2d, sym_cor = \
                        rotate_instance(image, mask, pts2d, sym_cor,
                                        self.rot_ang_min, self.rot_ang_max)
                foreground = np.sum(mask)
                if foreground > 0:
                    image, mask, pts2d, sym_cor = \
                            crop_resize_instance_v1(image, mask, pts2d, sym_cor,
                                                    height, width)
                else:
                    image, mask, sym_cor = \
                            crop_or_padding_to_fixed_size(image, mask, sym_cor,
                                                          height, width)
            else:
                image, mask, sym_cor = \
                        crop_or_padding_to_fixed_size(image, mask, sym_cor,
                                                      height, width)
        image = image.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image)
        image = self.img_transform(image)
        mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
        sym_cor = sym_cor.transpose([2, 0, 1])
        # keypoint map
        pts2d_map = self.keypoints_to_map(mask, pts2d)
        # graph
        graph = self.keypoints_to_graph(mask, pts2d)
        return {
                'image': image,
                'image_name': image_name,
                'pts2d': pts2d,
                'pts2d_map': pts2d_map,
                'sym_cor': sym_cor,
                'mask': mask,
                'graph': graph
                }

if __name__ == '__main__':
    ape_dataset = FuseLinemodDataset(object_name='ape')
    ape_dataset[0, 400, 600]
    pdb.set_trace()
