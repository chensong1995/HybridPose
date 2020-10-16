import PIL.Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import os
import random
import pickle
from torch.utils.data import Dataset
import pdb

import sys
sys.path.insert(0, '.')
from lib.datasets.augmentation import rotate_instance, crop_resize_instance_v1
from lib.datasets.augmentation import crop_or_padding_to_fixed_size

cuda = torch.cuda.is_available()

class BlenderLinemodDataset(Dataset):
    def __init__(self,
                 base_dir='data/blender_linemod',
                 linemod_base_dir='data/linemod',
                 object_name='ape',
                 size=10000,
                 augment=True,
                 occlude=True,
                 split='train'):
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
        data_dir = os.path.join(base_dir, self.object_name)
        self.length = len(list(filter(lambda x: x.endswith('jpg'), os.listdir(data_dir))))
        self.size = size
        # pre-load data into memory
        label_dir = os.path.join(base_dir, '{}_labels'.format(self.object_name))
        self.pts2d = np.float32(np.load(os.path.join(label_dir, 'keypoints_2d.npy')))
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        self.augment = augment
        if augment:
            self.rot_ang_min = -30
            self.rot_ang_max = 30
        self.occlude = occlude
        self.split = split
        pts3d_name = os.path.join(linemod_base_dir, 'keypoints',
                                  object_name, 'keypoints_3d.npy')
        self.pts3d = np.float32(np.load(pts3d_name))
        normal_name = os.path.join(linemod_base_dir, 'symmetries',
                                   object_name, 'symmetries.txt')
        self.normal = self.read_normal(normal_name)

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

    def occlude_with_another_object(self, image, mask):
        orig_image, orig_mask = image.copy(), mask.copy()
        try:
            while True:
                other_object_name = random.choice(self.other_object_names)
                other_idx = random.randrange(10000)
                other_image_name = os.path.join(self.base_dir, other_object_name,
                                                '{}.jpg'.format(other_idx))
                other_image = cv2.imread(other_image_name)
                other_mask_name = os.path.join(self.base_dir, '{}_labels'.format(other_object_name),
                                               'mask{}.png'.format(other_idx))
                other_mask = cv2.imread(other_mask_name, cv2.IMREAD_GRAYSCALE)
                other_ys, other_xs = np.nonzero(other_mask)
                other_ymin, other_ymax = np.min(other_ys), np.max(other_ys)
                other_xmin, other_xmax = np.min(other_xs), np.max(other_xs)
                ys, xs = np.nonzero(mask)
                ymin, ymax = np.min(ys), np.max(ys)
                xmin, xmax = np.min(xs), np.max(xs)
                y_scale = (ymax - ymin) / (other_ymax - other_ymin)
                x_scale = (xmax - xmin) / (other_xmax - other_xmin)
                scale = max(y_scale, x_scale)
                scale = np.random.uniform(0.6*scale, 2.0*scale)
                other_image = cv2.resize(other_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                other_mask = cv2.resize(other_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                other_ys, other_xs = np.nonzero(other_mask)
                other_ymin, other_ymax = np.min(other_ys), np.max(other_ys)
                other_xmin, other_xmax = np.min(other_xs), np.max(other_xs)
                other_mask = other_mask[other_ymin:other_ymax+1, other_xmin:other_xmax+1]
                other_image = other_image[other_ymin:other_ymax+1, other_xmin:other_xmax+1]

                start_y = np.random.randint(ymin - other_mask.shape[0] + 1, ymax + 1)
                end_y = start_y + other_mask.shape[0]
                start_x = np.random.randint(xmin - other_mask.shape[1] + 1, xmax + 1)
                end_x = start_x + other_mask.shape[1]
                if start_y < 0:
                    other_mask = other_mask[-start_y:]
                    other_image = other_image[-start_y:]
                    start_y = 0
                if end_y > image.shape[0]:
                    end_y = image.shape[0]
                    other_mask = other_mask[:end_y-start_y]
                    other_image = other_image[:end_y-start_y]
                if start_x < 0:
                    other_mask = other_mask[-start_x:]
                    other_image = other_image[-start_x:]
                    start_x = 0
                if end_x > image.shape[0]:
                    end_x = image.shape[0]
                    other_mask = other_mask[:end_x-start_x]
                    other_image = other_image[:end_x-start_x]
                other_outline = (other_mask == 0)[:, :, None]
                image[start_y:end_y, start_x:end_x] *= other_outline
                other_image[other_mask == 0] = 0
                image[start_y:end_y, start_x:end_x] += other_image
                mask[start_y:end_y, start_x:end_x] *= (other_mask == 0)
                if mask.sum() >= 20:
                    break
        except:
            # maybe other are some rounding issues
            return orig_image, orig_mask
        return image, mask

    def read_pose(self, idx):
        filename = os.path.join(self.base_dir,
                                self.object_name,
                                '{}_RT.pkl'.format(idx))
        with open(filename, 'rb') as f:
            pkl = pickle.load(f)
            RT = pkl['RT']
            R = np.array(RT[:, :3])
            T = np.array(RT[:, -1]).transpose()
        return R, T

    def read_normal(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            normal = np.array(lines[3].strip().split(), dtype=np.float32)
        return normal

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, height, width = idx
        else:
            assert not self.augment
        multiplier = np.random.randint(0, self.length // self.size)
        idx = multiplier * self.size + idx
        # image
        image_name = os.path.join(self.base_dir, self.object_name,
                                  '{}.jpg'.format(idx))
        image = cv2.imread(image_name)
        # mask
        mask_name = os.path.join(self.base_dir, '{}_labels'.format(self.object_name),
                                 'mask{}.png'.format(idx))
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        # online occlusion
        if self.occlude:
            image, mask = self.occlude_with_another_object(image, mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.
        mask = np.float32(mask)
        mask[mask != 0.] = 1.
        # keypoints
        pts2d = self.pts2d[idx]
        # symmetry correspondences
        sym_cor_name = os.path.join(self.base_dir, '{}_labels'.format(self.object_name),
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
        if self.split == 'train':
            return {
                    'image': image,
                    'image_name': image_name,
                    'pts2d': pts2d,
                    'pts2d_map': pts2d_map,
                    'sym_cor': sym_cor,
                    'mask': mask,
                    'graph': graph
                    }
        else:
            R, t = self.read_pose(idx)
            pts3d = self.pts3d
            normal = self.normal
            return {
                    'image': image,
                    'image_name': image_name,
                    'pts2d': pts2d,
                    'pts2d_map': pts2d_map,
                    'sym_cor': sym_cor,
                    'mask': mask,
                    'graph': graph,
                    'R': R,
                    't': t,
                    'pts3d': pts3d,
                    'normal': normal
                    }

if __name__ == '__main__':
    ape_dataset = BlenderLinemodDataset(object_name='ape')
    ape_dataset[0, 400, 600]
    pdb.set_trace()
