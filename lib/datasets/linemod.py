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

class LinemodDataset(Dataset):
    def __init__(self,
                 base_dir='data/linemod',
                 object_name='all',
                 split='train',
                 augment=True,
                 occlude=False):
        self.camera_intrinsic = {'fu': 572.41140, 'fv': 573.57043,
                                 'uc': 325.26110, 'vc': 242.04899}
        self.K = np.matrix([[self.camera_intrinsic['fu'], 0, self.camera_intrinsic['uc']],
                            [0, self.camera_intrinsic['fv'], self.camera_intrinsic['vc']],
                            [0, 0, 1]], dtype=np.float32)
        self.base_dir = base_dir
        linemod_objects = ['ape', 'benchviseblue', 'cam', 'can', 'cat',
                           'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                           'iron', 'lamp', 'phone']
        if object_name == 'all':
            self.object_names = linemod_objects
        elif object_name in linemod_objects:
            self.object_names = [object_name]
        else:
            raise ValueError('Invalid object name: {}'.format(object_name))
        self.other_object_names = []
        for object_name in linemod_objects:
            if object_name in self.object_names:
                continue
            self.other_object_names.append(object_name)
        # compute length
        self.lengths = {}
        self.total_length = 0
        self.split_indices = {}
        self.split = split
        for object_name in self.object_names:
            train_range = os.path.join(base_dir, 'ranges', '{}.txt'.format(object_name))
            train_range = list(np.loadtxt(train_range).astype(np.int32))
            if split in ['train', 'trainval']:
                length = len(train_range)
                self.split_indices[object_name] = train_range
            else:
                train_test_length = len(list(filter(lambda x: x.endswith('jpg'),
                                                    os.listdir(os.path.join(base_dir, 'original_dataset',object_name, 'data')))))
                train_test_range = list(range(train_test_length))
                test_range = set(train_test_range).difference(set(train_range))
                test_range = list(test_range)
                test_range.sort()
                self.split_indices[object_name] = test_range
                length = len(test_range)
            self.lengths[object_name] = length
            self.total_length += length
        # pre-load data into memory
        self.pts2d = {}
        self.pts3d = {}
        self.normals = {}
        for object_name in self.object_names:
            # keypoints
            pts2d_name = os.path.join(self.base_dir, 'keypoints',
                                      object_name, 'keypoints_2d.npy')
            pts2d = np.float32(np.load(pts2d_name))
            self.pts2d[object_name] = pts2d
            pts3d_name = os.path.join(self.base_dir, 'keypoints',
                                      object_name, 'keypoints_3d.npy')
            pts3d = np.float32(np.load(pts3d_name))
            self.pts3d[object_name] = pts3d
            # symmetry plane normals
            normal_name = os.path.join(self.base_dir, 'symmetries',
                                       object_name, 'symmetries.txt')
            self.normals[object_name] = self.read_normal(normal_name)
        self.img_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        self.augment = augment and split in ['train', 'trainall']
        if augment:
            self.rot_ang_min = -30
            self.rot_ang_max = 30
        self.occlude = occlude

    def read_3d_points(self, filename):
        with open(filename) as f:
            in_vertex_list = False
            vertices = []
            in_mm = False
            for line in f:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    vertex = np.array([[float(vertex[0])],
                                       [float(vertex[1])],
                                       [float(vertex[2])]], dtype=np.float32)
                    if in_mm:
                        vertex = vertex / np.float32(10) # mm -> cm
                    vertex = vertex / np.float32(100) # cm -> m
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True
        return vertices

    def __len__(self):
        return self.total_length

    def read_rotation(self, filename):
        with open(filename) as f:
            f.readline()
            R = []
            for line in f:
                R.append(line.split())
            R = np.array(R, dtype=np.float32)
        return R

    def read_translation(self, filename):
        with open(filename) as f:
            f.readline()
            T = []
            for line in f:
                T.append([line.split()[0]])
            T = np.array(T, dtype=np.float32)
            T = T / np.float32(100) # cm -> m
        return T

    def read_normal(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            normal = np.array(lines[3].strip().split(), dtype=np.float32)
        return normal

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
        if self.split == 'test':
            return image, mask
        else:
            orig_image, orig_mask = image.copy(), mask.copy()
            try:
                while True:
                    other_object_name = random.choice(self.other_object_names)
                    other_length = len(list(filter(lambda x: x.endswith('jpg'),
                                                   os.listdir(os.path.join(self.base_dir, 'original_dataset', other_object_name, 'data')))))
                    other_idx = random.randrange(other_length)
                    other_image_name = os.path.join(self.base_dir, 'original_dataset',
                                                    other_object_name, 'data',
                                                    'color{}.jpg'.format(other_idx))
                    other_image = cv2.imread(other_image_name)
                    other_mask_name = os.path.join(self.base_dir, 'masks',
                                                   other_object_name, 'mask{}.png'.format(other_idx))
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

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            local_idx, height, width = idx
        else:
            local_idx = idx
            assert not self.augment
        for object_name in self.object_names:
            if local_idx < self.lengths[object_name]:
                local_idx = self.split_indices[object_name][local_idx]
                # image
                image_name = os.path.join(self.base_dir, 'original_dataset',
                                          object_name, 'data',
                                          'color{}.jpg'.format(local_idx))
                image = cv2.imread(image_name)
                # mask
                mask_name = os.path.join(self.base_dir, 'masks',
                                         object_name, 'mask{}.png'.format(local_idx))
                mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
                # online occlusion
                if self.occlude:
                    image, mask = self.occlude_with_another_object(image, mask)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.float32(image) / 255.
                mask = np.float32(mask)
                mask[mask != 0.] = 1.
                # keypoints
                pts2d = self.pts2d[object_name][local_idx]
                pts3d = self.pts3d[object_name]
                # symmetry correspondences
                sym_cor_name = os.path.join(self.base_dir, 'correspondences',
                                            object_name, 'cor{}.npy'.format(local_idx))
                sym_cor = np.float32(np.load(sym_cor_name))
                normal = self.normals[object_name]
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
                # pose
                R_name = os.path.join(self.base_dir, 'original_dataset', object_name,
                                      'data', 'rot{}.rot'.format(local_idx))
                R = self.read_rotation(R_name)
                t_name = os.path.join(self.base_dir, 'original_dataset', object_name,
                                      'data', 'tra{}.tra'.format(local_idx))
                t = self.read_translation(t_name)
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
                    return {
                            'object_name': object_name,
                            'local_idx': local_idx,
                            'image_name': image_name,
                            'image': image,
                            'pts2d': pts2d,
                            'pts2d_map': pts2d_map,
                            'pts3d': pts3d,
                            'R': R,
                            't': t,
                            'sym_cor': sym_cor,
                            'normal': normal,
                            'mask': mask,
                            'graph': graph
                            }
            else:
                local_idx -= self.lengths[object_name]
        raise ValueError('Invalid index: {}'.format(idx))

if __name__ == '__main__':
    linemod_objects = ['ape', 'benchviseblue', 'cam', 'can', 'cat',
                       'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                       'iron', 'lamp', 'phone']
    for name in linemod_objects:
        dataset = LinemodDataset(object_name=name, augment=False, occlude=False, split='test')
        print(name, len(dataset))
