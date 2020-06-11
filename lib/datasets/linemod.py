import PIL.Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset
import pdb

cuda = torch.cuda.is_available()

class LinemodDataset(Dataset):
    def __init__(self,
                 base_dir='data/linemod',
                 object_name='all'):
        self.camera_intrinsic = {'fu': 572.41140, 'fv': 573.57043,
                                 'uc': 325.26110, 'vc': 242.04899}
        self.K = np.matrix([[self.camera_intrinsic['fu'], 0, self.camera_intrinsic['uc']],
                            [0, self.camera_intrinsic['fv'], self.camera_intrinsic['vc']],
                            [0, 0, 1]], dtype=np.float32)
        self.img_shape = (480, 640) # (h, w)
        self.base_dir = base_dir
        linemod_objects = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                           'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                           'iron', 'lamp', 'phone']
        if object_name == 'all':
            self.object_names = linemod_objects
        elif object_name in linemod_objects:
            self.object_names = [object_name]
        else:
            raise ValueError('Invalid object name: {}'.format(object_name))
        # compute length
        self.lengths = {}
        self.total_length = 0
        for object_name in self.object_names:
            length = len(list(filter(lambda x: x.endswith('jpg'),
                                     os.listdir(os.path.join(base_dir, 'original_dataset',object_name, 'data')))))
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
        graph = np.zeros((num_edges, 2, self.img_shape[0], self.img_shape[1]),
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
        graph = graph.reshape((num_edges * 2, self.img_shape[0], self.img_shape[1]))
        return graph

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:
            if local_idx < self.lengths[object_name]:
                # image
                image_name = os.path.join(self.base_dir, 'original_dataset',
                                          object_name, 'data',
                                          'color{}.jpg'.format(local_idx))
                image = transforms.ToTensor()(PIL.Image.open(image_name).convert('RGB'))
                # keypoints
                pts2d = self.pts2d[object_name][local_idx]
                pts3d = self.pts3d[object_name]
                # pose
                R_name = os.path.join(self.base_dir, 'original_dataset', object_name,
                                      'data', 'rot{}.rot'.format(local_idx))
                R = self.read_rotation(R_name)
                t_name = os.path.join(self.base_dir, 'original_dataset', object_name,
                                      'data', 'tra{}.tra'.format(local_idx))
                t = self.read_translation(t_name)
                # symmetry correspondences
                sym_cor_name = os.path.join(self.base_dir, 'correspondences',
                                            object_name, 'cor{}.npy'.format(local_idx))
                sym_cor = np.float32(np.load(sym_cor_name)).transpose([2, 0, 1])
                normal = self.normals[object_name]
                # mask
                mask_name = os.path.join(self.base_dir, 'masks',
                                         object_name, 'mask{}.png'.format(local_idx))
                mask = transforms.ToTensor()(PIL.Image.open(mask_name).convert('L'))
                # keypoint map
                pts2d_map = self.keypoints_to_map(mask.numpy(), pts2d)
                # graph
                graph = self.keypoints_to_graph(mask.numpy(), pts2d)
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
    ape_dataset = LinemodDataset(object_name='ape')
    ape_dataset[0]
    pdb.set_trace()
    all_dataset = LinemodDataset(object_name='all')
