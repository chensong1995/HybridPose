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

class OcclusionLinemodDataset(Dataset):
    def __init__(self,
                 base_dir='data/occlusion_linemod',
                 object_name='all'):
        self.camera_intrinsic = {'fu': 572.41140, 'fv': 573.57043,
                                 'uc': 325.26110, 'vc': 242.04899}
        self.K = np.matrix([[self.camera_intrinsic['fu'], 0, self.camera_intrinsic['uc']],
                            [0, self.camera_intrinsic['fv'], self.camera_intrinsic['vc']],
                            [0, 0, 1]], dtype=np.float32)
        self.img_shape = (480, 640) # (h, w)
        # use alignment_flipping to correct pose labels
        self.alignment_flipping = np.matrix([[1., 0., 0.],
                                             [0., -1., 0.],
                                             [0., 0., -1.]], dtype=np.float32)
        self.base_dir = base_dir
        linemod_objects = ['ape', 'can', 'cat', 'driller',
                           'duck', 'eggbox', 'glue', 'holepuncher']
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
            length = len(list(filter(lambda x: x.endswith('txt'),
                                     os.listdir(os.path.join(base_dir, 'valid_poses', object_name)))))
            self.lengths[object_name] = length
            self.total_length += length
        # pre-load data into memory
        self.pts3d = {}
        self.normals = {}
        self.R_lo = {}
        self.t_lo = {}
        for object_name in self.object_names:
            # transformations
            self.R_lo[object_name], self.t_lo[object_name] = \
                    self.get_linemod_to_occlusion_transformation(object_name)
            # keypoints
            pts3d_name = os.path.join('data/linemod', 'keypoints',
                                      object_name, 'keypoints_3d.npy')
            pts3d = np.float32(np.load(pts3d_name))
            self.pts3d[object_name] = pts3d
            # symmetry plane normals
            normal_name = os.path.join('data/linemod', 'symmetries',
                                       object_name, 'symmetries.txt')
            self.normals[object_name] = self.read_normal(normal_name)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.total_length

    def get_linemod_to_occlusion_transformation(self, object_name):
        # https://github.com/ClayFlannigan/icp
        if object_name == 'ape':
            R = np.array([[-4.5991463e-08, -1.0000000e+00,  1.1828890e-08],
                          [ 8.5046146e-08, -4.4907327e-09, -1.0000000e+00],
                          [ 1.0000000e+00, -2.7365417e-09,  9.5073148e-08]], dtype=np.float32)
            t = np.array([ 0.00464956, -0.04454319, -0.00454451], dtype=np.float32)
        elif object_name == 'can':
            R = np.array([[ 1.5503679e-07, -1.0000000e+00,  2.0980373e-07],
                          [ 2.6769550e-08, -2.0030792e-07, -1.0000000e+00],
                          [ 1.0000000e+00,  1.5713613e-07,  2.8610597e-08]], dtype=np.float32)
            t = np.array([-0.009928,   -0.08974387, -0.00697199], dtype=np.float32)
        elif object_name == 'cat':
            R = np.array([[-7.1956642e-08, -1.0000000e+00, -7.8242387e-08],
                          [-9.9875002e-08,  6.7945813e-08, -1.0000000e+00],
                          [ 1.0000000e+00, -6.8791721e-08, -1.0492791e-07]], dtype=np.float32)
            t = np.array([-0.01460595, -0.05390565,  0.00600646], dtype=np.float32)
        elif object_name == 'driller':
            R = np.array([[-5.8952626e-08, -9.9999994e-01,  1.7797127e-07],
                          [ 6.7603776e-09, -1.7821345e-07, -1.0000000e+00],
                          [ 9.9999994e-01, -5.8378635e-08,  2.7301144e-08]], dtype=np.float32)
            t = np.array([-0.00176942, -0.10016585,  0.00840302], dtype=np.float32)
        elif object_name == 'duck':
            R = np.array([[-3.4352450e-07, -1.0000000e+00,  4.5238485e-07],
                          [-6.4654046e-08, -4.5092108e-07, -1.0000000e+00],
                          [ 1.0000000e+00, -3.4280166e-07, -4.6047357e-09]], dtype=np.float32)
            t = np.array([-0.00285449, -0.04044429,  0.00110274], dtype=np.float32)
        elif object_name == 'eggbox':
            R = np.array([[-0.02, -1.00, 0.00],
                          [-0.02, -0.00, -1.00],
                          [1.00, -0.02, -0.02]], dtype=np.float32)
            t = np.array([-0.01, -0.03, -0.00], dtype=np.float32)
        elif object_name == 'glue':
            R = np.array([[-1.2898508e-07, -1.0000000e+00,  6.7859062e-08],
                          [ 2.9789486e-08, -6.8855734e-08, -9.9999994e-01],
                          [ 1.0000000e+00, -1.2711939e-07,  2.9696672e-08]], dtype=np.float32)
            t = np.array([-0.00144855, -0.07744411, -0.00468425], dtype=np.float32)
        elif object_name == 'holepuncher':
            R = np.array([[-5.9812328e-07, -9.9999994e-01,  3.9026276e-07],
                          [ 8.9670505e-07, -3.8723923e-07, -1.0000001e+00],
                          [ 1.0000000e+00, -5.9914004e-07,  8.8171902e-07]], dtype=np.float32)
            t = np.array([-0.00425799, -0.03734197,  0.00175619], dtype=np.float32)
        t = t.reshape((3, 1))
        return R, t

    def read_pose_and_img_id(self, filename, example_id):
        read_rotation = False
        read_translation = False
        R = []
        T = []
        with open(filename) as f:
            for line in f:
                if read_rotation:
                    R.append(line.split())
                    if len(R) == 3:
                        read_rotation = False
                elif read_translation:
                    T = line.split()
                    read_translation = False
                if line.startswith('rotation'):
                    read_rotation = True
                elif line.startswith('center'):
                    read_translation = True
        R = np.array(R, dtype=np.float32) # 3*3
        T = np.array(T, dtype=np.float32).reshape((3, 1)) # 3*1
        img_id = int(line) # in the last line
        return R, T, img_id

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

    def read_3d_points_linemod(self, object_name):
        filename = 'data/linemod/original_dataset/{}/mesh.ply'.format(object_name)
        with open(filename) as f:
            in_vertex_list = False
            vertices = []
            in_mm = False
            for line in f:
                if in_vertex_list:
                    vertex = line.split()[:3]
                    vertex = np.array([float(vertex[0]),
                                       float(vertex[1]),
                                       float(vertex[2])], dtype=np.float32)
                    if in_mm:
                        vertex = vertex / np.float32(10) # mm -> cm
                    vertex = vertex / np.float32(100)
                    vertices.append(vertex)
                    if len(vertices) >= vertex_count:
                        break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    in_vertex_list = True
                elif line.startswith('element face'):
                    in_mm = True
        return np.matrix(vertices)

    def read_3d_points_occlusion(self, object_name):
        import glob
        filename = glob.glob('data/occlusion_linemod/models/{}/*.xyz'.format(object_name))[0]
        with open(filename) as f:
            vertices = []
            for line in f:
                vertex = line.split()[:3]
                vertex = np.array([float(vertex[0]),
                                   float(vertex[1]),
                                   float(vertex[2])], dtype=np.float32)
                vertices.append(vertex)
        vertices = np.matrix(vertices)
        return vertices

    def __getitem__(self, idx):
        local_idx = idx
        for object_name in self.object_names:
            if local_idx < self.lengths[object_name]:
                # pose
                pose_name = os.path.join(self.base_dir, 'valid_poses', object_name, '{}.txt'.format(local_idx))
                R, t, img_id = self.read_pose_and_img_id(pose_name, local_idx)
                R = np.array(self.alignment_flipping * R, dtype=np.float32)
                t = np.array(self.alignment_flipping * t, dtype=np.float32)
                # apply linemod->occlusion alignment
                t = np.matmul(R, self.t_lo[object_name]) + t
                R = np.matmul(R, self.R_lo[object_name])
                # image
                image_name = os.path.join(self.base_dir, 'RGB-D', 'rgb_noseg', 'color_{:05d}.png'.format(img_id))
                image = self.img_transform(PIL.Image.open(image_name).convert('RGB'))
                # keypoints
                pts3d = self.pts3d[object_name]
                pts2d = self.K * (np.matrix(R) * np.matrix(pts3d).transpose() + np.matrix(t))
                pts2d = np.array(pts2d)
                pts2d[0] /= pts2d[2]
                pts2d[1] /= pts2d[2]
                pts2d = pts2d[[0, 1]].transpose()
                # symmetry correspondences (this is invalid. do not use)
                sym_cor_name = os.path.join(self.base_dir, 'my_labels', object_name, 'cor', '{}.npy'.format(local_idx))
                sym_cor = np.float32(np.load(sym_cor_name)).transpose([2, 0, 1])
                normal = self.normals[object_name]
                # mask
                mask_name = os.path.join(self.base_dir, 'masks', object_name, '{}.png'.format(img_id))
                mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
                mask = np.float32(mask).reshape((1, mask.shape[0], mask.shape[1]))
                mask[mask != 0.] = 1.
                # keypoint map
                pts2d_map = self.keypoints_to_map(mask, pts2d)
                # graph
                graph = self.keypoints_to_graph(mask, pts2d)
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
    linemod_objects = ['ape', 'can', 'cat', 'driller',
                       'duck', 'eggbox', 'glue', 'holepuncher']
    for name in linemod_objects:
        dataset = OcclusionLinemodDataset(object_name=name)
        print(name, len(dataset))
