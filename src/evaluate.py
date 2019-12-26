import _init_paths
import argparse
import numpy as np
import glob
from lib.utils import compute_add_score, compute_adds_score
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='occlusion_linemod', choices=['linemod', 'occlusion_linemod'])
    parser.add_argument('--object_name', type=str, default='ape')
    parser.add_argument('--prediction_file', type=str, default='output/occlusion_linemod/record_ape.npy')
    args = parser.parse_args()
    return args


def read_3d_points_linemod(object_name):
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

def read_3d_points_occlusion(object_name):
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

def read_diameter(object_name):
    # this is the same for linemod and occlusion linemod
    filename = 'data/linemod/original_dataset/{}/distance.txt'.format(object_name)
    with open(filename) as f:
        diameter_in_cm = float(f.readline())
    return diameter_in_cm * 0.01

# main function
if __name__ == '__main__':
    args = parse_args()
    record = np.load(args.prediction_file, allow_pickle=True).item()
    if args.dataset == 'linemod':
        read_3d_points = read_3d_points_linemod
    elif args.dataset == 'occlusion_linemod':
        read_3d_points = read_3d_points_occlusion
    pts3d = read_3d_points(args.object_name)
    diameter = read_diameter(args.object_name)
    if args.object_name in ['eggbox', 'glue']:
        compute_score = compute_adds_score
    else:
        compute_score = compute_add_score
    
    #jump data of validation set
    num = 0
    for t in record['t_pred']:
        if np.sum(t) == 0:
            num += 1
    
    score_init = compute_score(pts3d,
                               diameter,
                               (record['R_gt'][num:], record['t_gt'][num:]),
                               (record['R_init'][num:], record['t_init'][num:]))
    print('ADD(-S) score of initial prediction is: {}'.format(score_init))
    score_pred = compute_score(pts3d,
                               diameter,
                               (record['R_gt'][num:], record['t_gt'][num:]),
                               (record['R_pred'][num:], record['t_pred'][num:]))
    print('ADD(-S) score of final prediction is: {}'.format(score_pred))
