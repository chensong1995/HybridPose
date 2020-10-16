import _init_paths
import argparse
import numpy as np
import glob
from lib.utils import compute_add_score, compute_adds_score
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_name', type=str, default='ape')
    parser.add_argument('--prediction_file', type=str, default='output/occlusion_linemod/test_set_ape_40.npy')
    args = parser.parse_args()
    return args

def read_3d_points(object_name):
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
    pts3d = read_3d_points(args.object_name)
    diameter = read_diameter(args.object_name)
    if args.object_name in ['eggbox', 'glue']:
        compute_score = compute_adds_score
    else:
        compute_score = compute_add_score
    score_init = compute_score(pts3d,
                               diameter,
                               (record['R_gt'], record['t_gt']),
                               (record['R_init'], record['t_init']))
    print('ADD(-S) score of initial prediction is: {}'.format(score_init))
    score_pred = compute_score(pts3d,
                               diameter,
                               (record['R_gt'], record['t_gt']),
                               (record['R_pred'], record['t_pred']))
    print('ADD(-S) score of final prediction is: {}'.format(score_pred))
