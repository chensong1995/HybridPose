import argparse
import os
import cv2
import numpy as np
import pickle
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fx', type=float, default=700.)
    parser.add_argument('--fy', type=float, default=700.)
    parser.add_argument('--px', type=float, default=320.)
    parser.add_argument('--py', type=float, default=240.)
    parser.add_argument('--img_h', type=int, default=480)
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--object_name', type=str, default='ape')
    parser.add_argument('--orig_path', type=str, default='../../linemod')
    parser.add_argument('--pvnet_linemod_path', type=str, default='..')
    args = parser.parse_args()
    return args

linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchvise',
                     'can', 'driller', 'eggbox', 'holepuncher', 'lamp']


def get_num_examples(object_name):
    return len(list(filter(lambda x: x.endswith('jpg'), os.listdir(object_name))))

def read_pose(object_name, example_id):
    filename = os.path.join(object_name, '{}_RT.pkl'.format(example_id))
    with open(filename, 'rb') as f:
        pkl = pickle.load(f)
        RT = pkl['RT']
        R = np.matrix(RT[:, :3])
        T = np.matrix(RT[:, -1]).transpose()
    return R, T

def read_3d_points(filename, skip_normalize=False):
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
                if not skip_normalize:
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
    return vertices, in_mm

def parse_symmetry(filename):
    with open(filename) as f:
        lines = f.readlines()
        point = lines[1].split()
        point = (float(point[0]), float(point[1]), float(point[2]))
        normal = lines[3].split()
        normal = (float(normal[0]), float(normal[1]), float(normal[2]))
    return point, normal

def get_camera_intrinsic_matrix(args):
    return np.matrix([[args.fx, 0, args.px],
                      [0, args.fy, args.py],
                      [0, 0, 1]], dtype=np.float32)

def nearest_nonzero_idx_v2(a, x, y):
    # https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-array
    # x: (N,)
    # y: (N,)
    tmp = a[x, y]
    a[x, y] = 0
    r, c = np.nonzero(a)
    r = r.reshape((1, -1)).repeat(x.shape[0], axis=0)
    c = c.reshape((1, -1)).repeat(x.shape[0], axis=0)
    a[x, y] = tmp
    min_idx = ((r - x.reshape((-1, 1))) ** 2 + (c - y.reshape((-1, 1))) ** 2).argmin(axis=1)
    return np.array([r[0, min_idx], c[0, min_idx]]).transpose()

def fill(im_in):
    # based on https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out

def main(args):
    K = get_camera_intrinsic_matrix(args)
    P_list, in_mm = read_3d_points(os.path.join(args.orig_path, 'original_dataset', args.object_name, 'mesh.ply'))
    P_blender_list, _ = read_3d_points(os.path.join(args.pvnet_linemod_path, args.object_name, '{}.ply'.format(args.object_name)), skip_normalize=True)
    O, n = parse_symmetry(os.path.join(args.orig_path, 'symmetries', args.object_name, 'symmetries.txt'))
    if in_mm: # convert to m
        O = (O[0] / 1000, O[1] / 1000, O[2] / 1000)

    # for each 3D point P, find its correspondence P'
    P_prime_list = []
    for P_idx, P in enumerate(P_list):
        PO = (O[0] - P[0], O[1] - P[1], O[2] - P[2])
        dot_product = PO[0] * n[0] + PO[1] * n[1] + PO[2] * n[2]
        P_prime = (P[0] + 2 * dot_product * n[0], P[1] + 2 * dot_product * n[1], P[2] + 2 * dot_product * n[2])
        P_prime_list.append(P_prime)
    P_prime_list = np.array(P_prime_list)
 
    def project(P, R, T):
        P_RT = R * P + T
        p = K * P_RT
        x = int(round(p[0, 0] / p[2, 0]))
        y = int(round(p[1, 0] / p[2, 0]))
        return (x, y, P_RT[2, 0])

    num_examples = get_num_examples(args.object_name)
    keypts_3d = np.load(os.path.join(args.orig_path, 'keypoints', args.object_name, 'keypoints_3d.npy'))
    keypts_2d = np.zeros((num_examples, keypts_3d.shape[0], 2), dtype=np.float32)

    rotation_transform = np.matrix([[1., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., -1.]])
    # get translation transform
    # https://github.com/zju3dv/pvnet/blob/abc3f07cfcf352df3d718f10944213e1cde02db1/lib/utils/base_utils.py#L110
    orig_model = np.array(P_list)[:, :, 0]
    blender_model = np.array(P_blender_list)[:, :, 0]
    blender_model = np.dot(blender_model, rotation_transform.T)
    translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
    translation_transform = translation_transform.transpose()

    os.makedirs('{}_labels'.format(args.object_name), exist_ok=True)

    for example_id in range(num_examples):
        print('example {}/{}'.format(example_id, num_examples), end='\r')
        R, T = read_pose(args.object_name, example_id)
        R = np.dot(R, rotation_transform)
        T = T - np.dot(R, translation_transform)

        # project 3D correspondeces to 2D
        correspondences = np.zeros((args.img_h, args.img_w, 2), dtype=np.int16)
        z_buffer = np.zeros((args.img_h, args.img_w), dtype=np.float32)
        is_filled = np.zeros((args.img_h, args.img_w), dtype=np.uint8)
        sample = example_id == 0
        if sample:
            img = cv2.imread('{}/{}.jpg'.format(args.object_name, example_id))
        mask = np.zeros((args.img_h, args.img_w), dtype=np.uint8)
        for P_idx, P in enumerate(P_list):
            P_prime = P_prime_list[P_idx]
            (x, y, z) = project(P, R, T)
            (x_prime, y_prime, _) = project(P_prime, R, T)
            if y >= 0 and y < args.img_h and x >= 0 and x < args.img_w:
                if is_filled[y, x] == 0 or z_buffer[y, x] > z:
                    # I did a simple experiment: a smaller z is closer to the camera than a bigger z
                    is_filled[y, x] = 1
                    z_buffer[y, x] = z
                    delta_x = x_prime - x
                    delta_y = y_prime - y
                    correspondences[y, x, 0] = delta_x
                    correspondences[y, x, 1] = delta_y
                    if sample and P_idx % 50 == 0:
                        # 1 sample every 50 points
                        # color: red, thickness: 1
                        img = cv2.line(img, (x, y), (x_prime, y_prime), (0, 0, 255), 1)
                    mask[y, x] = 255
        mask = fill(mask)
        cv2.imwrite('{}_labels/mask{}.png'.format(args.object_name, example_id), mask)
        yx = np.argwhere((mask != 0) & (is_filled == 0.))
        if yx.shape[0] > 0:
            yx_ = nearest_nonzero_idx_v2(is_filled, yx[:, 0], yx[:, 1])
            for i in range(yx.shape[0]):
                y, x = yx[i]
                y_, x_ = yx_[i]
                correspondences[y, x] = correspondences[y_, x_]
        np.save('{}_labels/cor{}.npy'.format(args.object_name, example_id),
                correspondences)
        if sample:
            cv2.imwrite('cor_{}.jpg'.format(args.object_name), img)

        # project 3D keypoints to 2D
        for keypt_idx in range(keypts_3d.shape[0]):
            (x, y, _) = project(keypts_3d[keypt_idx].reshape((3, 1)), R, T)
            keypts_2d[example_id, keypt_idx, 0] = x
            keypts_2d[example_id, keypt_idx, 1] = y
    np.save(os.path.join('{}_labels'.format(args.object_name), 'keypoints_2d.npy'), keypts_2d)

if __name__ == '__main__':
    args = parse_args()
    main(args)
