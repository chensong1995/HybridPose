import os
import pdb
import torch
import numpy as np
from sklearn.neighbors import KDTree
from scipy.linalg import logm
from numpy import linalg as LA

def save_session(model, optim, save_dir, note, epoch):
    # note0 loss note1 lr
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, note, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model into {}'.format(path))

def load_session(model, optim, args):
    try:
        start_epoch = int(args.load_dir.split('/')[-1]) + 1
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        for param_group in optim.param_groups:
            param_group['lr'] = args.lr
        print('Successfully loaded model from {}'.format(args.load_dir))
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly, check the load_dir')

    return model, optim, start_epoch

def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]        
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)            

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score

def compute_pose_error(diameter, pose_gt, pose_pred):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    R_err = np.zeros(count)
    t_err = np.zeros(count)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            continue
        r_err = logm(np.dot(R_pred[i].transpose(), R_gt[i])) / 2
        R_err[i] = LA.norm(r_err, 'fro')
        t_err[i] = LA.norm(t_pred[i] - t_gt[i])
    return np.median(R_err) * 180 / np.pi, np.median(t_err) / diameter

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count

