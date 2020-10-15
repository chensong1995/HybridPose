import torch
import torch.nn.functional as F
import time
import os
import cv2
import numpy as np
from sklearn.neighbors import KDTree
from random import sample
from lib.utils import save_session, AverageMeter
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import ransac_voting_layer_v3
from lib.ransac_voting_gpu_layer.ransac_voting_gpu import estimate_voting_distribution_with_mean
from lib.regressor.regressor import load_wrapper, get_2d_ctypes
from src.evaluate import read_diameter
import pdb

cuda = torch.cuda.is_available()

class CoreTrainer(object):
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        super(CoreTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.args = args

    def train(self, epoch):
        self.model.train()
        time_record = AverageMeter()
        sym_cor_loss_record = AverageMeter()
        mask_loss_record = AverageMeter()
        pts2d_loss_record = AverageMeter()
        graph_loss_record = AverageMeter()
        total_loss_record = AverageMeter()

        for i_batch, batch in enumerate(self.train_loader):
            start_time = time.time()
            if cuda:
                batch['image'] = batch['image'].cuda()
                batch['sym_cor'] = batch['sym_cor'].cuda()
                batch['mask'] = batch['mask'].cuda()
                batch['pts2d_map'] = batch['pts2d_map'].cuda()
                batch['graph'] = batch['graph'].cuda()
            sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                    self.model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])

            # losses: move to the same device
            sym_cor_loss = sym_cor_loss.mean()
            mask_loss = mask_loss.mean()
            pts2d_loss = pts2d_loss.mean()
            graph_loss = graph_loss.mean()
            current_loss = self.args.lambda_sym_cor * sym_cor_loss + \
                           self.args.lambda_mask * mask_loss + \
                           self.args.lambda_pts2d * pts2d_loss + \
                           self.args.lambda_graph * graph_loss
            # Step optimizer
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()

            # print information during training
            time_record.update(time.time() - start_time)
            sym_cor_loss_record.update(sym_cor_loss.detach().cpu().numpy(), len(batch['image']))
            mask_loss_record.update(mask_loss.detach().cpu().numpy(), len(batch['image']))
            pts2d_loss_record.update(pts2d_loss.detach().cpu().numpy(), len(batch['image']))
            graph_loss_record.update(graph_loss.detach().cpu().numpy(), len(batch['image']))
            total_loss_record.update(current_loss.detach().cpu().numpy(), len(batch['image']))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {time.val:.3f} ({time.avg:.3f})\t'
                  'Sym: {sym.val:.4f} ({sym.avg:.4f})\t'
                  'Mask: {mask.val:.4f} ({mask.avg:.4f})\t'
                  'Pts: {pts.val:.4f} ({pts.avg:.4f})\t'
                  'Graph: {graph.val:.4f} ({graph.avg:.4f})\t'
                  'Total: {total.val:.4f} ({total.avg:.4f})'.format(epoch, i_batch, len(self.train_loader),
                                                                    time=time_record, sym=sym_cor_loss_record,
                                                                    mask=mask_loss_record, pts=pts2d_loss_record,
                                                                    graph=graph_loss_record, total=total_loss_record))

    def visualize_symmetry(self, sym_cor_pred, mask_pred, sym_cor, mask, image, epoch, i_batch):
        img_dir = os.path.join(self.args.save_dir, 'image', str(self.args.lr))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # visualize prediction
        image_pred = image.copy()
        mask_pred = mask_pred.detach().cpu().numpy()[0]
        sym_cor_pred = sym_cor_pred.detach().cpu().numpy()
        ys, xs = np.nonzero(mask_pred)
        for i_pt in sample([i for i in range(len(ys))], min(100, len(ys))):
            y = int(round(ys[i_pt]))
            x = int(round(xs[i_pt]))
            x_cor, y_cor = sym_cor_pred[:, y, x]
            x_cor = int(round(x + x_cor))
            y_cor = int(round(y + y_cor))
            image_pred = cv2.line(image_pred, (x, y), (x_cor, y_cor), (0, 0, 255), 1)
        img_pred_name = os.path.join(img_dir, '{}_{}_sym.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_pred_name, image_pred)
        # visualize ground truth
        image_gt = image.copy()
        mask = mask.detach().cpu().numpy()[0]
        sym_cor = sym_cor.detach().cpu().numpy()
        ys, xs = np.nonzero(mask)
        for i_pt in sample([i for i in range(len(ys))], min(100, len(ys))):
            y = int(round(ys[i_pt]))
            x = int(round(xs[i_pt]))
            x_cor, y_cor = sym_cor[:, y, x]
            x_cor = int(round(x + x_cor))
            y_cor = int(round(y + y_cor))
            image_gt = cv2.line(image_gt, (x, y), (x_cor, y_cor), (0, 0, 255), 1)
        img_gt_name = os.path.join(img_dir, '{}_{}_sym_gt.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_gt_name, image_gt)

    def visualize_mask(self, mask_pred, mask, epoch, i_batch):
        mask_pred = mask_pred.detach().cpu().numpy()[0]
        mask = np.uint8(mask.detach().cpu().numpy()[0])
        image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # red: prediction
        image[mask_pred == 1.] += np.array([0, 0, 128], dtype=np.uint8)
        # blue: gt
        image[mask != 0] += np.array([128, 0, 0], dtype=np.uint8)
        img_dir = os.path.join(self.args.save_dir, 'image', str(self.args.lr))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img_name = os.path.join(img_dir, '{}_{}_mask.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_name, image)

    def visualize_keypoints(self, pts2d_map_pred, pts2d, mask_pred, image, epoch, i_batch):
        img_dir = os.path.join(self.args.save_dir, 'image', str(self.args.lr))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        # vote keypoints
        pts2d_pred, _ = self.vote_keypoints(pts2d_map_pred, mask_pred)
        pts2d_pred = pts2d_pred.detach().cpu().numpy()[0]
        # draw predication
        image_pred = image.copy()
        for i in range(pts2d_pred.shape[0]):
            x, y = pts2d_pred[i]
            x = int(round(x))
            y = int(round(y))
            # radius=2, color=red, thickness=filled
            image_pred = cv2.circle(image_pred, (x, y), 2, (0, 0, 255), thickness=-1)
        img_pred_name = os.path.join(img_dir, '{}_{}_pts.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_pred_name, image_pred)
        # draw ground truth
        pts2d = pts2d.detach().cpu().numpy()
        image_gt = image.copy()
        for i in range(pts2d.shape[0]):
            x, y = pts2d[i]
            x = int(round(x))
            y = int(round(y))
            # radius=2, color=white, thickness=filled
            image_gt = cv2.circle(image_gt, (x, y), 2, (255, 255, 255), thickness=-1)
        img_gt_name = os.path.join(img_dir, '{}_{}_pts_gt.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_gt_name, image_gt)

    def visualize_votes(self, map_pred, map_gt, mask_gt, epoch, i_batch):
        img_dir = os.path.join(self.args.save_dir, 'image', str(self.args.lr))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        map_pred = map_pred.detach().cpu().numpy()
        map_gt = map_gt.detach().cpu().numpy()
        mask = np.uint8(mask_gt.detach().cpu().numpy()[0])
        image = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        ys, xs = np.nonzero(mask)
        # visualize pred
        images_pred = [image.copy() for _ in range(map_pred.shape[0] // 2)]
        for i_pt in range(len(ys)):
            for i_keypt in range(map_pred.shape[0] // 2):
                y = ys[i_pt]
                x = xs[i_pt]
                map_x = map_pred[i_keypt * 2, y, x]
                map_y = map_pred[i_keypt * 2 + 1, y, x]
                if map_x == 0:
                    continue
                angle = np.arctan(np.abs(map_y) / np.abs(map_x)) / (np.pi / 2) * 90
                if map_x < 0 and map_y > 0:
                    angle = 180 - angle
                if map_x < 0 and map_y < 0:
                    angle = 180 + angle
                if map_x >= 0 and map_y < 0:
                    angle = 360 - angle
                images_pred[i_keypt][y, x] = int(round(angle / 360 * 255))
        images_pred = [cv2.applyColorMap(im_gray, cv2.COLORMAP_HSV) for im_gray in images_pred]
        for i, im in enumerate(images_pred):
            im[mask == 0] = (0, 0, 0)
            img_pred_name = os.path.join(img_dir, '{}_{}_vote_kp_{}_pred.jpg'.format(epoch, i_batch, i))
            cv2.imwrite(img_pred_name, im)
        # visualize gt
        images_gt = [image.copy() for _ in range(map_gt.shape[0] // 2)]
        for i_pt in range(len(ys)):
            for i_keypt in range(map_gt.shape[0] // 2):
                y = ys[i_pt]
                x = xs[i_pt]
                map_x = map_gt[i_keypt * 2, y, x]
                map_y = map_gt[i_keypt * 2 + 1, y, x]
                if map_x == 0:
                    continue
                angle = np.arctan(np.abs(map_y) / np.abs(map_x)) / (np.pi / 2) * 90
                if map_x < 0 and map_y > 0:
                    angle = 180 - angle
                if map_x < 0 and map_y < 0:
                    angle = 180 + angle
                if map_x >= 0 and map_y < 0:
                    angle = 360 - angle
                images_gt[i_keypt][y, x] = int(round(angle / 360 * 255))
        images_gt = [cv2.applyColorMap(im_gray, cv2.COLORMAP_HSV) for im_gray in images_gt]
        for i, im in enumerate(images_gt):
            im[mask == 0] = (0, 0, 0)
            img_gt_name = os.path.join(img_dir, '{}_{}_vote_kp_{}_gt.jpg'.format(epoch, i_batch, i))
            cv2.imwrite(img_gt_name, im)

    def visualize_graph(self, graph_pred, graph_gt, pts2d_gt, mask_pred, mask_gt, image, epoch, i_batch):
        img_dir = os.path.join(self.args.save_dir, 'image', str(self.args.lr))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        image_gt = image.copy()
        image_pred = image.copy()
        graph_pred = graph_pred.detach().cpu().numpy()
        graph_pred = graph_pred.reshape((-1, 2, image.shape[0], image.shape[1]))
        graph_gt = graph_gt.detach().cpu().numpy()
        graph_gt = graph_gt.reshape((-1, 2, image.shape[0], image.shape[1]))
        pts2d_gt = pts2d_gt.numpy()
        mask_pred = mask_pred.detach().cpu().numpy()[0]
        mask_gt = mask_gt.detach().cpu().numpy()[0]
        num_pts = pts2d_gt.shape[0]
        i_edge = 0
        for start_idx in range(0, num_pts - 1):
            for end_idx in range(start_idx + 1, num_pts):
                # pred, red
                start = np.int16(np.round(pts2d_gt[start_idx]))
                edge_x = graph_pred[i_edge, 0][mask_pred == 1.].mean()
                edge_y = graph_pred[i_edge, 1][mask_pred == 1.].mean()
                edge = np.array([edge_x, edge_y])
                end = np.int16(np.round(pts2d_gt[start_idx] + edge))
                image_pred = cv2.line(image_pred, tuple(start), tuple(end), (0, 0, 255), 1)
                # gt, green
                start = np.int16(np.round(pts2d_gt[start_idx]))
                edge_x = graph_gt[i_edge, 0][mask_gt == 1.].mean()
                edge_y = graph_gt[i_edge, 1][mask_gt == 1.].mean()
                edge = np.array([edge_x, edge_y])
                end = np.int16(np.round(pts2d_gt[start_idx] + edge))
                image_gt = cv2.line(image_gt, tuple(start), tuple(end), (0, 255, 0), 1)
                i_edge += 1
        img_gt_name = os.path.join(img_dir, '{}_{}_gt_graph.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_gt_name, image_gt)
        img_pred_name = os.path.join(img_dir, '{}_{}_pred_graph.jpg'.format(epoch, i_batch))
        cv2.imwrite(img_pred_name, image_pred)

    def test(self, epoch):
        print('Testing...')
        self.model.eval()
        loss_record = AverageMeter()
        data_loader = self.test_loader
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                if cuda:
                    batch['image'] = batch['image'].cuda()
                    batch['sym_cor'] = batch['sym_cor'].cuda()
                    batch['mask'] = batch['mask'].cuda()
                    batch['pts2d_map'] = batch['pts2d_map'].cuda()
                    batch['graph'] = batch['graph'].cuda()
                sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                        self.model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
                mask_pred[mask_pred > 0.5] = 1.
                mask_pred[mask_pred <= 0.5] = 0.

                # losses: move to the same device
                sym_cor_loss = sym_cor_loss.mean()
                mask_loss = mask_loss.mean()
                pts2d_loss = pts2d_loss.mean()
                graph_loss = graph_loss.mean()
                current_loss = self.args.lambda_sym_cor * sym_cor_loss + \
                               self.args.lambda_mask * mask_loss + \
                               self.args.lambda_pts2d * pts2d_loss + \
                               self.args.lambda_graph * graph_loss
                if i_batch < 3:
                    # some visualizations
                    image = cv2.imread(batch['image_name'][0])
                    self.visualize_symmetry(sym_cor_pred[0],
                                            mask_pred[0],
                                            batch['sym_cor'][0],
                                            batch['mask'][0],
                                            image,
                                            epoch,
                                            i_batch)
                    self.visualize_mask(mask_pred[0],
                                        batch['mask'][0],
                                        epoch,
                                        i_batch)
                    self.visualize_votes(pts2d_map_pred[0],
                                         batch['pts2d_map'][0],
                                         batch['mask'][0],
                                         epoch,
                                         i_batch)
                    try:
                        self.visualize_keypoints(pts2d_map_pred[:1],
                                                 batch['pts2d'][0],
                                                 batch['mask'][:1],
                                                 image,
                                                 epoch,
                                                 i_batch)
                    except:
                        # we may not be able to vote keypoints at early stages
                        pass
                    self.visualize_graph(graph_pred[0],
                                         batch['graph'][0],
                                         batch['pts2d'][0],
                                         mask_pred[0],
                                         batch['mask'][0],
                                         image,
                                         epoch,
                                         i_batch)
                loss_record.update(current_loss.detach().cpu().numpy(), len(batch['image']))

        print('Loss: {:.4f}'.format(loss_record.avg))
        return loss_record.avg

    def vote_keypoints(self, pts2d_map, mask):
        mask = mask[:, 0] # remove dummy dimension
        mask = (mask > 0.5).long() # convert to binary and int64 to comply with pvnet interface
        pts2d_map = pts2d_map.permute((0, 2, 3, 1))
        bs, h, w, num_keypts_2 = pts2d_map.shape
        pts2d_map = pts2d_map.view((bs, h, w, num_keypts_2 // 2, 2))
        mean = ransac_voting_layer_v3(mask, pts2d_map, 512, inlier_thresh=0.99)
        mean, var = estimate_voting_distribution_with_mean(mask, pts2d_map, mean)
        return mean, var

    def flatten_sym_cor(self, sym_cor, mask):
        ys, xs = np.nonzero(mask)
        flat = np.zeros((ys.shape[0], 2, 2), dtype=np.float32)
        for i_pt in range(len(ys)):
            y = ys[i_pt]
            x = xs[i_pt]
            x_cor, y_cor = sym_cor[:, y, x]
            flat[i_pt, 0] = [x, y]
            flat[i_pt, 1] = [x + x_cor, y + y_cor]
        return flat

    def filter_symmetry(self, vecs_pred, sigma=0.01, min_count=100, n_neighbors=100):
        # Chen: I have to set min_count >= neighbors here.
        #       Otherwise kdtree will complain "k must be less than or equal to the number of training points"
        if len(vecs_pred) < min_count:
            qs1_cross_qs2 = np.zeros((0, 3), dtype=np.float32)
            symmetry_weight = np.zeros((0,), dtype=np.float32)
            return qs1_cross_qs2, symmetry_weight
        vecs_pred /= np.sqrt(np.sum(vecs_pred[:, :2]**2, axis=1)).reshape((-1, 1))
        kdt = KDTree(vecs_pred, leaf_size=40, metric='euclidean') # following matlab default values
        dis, _ = kdt.query(vecs_pred, k=n_neighbors)
        saliency = np.mean(dis * dis, axis=1, dtype=np.float32)
        order = np.argsort(saliency)
        seeds = np.zeros((2, order.shape[0]), dtype=np.uint32)
        seeds[0][0] = order[0]
        seeds[1][0] = 1
        seeds_size = 1
        flags = np.zeros((order.shape[0],), dtype=np.uint32)
        flags[order[0]] = 0
        for i in range(1, order.shape[0]):
            vec = vecs_pred[order[i]]
            candidates = vecs_pred[seeds[0]]
            dif = candidates - vec
            norm = np.linalg.norm(dif, axis=1)
            closest_seed_i = norm.argmin()
            min_dis = norm[closest_seed_i]
            if min_dis < sigma:
                flags[order[i]] = closest_seed_i
                seeds[1][closest_seed_i] = seeds[1][closest_seed_i] + 1
            else:
                seeds[0, seeds_size] = order[i]
                seeds[1, seeds_size] = 1
                flags[order[i]] = seeds_size
                seeds_size += 1
        seeds = seeds[:, :seeds_size]
        valid_is = np.argwhere(seeds[1] > (np.max(seeds[1]) / 3)).transpose()[0]
        seeds = seeds[:, valid_is]
        n_symmetry = seeds.shape[1]
        qs1_cross_qs2 = np.zeros((n_symmetry, 3), dtype=np.float32)
        for i in range(n_symmetry):            
            row_is = np.argwhere(flags == valid_is[i]).transpose()[0]
            qs1_cross_qs2[i] = np.mean(vecs_pred[row_is], axis=0)
            qs1_cross_qs2[i] /= np.linalg.norm(qs1_cross_qs2[i])
        symmetry_weight = np.float32(seeds[1])
        symmetry_weight /= np.max(symmetry_weight)
        return qs1_cross_qs2, symmetry_weight

    def fill_intermediate_predictions(self, regressor, predictions, K_inv, pts3d, pts2d_pred_loc, pts2d_pred_var, graph_pred, sym_cor_pred, mask_pred, normal_gt):
        # load intermediate representations to regressor
        n_keypts = self.args.num_keypoints
        n_edges = n_keypts * (n_keypts - 1) // 2
        # point3D_gt
        regressor.set_point3D_gt(predictions, get_2d_ctypes(pts3d), n_keypts)
        # point2D_pred
        point2D_pred = np.matrix(np.ones((3, n_keypts), dtype=np.float32))
        point2D_pred[:2] = pts2d_pred_loc.transpose()
        point2D_pred = np.array((K_inv * point2D_pred)[:2]).transpose()
        regressor.set_point2D_pred(predictions,
                                   get_2d_ctypes(point2D_pred),
                                   n_keypts)
        # point_inv_half_var
        point_inv_half_var = np.zeros((n_keypts, 2, 2), dtype=np.float32)
        for i in range(n_keypts): # compute cov^{-1/2}
            cov = np.matrix(pts2d_pred_var[i])
            cov = (cov + cov.transpose()) / 2 # ensure the covariance matrix is symmetric
            v, u = np.linalg.eig(cov)
            v = np.matrix(np.diag(1. / np.sqrt(v)))
            point_inv_half_var[i] = u * v * u.transpose()
        point_inv_half_var = point_inv_half_var.reshape((n_keypts, 4))
        regressor.set_point_inv_half_var(predictions,
                                         get_2d_ctypes(point_inv_half_var),
                                         n_keypts)
        # normal_gt
        regressor.set_normal_gt(predictions, normal_gt.ctypes)
        # vec_pred and edge_inv_half_var
        graph_pred = graph_pred.reshape((n_edges, 2, graph_pred.shape[1], graph_pred.shape[2]))
        vec_pred = np.zeros((n_edges, 2), dtype=np.float32)
        edge_inv_half_var = np.zeros((n_edges, 2, 2), dtype=np.float32)
        for i in range(n_edges):
            xs = graph_pred[i, 0][mask_pred == 1.]
            ys = graph_pred[i, 1][mask_pred == 1.]
            vec_pred[i] = [xs.mean(), ys.mean()]
            try:
                cov = np.cov(xs, ys)
                cov = (cov + cov.transpose()) / 2 # ensure the covariance matrix is symmetric
                v, u = np.linalg.eig(cov)
                v = np.matrix(np.diag(1. / np.sqrt(v)))
                edge_inv_half_var[i] = u * v * u.transpose()
            except:
                edge_inv_half_var[i] = np.eye(2)
        vec_pred = np.array(K_inv[:2, :2] * np.matrix(vec_pred).transpose()).transpose()
        edge_inv_half_var = edge_inv_half_var.reshape((n_edges, 4))
        regressor.set_vec_pred(predictions,
                               get_2d_ctypes(vec_pred),
                               n_edges)
        regressor.set_edge_inv_half_var(predictions,
                                        get_2d_ctypes(edge_inv_half_var),
                                        n_edges)
        # qs1_cross_qs2 and symmetry weight
        sym_cor_pred = self.flatten_sym_cor(sym_cor_pred, mask_pred)

        qs1_cross_qs2_all = np.zeros((sym_cor_pred.shape[0], 3), dtype=np.float32)
        for i in range(sym_cor_pred.shape[0]):
            qs1 = np.ones((3,), dtype=np.float32)
            qs2 = np.ones((3,), dtype=np.float32)
            qs1[:2] = sym_cor_pred[i][0]
            qs2[:2] = sym_cor_pred[i][1]
            qs1 = np.array(K_inv * np.matrix(qs1).transpose()).transpose()[0]
            qs2 = np.array(K_inv * np.matrix(qs2).transpose()).transpose()[0]
            qs1_cross_qs2_all[i] = np.cross(qs1, qs2)
        qs1_cross_qs2_filtered, symmetry_weight = self.filter_symmetry(qs1_cross_qs2_all)
        n_symmetry = qs1_cross_qs2_filtered.shape[0]
        regressor.set_qs1_cross_qs2(predictions,
                                    get_2d_ctypes(qs1_cross_qs2_filtered),
                                    n_symmetry)
        regressor.set_symmetry_weight(predictions,
                                      symmetry_weight.ctypes,
                                      n_symmetry)  

    def regress_pose(self, regressor, predictions, pr_para, pi_para, K_inv, pts3d, pts2d_pred_loc, pts2d_pred_var, graph_pred, sym_cor_pred, mask_pred, normal_gt):
        if mask_pred.sum() == 0:
            # object is not detected
            R = np.eye(3, dtype=np.float32)
            t = np.zeros((3, 1), dtype=np.float32)
            return R, t, R, t
        self.fill_intermediate_predictions(regressor,
                                           predictions,
                                           K_inv,
                                           pts3d,
                                           pts2d_pred_loc,
                                           pts2d_pred_var,
                                           graph_pred,
                                           sym_cor_pred,
                                           mask_pred,
                                           normal_gt)
        # initialize pose
        predictions = regressor.initialize_pose(predictions, pi_para)
        pose_init = np.zeros((4, 3), dtype=np.float32)
        regressor.get_pose(predictions, get_2d_ctypes(pose_init))
        R_init = pose_init[1:].transpose()
        t_init = pose_init[0].reshape((3, 1))
        # refine pose
        predictions = regressor.refine_pose(predictions, pr_para)
        pose_final = np.zeros((4, 3), dtype=np.float32)
        regressor.get_pose(predictions, get_2d_ctypes(pose_final))
        R_final = pose_final[1:].transpose()
        t_final = pose_final[0].reshape((3, 1))
        return R_final, t_final, R_init, t_init

    def search_para(self, regressor, predictions_para, poses_para, K_inv, normal_gt, diameter, val_set):
        para_id = 0
        for data_id in range(len(val_set['pts3d'])):
            if val_set['mask_pred'][data_id].sum() == 0 or \
                    np.sum(val_set['pts2d_pred_loc'][data_id]) == 0:
                # object not detected
                continue
            predictions = regressor.get_prediction_container(predictions_para, para_id)
            # fill intermediate predictions
            self.fill_intermediate_predictions(regressor,
                                               predictions,
                                               K_inv,
                                               val_set['pts3d'][data_id],
                                               val_set['pts2d_pred_loc'][data_id],
                                               val_set['pts2d_pred_var'][data_id],
                                               val_set['graph_pred'][data_id],
                                               val_set['sym_cor_pred'][data_id],
                                               val_set['mask_pred'][data_id],
                                               normal_gt)
            # fill ground-truth poses
            pose_gt = np.zeros((4, 3), dtype=np.float32)
            tvec = val_set['t_gt'][data_id]
            r = val_set['R_gt'][data_id]
            pose_gt[0] = tvec.transpose()[0]
            pose_gt[1:] = r.transpose()
            regressor.set_pose_gt(poses_para, para_id, get_2d_ctypes(pose_gt))
            # increment number of valid examples in the val set
            para_id += 1
        # search parameter
        # para_id is datasize for parameter search
        pi_para = regressor.search_pose_initial(predictions_para, poses_para, para_id, diameter)
        pr_para = regressor.search_pose_refine(predictions_para, poses_para, para_id, diameter)
        return pr_para, pi_para

    def generate_data(self, val_loader, val_size=50):
        self.model.eval()
        camera_intrinsic = self.test_loader.dataset.camera_intrinsic
        n_examples = len(self.test_loader.dataset)
        test_set = {
                'object_name': [],
                'local_idx': [],
                'R_gt': np.zeros((n_examples, 3, 3), dtype=np.float32),
                't_gt': np.zeros((n_examples, 3, 1), dtype=np.float32),
                'R_pred': np.zeros((n_examples, 3, 3), dtype=np.float32),
                't_pred': np.zeros((n_examples, 3, 1), dtype=np.float32),
                'R_init': np.zeros((n_examples, 3, 3), dtype=np.float32),
                't_init': np.zeros((n_examples, 3, 1), dtype=np.float32)
                }  
        val_set = {
                    'pts3d' : [],
                    'pts2d_pred_loc' : [],
                    'pts2d_pred_var' : [],
                    'graph_pred' : [],
                    'sym_cor_pred' : [],
                    'mask_pred' : [],
                    'R_gt' : [],
                    't_gt' : []
                    }
        K = np.matrix([[camera_intrinsic['fu'], 0, camera_intrinsic['uc']],
                       [0, camera_intrinsic['fv'], camera_intrinsic['vc']],
                       [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        regressor = load_wrapper()
        # intermediate predictions in the test set
        predictions = regressor.new_container()
        # intermediate predictions in the val set
        predictions_para = regressor.new_container_para()
        # ground-truth poses in the val set
        poses_para = regressor.new_container_pose()
        with torch.no_grad():
            # search parameters
            keep_searching = True
            for i_batch, batch in enumerate(val_loader):
                if not keep_searching:
                    break
                base_idx = self.args.batch_size * i_batch
                if cuda:
                    batch['image'] = batch['image'].cuda()
                    batch['sym_cor'] = batch['sym_cor'].cuda()
                    batch['mask'] = batch['mask'].cuda()
                    batch['pts2d_map'] = batch['pts2d_map'].cuda()
                    batch['graph'] = batch['graph'].cuda()
                sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                        self.model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
                mask_pred[mask_pred > 0.5] = 1.
                mask_pred[mask_pred <= 0.5] = 0.
                pts2d_pred_loc, pts2d_pred_var = self.vote_keypoints(pts2d_map_pred, mask_pred)
                mask_pred = mask_pred.detach().cpu().numpy()     
                for i in range(batch['image'].shape[0]):
                    R = batch['R'].numpy()
                    t = batch['t'].numpy()

                    if (base_idx + i) < val_size:
                        # save data for parameter search                    
                        val_set['pts3d'].append(batch['pts3d'][i].numpy())
                        val_set['pts2d_pred_loc'].append(pts2d_pred_loc[i].detach().cpu().numpy())
                        val_set['pts2d_pred_var'].append(pts2d_pred_var[i].detach().cpu().numpy())
                        val_set['graph_pred'].append(graph_pred[i].detach().cpu().numpy())
                        val_set['sym_cor_pred'].append(sym_cor_pred[i].detach().cpu().numpy())
                        val_set['mask_pred'].append(mask_pred[i][0]) 
                        val_set['R_gt'].append(R[i])
                        val_set['t_gt'].append(t[i])
                    elif (base_idx + i) == val_size:
                        # search hyper-parameters of both initialization and refinement sub-modules
                        pr_para, pi_para = self.search_para(regressor,
                                                            predictions_para,
                                                            poses_para,
                                                            K_inv,
                                                            batch['normal'][i].numpy(),
                                                            read_diameter(self.args.object_name),
                                                            val_set)
                        keep_searching = False
                        break
            # prediction
            for i_batch, batch in enumerate(self.test_loader):
                base_idx = self.args.batch_size * i_batch
                if cuda:
                    batch['image'] = batch['image'].cuda()
                    batch['sym_cor'] = batch['sym_cor'].cuda()
                    batch['mask'] = batch['mask'].cuda()
                    batch['pts2d_map'] = batch['pts2d_map'].cuda()
                    batch['graph'] = batch['graph'].cuda()
                sym_cor_pred, mask_pred, pts2d_map_pred, graph_pred, sym_cor_loss, mask_loss, pts2d_loss, graph_loss = \
                        self.model(batch['image'], batch['sym_cor'], batch['mask'], batch['pts2d_map'], batch['graph'])
                mask_pred[mask_pred > 0.5] = 1.
                mask_pred[mask_pred <= 0.5] = 0.
                pts2d_pred_loc, pts2d_pred_var = self.vote_keypoints(pts2d_map_pred, mask_pred)
                mask_pred = mask_pred.detach().cpu().numpy()
                for i in range(batch['image'].shape[0]):
                    R = batch['R'].numpy()
                    t = batch['t'].numpy()
                    # regress pose: test set starts from the `val_size`^{th} example
                    # save ground-truth information
                    test_set['object_name'] += batch['object_name'][i:]
                    test_set['local_idx'] += batch['local_idx'].numpy()[i:].tolist()
                    test_set['R_gt'][base_idx + i] = R[i]
                    test_set['t_gt'][base_idx + i] = t[i]
                    # save predicted information
                    R_pred, t_pred, R_init, t_init = self.regress_pose(regressor,
                                                                       predictions,
                                                                       pr_para,
                                                                       pi_para,
                                                                       K_inv,
                                                                       batch['pts3d'][i].numpy(),
                                                                       pts2d_pred_loc[i].detach().cpu().numpy(),
                                                                       pts2d_pred_var[i].detach().cpu().numpy(),
                                                                       graph_pred[i].detach().cpu().numpy(),
                                                                       sym_cor_pred[i].detach().cpu().numpy(),
                                                                       mask_pred[i][0],
                                                                       batch['normal'][i].numpy())
                    test_set['R_pred'][base_idx + i] = R_pred
                    test_set['t_pred'][base_idx + i] = t_pred
                    test_set['R_init'][base_idx + i] = R_init
                    test_set['t_init'][base_idx + i] = t_init
            os.makedirs('output/{}'.format(self.args.dataset), exist_ok=True)
            np.save('output/{}/test_set_{}.npy'.format(self.args.dataset, self.args.object_name), test_set)
            print('saved')
        regressor.delete_container(predictions, predictions_para, poses_para, pr_para, pi_para)

    def save_model(self, epoch):
        ckpt_dir = os.path.join(self.args.save_dir, 'checkpoints')
        note = str(self.args.lr)
        save_session(self.model, self.optimizer, ckpt_dir, note, epoch)
