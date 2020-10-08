import numpy as np
import cv2
import pdb

# https://github.com/zju3dv/clean-pvnet/blob/master/lib/datasets/augmentation.py

def debug_visualize(image, mask, pts2d, sym_cor, name_prefix='debug'):
    from random import sample
    cv2.imwrite('{}_image.png'.format(name_prefix), image * 255)
    cv2.imwrite('{}_mask.png'.format(name_prefix), mask * 255)
    img_pts = image.copy() * 255
    for i in range(pts2d.shape[0]):
        x = int(round(pts2d[i, 0]))
        y = int(round(pts2d[i, 1]))
        img_pts = cv2.circle(img_pts, (x, y), 2, (0, 0, 255), thickness=-1)
    cv2.imwrite('{}_pts.png'.format(name_prefix), img_pts)
    img_sym = image.copy() * 255
    ys, xs = np.nonzero(mask)
    for i_pt in sample([i for i in range(len(ys))], min(100, len(ys))):
        y = int(round(ys[i_pt]))
        x = int(round(xs[i_pt]))
        x_cor, y_cor = sym_cor[y, x]
        x_cor = int(round(x + x_cor))
        y_cor = int(round(y + y_cor))
        img_sym = cv2.line(img_sym, (x, y), (x_cor, y_cor), (0, 0, 255), 1)
    cv2.imwrite('{}_sym.png'.format(name_prefix), img_sym)

def rotate_sym_cor(sym_cor, mask, R):
    h, w = sym_cor.shape[:2]
    ys, xs = np.nonzero(mask)
    source = np.float32(np.stack([xs, ys], axis=-1))
    delta = np.float32(sym_cor[ys, xs])
    target = source + delta
    last_col = np.ones((source.shape[0], 1), dtype=np.float32)
    source = np.concatenate([source, last_col], axis=-1)
    target = np.concatenate([target, last_col], axis=-1)
    last_row = np.asarray([[0, 0, 1]], dtype=np.float32)
    R = np.concatenate([R, last_row], axis=0).transpose()
    source = np.matmul(source, R)[:, :2]
    target = np.matmul(target, R)[:, :2]
    source = np.uint32(np.round(source))
    delta = target - source
    # remove invalid indices
    xs, ys = source[:, 0], source[:, 1]
    valid = (xs > 0) & (xs < w) & (ys > 0) & (ys < h)
    xs, ys, delta = xs[valid], ys[valid], delta[valid]
    sym_cor = np.zeros_like(sym_cor)
    sym_cor[ys, xs] = delta
    return sym_cor

def rotate_instance(img, mask, hcoords, sym_cor, rot_ang_min, rot_ang_max):
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    hs, ws = np.nonzero(mask)
    R = cv2.getRotationMatrix2D((np.mean(ws), np.mean(hs)), degree, 1)
    sym_cor = rotate_sym_cor(sym_cor, mask, R)
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    last_row = np.asarray([[0, 0, 1]], dtype=np.float32)
    R = np.concatenate([R, last_row], axis=0).transpose()
    last_col = np.ones((hcoords.shape[0], 1), dtype=np.float32)
    hcoords = np.concatenate([hcoords, last_col], axis=1)
    hcoords = np.float32(np.matmul(hcoords, R))
    hcoords = hcoords[:, :2]
    return img, mask, hcoords, sym_cor

def crop_resize_instance_v1(img, mask, hcoords, sym_cor, imheight, imwidth,
                            overlap_ratio=0.5, ratio_min=0.8, ratio_max=1.2):
    '''
    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap
    '''
    hcoords_last_col = np.ones((hcoords.shape[0], 1), dtype=np.float32)
    hcoords = np.concatenate([hcoords, hcoords_last_col], axis=1)

    resize_ratio = np.random.uniform(ratio_min, ratio_max)
    target_height = int(imheight * resize_ratio)
    target_width = int(imwidth * resize_ratio)

    img, mask, hcoords, sym_cor = crop_or_padding_to_fixed_size_instance(
        img, mask, hcoords, sym_cor, target_height, target_width, overlap_ratio)

    img = cv2.resize(img, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)
    sym_cor = cv2.resize(sym_cor, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)
    sym_cor /= resize_ratio

    hcoords[:, 0] = hcoords[:, 0] / resize_ratio
    hcoords[:, 1] = hcoords[:, 1] / resize_ratio
    hcoords = hcoords[:, :2]
    return img, mask, hcoords, sym_cor

def crop_or_padding_to_fixed_size_instance(img, mask, hcoords, sym_cor, th, tw,
                                           overlap_ratio=0.5):
    h, w, _ = img.shape
    hs, ws = np.nonzero(mask)

    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)
    fh, fw = hmax - hmin, wmax - wmin
    hpad, wpad = th >= h, tw >= w

    hrmax = int(min(hmin + overlap_ratio * fh, h - th))  # h must > target_height else hrmax<0
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmin + overlap_ratio * fw, w - tw))  # w must > target_width else wrmax<0
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))

    hbeg = 0 if (hpad or hrmin == hrmax) else np.random.randint(hrmin, hrmax)
    hend = hbeg + th
    wbeg = 0 if (wpad or wrmin == wrmax) else np.random.randint(wrmin, wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]
    sym_cor = sym_cor[hbeg:hend, wbeg:wend]

    hcoords[:, 0] -= wbeg * hcoords[:, 2]
    hcoords[:, 1] -= hbeg * hcoords[:, 2]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)
        new_sym_cor = np.zeros([th, tw, 2], dtype=sym_cor.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        new_sym_cor[hbeg:hbeg + nh, wbeg:wbeg + nw] = sym_cor
        hcoords[:, 0] += wbeg * hcoords[:, 2]
        hcoords[:, 1] += hbeg * hcoords[:, 2]

        img, mask, sym_cor = new_img, new_mask, new_sym_cor

    return img, mask, hcoords, sym_cor

def crop_or_padding_to_fixed_size(img, mask, sym_cor, th, tw):
    h, w, _ = img.shape
    hpad, wpad = th >= h, tw >= w

    hbeg = 0 if hpad else np.random.randint(0, h - th)
    wbeg = 0 if wpad else np.random.randint(0,
                                            w - tw)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    hend = hbeg + th
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]
    sym_cor = sym_cor[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)
        new_sym_cor = np.zeros([th, tw, 2], dtype=sym_cor.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        new_sym_cor[hbeg:hbeg + nh, wbeg:wbeg + nw] = sym_cor

        img, mask, sym_cor = new_img, new_mask, new_sym_cor

    return img, mask, sym_cor
