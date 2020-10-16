from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import pdb

def get_2d_ctypes(arr2d):
    if not arr2d.flags['C_CONTIGUOUS']:
        arr2d = np.ascontiguousarray(arr2d)
    return (arr2d.__array_interface__['data'][0] 
            + np.arange(arr2d.shape[0]) * arr2d.strides[0]).astype(np.uintp)

def load_wrapper():
    regressor = CDLL('wrapper.so')
    # specify argument and return value types
    # https://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes
    c_2darr_p = ndpointer(dtype=np.uintp, ndim=1, flags='C')
    # new_container()
    regressor.new_container.restype = c_void_p    
    # set_pose()
    regressor.set_pose.argtypes = [c_void_p, c_2darr_p]
    # get_pose()
    regressor.get_pose.argtypes = [c_void_p, c_2darr_p]
    # set_point3D_gt()
    regressor.set_point3D_gt.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_point3D_gt()
    regressor.get_point3D_gt.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_point2D_pred()
    regressor.set_point2D_pred.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_point2D_pred()
    regressor.get_point2D_pred.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_point_inv_half_var()
    regressor.set_point_inv_half_var.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_point_inv_half_var()
    regressor.get_point_inv_half_var.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_edge_ids()
    regressor.set_edge_ids.argtypes = [c_void_p, c_void_p, c_void_p, c_int]
    # set_vec_pred()
    regressor.set_vec_pred.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_vec_pred()
    regressor.get_vec_pred.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_edge_inv_half_var()
    regressor.set_edge_inv_half_var.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_edge_inv_half_var()
    regressor.get_edge_inv_half_var.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_qs1_cross_qs2()
    regressor.set_qs1_cross_qs2.argtypes = [c_void_p, c_2darr_p, c_int]
    # get_qs1_cross_qs2()
    regressor.get_qs1_cross_qs2.argtypes = [c_void_p, c_2darr_p, c_int]
    # set_symmetry_weight()
    regressor.set_symmetry_weight.argtypes = [c_void_p, c_void_p, c_int]
    # get_symmetry_weight()
    regressor.get_symmetry_weight.argtypes = [c_void_p, c_void_p, c_int]
    # set_normal_gt()
    regressor.set_normal_gt.argtypes = [c_void_p, c_void_p]
    # initialize_pose()
    regressor.initialize_pose.argtypes = [c_void_p, c_void_p]
    regressor.initialize_pose.restype = c_void_p
    # refine_pose()
    regressor.refine_pose.argtypes = [c_void_p, c_void_p]
    regressor.refine_pose.restype = c_void_p

    ## search parameter
    regressor.new_container_para.restype = c_void_p
    regressor.new_container_pose.restype = c_void_p
    # get_prediction_container()
    regressor.get_prediction_container.argtypes = [c_void_p, c_int]
    regressor.get_prediction_container.restype = c_void_p
    # set_pose_gt()
    regressor.set_pose_gt.argtypes = [c_void_p, c_int, c_2darr_p]
    # search_pose_initial()
    regressor.search_pose_initial.argtypes = [c_void_p, c_void_p, c_int, c_double]
    regressor.search_pose_initial.restype = c_void_p
    # search_pose_refine()
    regressor.search_pose_refine.argtypes = [c_void_p, c_void_p, c_int, c_double]
    regressor.search_pose_refine.restype = c_void_p
    # delete_container()
    regressor.delete_container.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    
    return regressor

def test():
    regressor = load_wrapper()

    predictions = regressor.new_container()
    pose = np.ones((4, 3), dtype=np.float32)
    posepp = get_2d_ctypes(pose)
    regressor.get_pose(predictions, posepp)
    print(pose)
    pose = np.ones((4, 3), dtype=np.float32)
    posepp = get_2d_ctypes(pose)
    regressor.set_pose(predictions, posepp)
    regressor.get_pose(predictions, posepp)
    print(pose)
    regressor.delete_container(predictions)

if __name__ == '__main__':
    test()
