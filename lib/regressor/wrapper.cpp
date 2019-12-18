#include "pose_regression.h"

extern "C" {
  void set_pose(HybridPredictionContainer* container, float** pose) {
    AffineXform3d* p = container->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        (*p)[i][j] = pose[i][j];
      }
    }
  }

  void get_pose(HybridPredictionContainer* container, float** pose) {
    AffineXform3d* p = container->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        pose[i][j] = (*p)[i][j];
      }
    }
  }

  void set_point3D_gt(HybridPredictionContainer* container,
                      float** point3D_gt,
                      int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    keypoints->resize(nk);
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 3; j++) {
        (*keypoints)[i].point3D_gt[j] = point3D_gt[i][j];
      }
    }
  }

  void get_point3D_gt(HybridPredictionContainer* container,
                      float** point3D_gt,
                      int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 3; j++) {
        point3D_gt[i][j] = (*keypoints)[i].point3D_gt[j];
      }
    }
  }
  
  void set_point2D_pred(HybridPredictionContainer* container,
                        float** point2D_pred,
                        int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    keypoints->resize(nk);
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 2; j++) {
        (*keypoints)[i].point2D_pred[j] = point2D_pred[i][j];
      }
    }
  }

  void get_point2D_pred(HybridPredictionContainer* container,
                        float** point2D_pred,
                        int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 2; j++) {
        point2D_pred[i][j] = (*keypoints)[i].point2D_pred[j];
      }
    }
  }

  void set_point_inv_half_var(HybridPredictionContainer* container,
                              float** inv_half_var,
                              int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    keypoints->resize(nk);
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          (*keypoints)[i].inv_half_var(j, k) = inv_half_var[i][j * 2 + k];
        }
      }
    }
  }

  void get_point_inv_half_var(HybridPredictionContainer* container,
                              float** inv_half_var,
                              int nk) {
    vector<Keypoint>* keypoints = container->GetKeypoints();
    for (int i = 0; i < nk; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          inv_half_var[i][j * 2 + k] = (*keypoints)[i].inv_half_var(j, k);
        }
      }
    }
  }

  void set_edge_ids(HybridPredictionContainer* container,
                    int* start_id,
                    int* end_id,
                    int ne) {
    vector<EdgeVector>* edges = container->GetEdgeVectors();
    edges->resize(ne);
    for (int i = 0; i < ne; i++) {
      (*edges)[i].start_id = start_id[i];
      (*edges)[i].end_id = end_id[i];
      (*edges)[i].start_id--;
      (*edges)[i].end_id--;
    }
  }

  void set_vec_pred(HybridPredictionContainer* container,
                    float** vec_pred,
                    int ne) {
    vector<EdgeVector>* edges = container->GetEdgeVectors();
    edges->resize(ne);
    for (int i = 0; i < ne; i++) {
      (*edges)[i].vec_pred[0] = vec_pred[i][0];
      (*edges)[i].vec_pred[1] = vec_pred[i][1];
    }
  }

  void get_vec_pred(HybridPredictionContainer* container,
                    float** vec_pred,
                    int ne) {
    vector<EdgeVector>* edges = container->GetEdgeVectors();
    for (int i = 0; i < ne; i++) {
      vec_pred[i][0] = (*edges)[i].vec_pred[0];
      vec_pred[i][1] = (*edges)[i].vec_pred[1];
    }
  }

  void set_edge_inv_half_var(HybridPredictionContainer* container,
                             float** inv_half_var,
                             int ne) {
    vector<EdgeVector>* edges = container->GetEdgeVectors();
    edges->resize(ne);
    for (int i = 0; i < ne; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          (*edges)[i].inv_half_var(k, j) = inv_half_var[i][j * 2 + k];
        }
      }
    }
  }

  void get_edge_inv_half_var(HybridPredictionContainer* container,
                             float** inv_half_var,
                             int ne) {
    vector<EdgeVector>* edges = container->GetEdgeVectors();
    for (int i = 0; i < ne; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          inv_half_var[i][j * 2 + k] = (*edges)[i].inv_half_var(k, j);
        }
      }
    }
  }

  void set_qs1_cross_qs2(HybridPredictionContainer* container,
                         float** qs1_cross_qs2,
                         int ns) {
    vector<SymmetryCorres>* symcorres = container->GetSymmetryCorres();
    symcorres->resize(ns);
    for (int i = 0; i < ns; i++) {
      for (int j = 0; j < 3; j++) {
        (*symcorres)[i].qs1_cross_qs2[j] = qs1_cross_qs2[i][j];
      }
    }
  }

  void get_qs1_cross_qs2(HybridPredictionContainer* container,
                         float** qs1_cross_qs2,
                         int ns) {
    vector<SymmetryCorres>* symcorres = container->GetSymmetryCorres();
    for (int i = 0; i < ns; i++) {
      for (int j = 0; j < 3; j++) {
        qs1_cross_qs2[i][j] = (*symcorres)[i].qs1_cross_qs2[j];
      }
    }
  }

  void set_symmetry_weight(HybridPredictionContainer* container,
                           float* weight,
                           int ns) {
    vector<SymmetryCorres>* symcorres = container->GetSymmetryCorres();
    symcorres->resize(ns);
    double sum = 0.0;
    for (int i = 0; i < ns; i++) {
      (*symcorres)[i].weight = weight[i];
      sum += weight[i];
    }
    for (int i = 0; i < ns; i++) {
      (*symcorres)[i].weight /= sum;
    }
  }

  void get_symmetry_weight(HybridPredictionContainer* container,
                           float* weight,
                           int ns) {
    vector<SymmetryCorres>* symcorres = container->GetSymmetryCorres();
    for (int i = 0; i < ns; i++) {
      weight[i] = (*symcorres)[i].weight = weight[i];
    }
  }

  void set_normal_gt(HybridPredictionContainer* container,
                     float* normal_gt) {
    Vector3d normal;
    for (int i = 0; i < 3; i++) {
      normal[i] = normal_gt[i];
    }
    container->SetReflectionPlaneNormal(normal);
  }

  HybridPredictionContainer* regress(HybridPredictionContainer* predictions) {
    AffineXform3d pose = *(predictions->GetRigidPose());
    PoseRegression pr;
    PoseRegressionPara pr_para;
    pr.InitializePose(*predictions, pr_para, &pose);
    pr.RefinePose(*predictions, pr_para, &pose);
    AffineXform3d* p = predictions->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        (*p)[i][j] = pose[i][j];
      }
    }
    return predictions;
  }

  HybridPredictionContainer* new_container() {
    HybridPredictionContainer* hpc = new HybridPredictionContainer();

    int start_id[] = {2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 5, 6, 7, 8, 6, 7, 8, 7, 8, 8};
    int   end_id[] = {1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7};
    set_edge_ids(hpc, start_id, end_id, 28);
    return hpc;
  }

  void delete_container(HybridPredictionContainer* container) {
    delete container;
  }

}
