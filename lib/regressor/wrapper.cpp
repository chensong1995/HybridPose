#include "pose_regression.h"
#include "pose_regression_parameter_search.h"

extern "C" {
  void set_pose(HybridPredictionContainer* container, float** pose) {
    AffineXform3d* p = container->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        (*p)[i][j] = pose[i][j];
      }
    }
  }

  void set_pose_gt(vector<AffineXform3d>* pose_container, int pose_id, float** pose) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        ((*pose_container)[pose_id])[i][j] = pose[i][j];
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
          (*edges)[i].inv_half_var(j, k) = inv_half_var[i][j * 2 + k];
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
          inv_half_var[i][j * 2 + k] = (*edges)[i].inv_half_var(j, k);
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


  HybridPredictionContainer* initialize_pose(HybridPredictionContainer* predictions, 
                                             PRInitPara* pi_para, 
                                             int turn_on_kpt, 
                                             int turn_on_edge, 
                                             int turn_on_sym) {
    AffineXform3d pose;
    PoseRegression pr;
    PoseRegressionPara para;

    para.gamma_edge = pi_para->gamma_edge * turn_on_edge;
    para.gamma_sym = pi_para->gamma_sym * turn_on_sym;
    para.gamma_kpts = para.gamma_kpts * turn_on_kpt;

    pr.InitializePose(*predictions, para, &pose);

    AffineXform3d* p = predictions->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        (*p)[i][j] = pose[i][j];
      }
    }
    return predictions;
  }

  HybridPredictionContainer* refine_pose(HybridPredictionContainer* predictions,
                                         PRRefinePara* pr_para,
                                         int turn_on_kpt, 
                                         int turn_on_edge, 
                                         int turn_on_sym) {
    AffineXform3d pose = *(predictions->GetRigidPose());;
    PoseRegression pr;
    PoseRegressionPara para;

    para.beta_edge = pr_para->beta_edge * turn_on_edge;
    para.beta_sym = pr_para->beta_sym * turn_on_sym;
    para.beta_kpts = para.beta_kpts * turn_on_kpt;
    para.alpha_kpts = pr_para->alpha_kpts;
    para.alpha_edge = pr_para->alpha_edge;
    para.alpha_sym = pr_para->alpha_sym;

    pr.RefinePose(*predictions, para, &pose);
    AffineXform3d* p = predictions->GetRigidPose();
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        (*p)[i][j] = pose[i][j];
      }
    }

    return predictions;
  }

  PRRefinePara* search_pose_refine(vector<HybridPredictionContainer>* predictions_para, 
                       vector<AffineXform3d>* poses_gt, int data_size, double diameter) {

    predictions_para->resize(data_size);
    poses_gt->resize(data_size);    
    PoseRegressionParameterSearch pr_ps;
    ParameterSearchConfig ps_config;    
    ps_config.lambda_trans = 4. / (diameter * diameter);
    PRRefinePara* pr_para = new PRRefinePara();
    pr_ps.Refinement(*predictions_para, *poses_gt, (const ParameterSearchConfig) ps_config, pr_para);
    return pr_para;
  }

  PRInitPara* search_pose_initial(vector<HybridPredictionContainer>* predictions_para, 
                       vector<AffineXform3d>* poses_gt, int data_size, double diameter) {

    predictions_para->resize(data_size);
    poses_gt->resize(data_size);    
    PoseRegressionParameterSearch pr_ps;
    ParameterSearchConfig ps_config;    
    ps_config.lambda_trans = 4. / (diameter * diameter);
    PRInitPara* pi_para = new PRInitPara();
    pr_ps.Initialization(*predictions_para, *poses_gt, (const ParameterSearchConfig) ps_config, pi_para);
    
    // set pose initial for validation set using searched parameters
    for (unsigned id = 0; id < data_size; id++) {
      initialize_pose(&((*predictions_para)[id]), pi_para, 1, 1, 1);
    }
    return pi_para;
  }

  HybridPredictionContainer* new_container() {
    HybridPredictionContainer* hpc = new HybridPredictionContainer();

    int start_id[] = {2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 5, 6, 7, 8, 6, 7, 8, 7, 8, 8};
    int   end_id[] = {1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7};
    set_edge_ids(hpc, start_id, end_id, 28);
    return hpc;
  }
  
  vector<HybridPredictionContainer>* new_container_para() {
    // a vector of intermediate predictions for N=20 examples in the val set
    vector<HybridPredictionContainer>* hpc_para = new vector<HybridPredictionContainer>();   
    (*hpc_para).resize(20);    

    int start_id[] = {2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 5, 6, 7, 8, 6, 7, 8, 7, 8, 8};
    int   end_id[] = {1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7};        
    for (unsigned id = 0; id < (*hpc_para).size(); ++id)   
      set_edge_ids(&((*hpc_para)[id]), start_id, end_id, 28);       
    return hpc_para;
  }

  HybridPredictionContainer* get_prediction_container(vector<HybridPredictionContainer>* predictions_para,
                                                      int container_id) {
    HybridPredictionContainer *predictions;
    predictions = &((*predictions_para)[container_id]);
    return predictions;
  }

  vector<AffineXform3d>* new_container_pose() {    
    // a vector of ground-truth poses for N=20 exmaples in the val set
    vector<AffineXform3d>* poses = new vector<AffineXform3d>();   
    (*poses).resize(20);
    return poses;
  }

  void delete_container(HybridPredictionContainer* container,
                        vector<HybridPredictionContainer>* c2,
                        vector<AffineXform3d>* c3, 
                        PRRefinePara* c4, 
                        PRInitPara* c5) { 
    delete container;
    delete c2;
    delete c3;
    delete c4;
    delete c5;
  }
}
