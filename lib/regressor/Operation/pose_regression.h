#ifndef pose_regression_h_
#define pose_regression_h_
/**************************************************************************/
// This file implements the data structure for a bounding box of a surface
// represented as a point cloud or a triangular mesh
/**************************************************************************/


#include "hybrid_prediction_container.h"

struct PoseRegressionPara {
 public:
  PoseRegressionPara() {
    beta_kpts_init = 1;
    beta_edge_init = 1;
    beta_sym_init = 0.05;
    numAlternatingIters = 5;
    numGaussNewtonIters = 10;
    beta_kpts = 1;
    beta_edge = 1;
    beta_sym = 2e-4;
    alpha_kpts = 2e-3;
    alpha_edge = 2e-3;
    alpha_sym = 5e-2;
  }
  ~PoseRegressionPara() {
  }
  double beta_kpts_init;
  double beta_edge_init;
  double beta_sym_init;
  unsigned numAlternatingIters;
  unsigned numGaussNewtonIters;
  double beta_kpts;
  double beta_edge;
  double beta_sym;
  double alpha_kpts;
  double alpha_edge;
  double alpha_sym;
};

class PoseRegression {
 public:
  PoseRegression() {
  }
  ~PoseRegression() {
  }

  void InitializePose(const HybridPredictionContainer& predictions,
    const PoseRegressionPara& para,
    AffineXform3d* rigid_pose);
  void RefinePose(const HybridPredictionContainer& predictions,
    const PoseRegressionPara &para,
    AffineXform3d* rigid_pose);
 private:
  // The following functions are used for pose initialization
  void GenerateDataMatrix(const HybridPredictionContainer& predictions,
    const vector<double> &weight_kpts,
    const vector<double> &weight_edges,
    const vector<double> &weight_symcorres,
    const PoseRegressionPara& para,
    Matrix12d* data_matrix);
  // Leading eigen-space computation
  void LeadingEigenSpace(const Matrix12d& data_matrix, const unsigned& numEigs,
    vector<Vector12d>* eigenVectors);
  
  // The following functions are used for pose refinement
  void Reweighting(const HybridPredictionContainer& predictions,
    const PoseRegressionPara& para,
    const AffineXform3d& rigid_pose,
    vector<double>* weight_keypts,
    vector<double>* weight_edge,
    vector<double>* weight_symcorres);
  // Projection operator and Taylor expansions
  void Projection(const Keypoint& pt, const AffineXform3d& aff, Vector2d* p_cur);
  // Compute the Jacobi of the keypoint term
  void Projection_jacobi(const Keypoint& pt, const AffineXform3d& aff,
    Vector2d* p_cur, Vector6d* jacobi_x, Vector6d* jacobi_y);
  // Compute the Jacobi of the symmetry correspondence term
  void Symcorres_residual(const SymmetryCorres& sc, const AffineXform3d& aff, const Vector3d& normal_gt,
    double* sym_residual);
  void Symcorres_jacobi(const SymmetryCorres& sc, const AffineXform3d& aff, const Vector3d& normal_gt,
    double* sym_residual, Vector3d* jacobi_sym);
  double Objective_value(const HybridPredictionContainer& predictions,
    const PoseRegressionPara& para,
    const vector<double>& weight_keypts,
    const vector<double>& weight_edges,
    const vector<double>& weight_symcorres,
    const AffineXform3d& rigid_pose);
  bool Solve6x6(const Matrix6d& hessian, Vector6d& gradient, double* velocity);
};
#endif
