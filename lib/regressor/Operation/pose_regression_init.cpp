#include "pose_regression.h"

void PoseRegression::InitializePose(const HybridPredictionContainer& predictions,
  const PoseRegressionPara& para,
  AffineXform3d* rigid_pose) {
  vector<double> weight_keypts;

}

/*
Compute the 12x12 matrix 
*/
void PoseRegression::GenerateDataMatrix(const HybridPredictionContainer& predictions,
  const vector<double>& weight_kpts,
  const vector<double>& weight_edges,
  const vector<double>& weight_symcorres,
  const PoseRegressionPara& para,
  Matrix12d* data_matrix) {
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();

}
