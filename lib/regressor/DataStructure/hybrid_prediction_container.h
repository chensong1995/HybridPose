#ifndef hybrid_prediction_container_h_
#define hybrid_prediction_container_h_

#include <vector>
#include <Eigen/Dense>
#include "keypoint.h"
#include "edge_vector.h"
#include "symmetry_corres.h"
#include "affine_transformation.h"

using namespace std;
using namespace Eigen;

class HybridPredictionContainer {
 public:
  HybridPredictionContainer();
	~HybridPredictionContainer();

  vector<Keypoint>* GetKeypoints() {
    return &keypoints_;
  }
  const vector<Keypoint>* GetKeypoints() const {
    return &keypoints_;
  }
  vector<EdgeVector>* GetEdgeVectors() {
    return &edge_vectors_;
  }
  const vector<EdgeVector>* GetEdgeVectors() const {
    return &edge_vectors_;
  }
  vector<SymmetryCorres>* GetSymmetryCorres() {
    return &symmetry_corres_;
  }
  const vector<SymmetryCorres>* GetSymmetryCorres() const {
    return &symmetry_corres_;
  }
  AffineXform3d* GetRigidPose() {
    return &rigid_pose_;
  }
  const AffineXform3d* GetRigidPose() const{
    return &rigid_pose_;
  }
  Vector3d GetReflectionPlaneNormal() {
    return normal_gt_;
  }
  Vector3d GetReflectionPlaneNormal() const {
    return normal_gt_;
  }
  void SetReflectionPlaneNormal(Vector3d &normal_gt) {
    normal_gt_ = normal_gt;
  }
 private:
  // Collection of predicted keypoints and their ground-truth
  vector<Keypoint> keypoints_;
  // Collection of predicted edge vectors
  vector<EdgeVector> edge_vectors_;
  // Collection of predicted symmetry correspondences
  vector<SymmetryCorres> symmetry_corres_;
  // Ground-truth normal direction
  Vector3d normal_gt_;
  // Optimized rigid_pose
  AffineXform3d rigid_pose_;
};
#endif
