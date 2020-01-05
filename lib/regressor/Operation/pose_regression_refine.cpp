#include "pose_regression.h"

void PoseRegression::RefinePose(const HybridPredictionContainer& predictions,
  const PoseRegressionPara& para,
  AffineXform3d* rigid_pose) {
  vector<double> weight_keypts, weight_edges, weight_symcorres;  
  Reweighting(predictions, para, *rigid_pose,
    &weight_keypts, &weight_edges, &weight_symcorres);

  // Allocate space for the linear system
  Matrix6d A;
  Vector6d b;
  double val_cur, val_next;

  // Allocate space for the projected keypoints and first-order dervatives
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  vector<Vector2d> projected_keypoints;
  vector<Vector6d> projected_keypoints_1st_order_x;
  vector<Vector6d> projected_keypoints_1st_order_y;
  projected_keypoints.resize(keypoints->size());
  projected_keypoints_1st_order_x.resize(keypoints->size());
  projected_keypoints_1st_order_y.resize(keypoints->size());

  // Allocate space for each symmetry residual and its first-order derivative
  double sym_residual;
  Vector3d jacobi_sym;

  // The ground-truth normal to the reflection plane in the world coordinate system
  Vector3d normal_gt = predictions.GetReflectionPlaneNormal();

  // Points to edge vectors and symmetry correspondences
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();

  for (unsigned outerIter = 0; outerIter < para.numAlternatingIters; ++outerIter) {
    for (unsigned innerIter = 0; innerIter < para.numGaussNewtonIters; ++innerIter) {
      // buffer the projections and the first-order derivatives
      for (unsigned ptId = 0; ptId < keypoints->size(); ++ptId)
        Projection_jacobi((*keypoints)[ptId], *rigid_pose,
          &projected_keypoints[ptId],
          &projected_keypoints_1st_order_x[ptId],
          &projected_keypoints_1st_order_y[ptId]);

      // Initialize
      for (int i = 0; i < 6; ++i) {
        b[i] = 0.0;
        for (int j = i; j < 6; ++j)
          A(i, j) = 0.0;
      }
      val_cur = 0.0;

      // Update Keypoints and Edges
      for (unsigned ptId = 0; ptId < keypoints->size(); ++ptId) {
        const Keypoint& kp = (*keypoints)[ptId];
        Vector2d adjusted_dif = kp.inv_half_var * (kp.point2D_pred - projected_keypoints[ptId]);

        const Vector6d& jacobi_x = projected_keypoints_1st_order_x[ptId];
        const Vector6d& jacobi_y = projected_keypoints_1st_order_y[ptId];
        Vector6d adjusted_jacobi_x = jacobi_x * kp.inv_half_var(0, 0) + jacobi_y * kp.inv_half_var(0, 1);
        Vector6d adjusted_jacobi_y = jacobi_x * kp.inv_half_var(1, 0) + jacobi_y * kp.inv_half_var(1, 1);
        double w = para.beta_kpts * weight_keypts[ptId];

        // Update objective function value
        val_cur += w * adjusted_dif.squaredNorm();

        // Update first-order approximation
        b += adjusted_jacobi_x * (adjusted_dif[0] * w) + adjusted_jacobi_y * (adjusted_dif[1] * w);

        // Update second-order approximation
        for (int i = 0; i < 6; ++i)
          for (int j = i; j < 6; ++j)
            A(i, j) += (adjusted_jacobi_x[i] * adjusted_jacobi_x[j]
              + adjusted_jacobi_y[i] * adjusted_jacobi_y[j]) * w;
      }
      for (unsigned edgeId = 0; edgeId < edges->size(); ++edgeId) {
        const EdgeVector& ev = (*edges)[edgeId];
        Vector2d dif = ev.vec_pred - (projected_keypoints[ev.start_id] - projected_keypoints[ev.end_id]);
        Vector2d adjusted_dif = ev.inv_half_var * dif;
        Vector6d jacobi_edge_x = projected_keypoints_1st_order_x[ev.start_id]
          - projected_keypoints_1st_order_x[ev.end_id];
        Vector6d jacobi_edge_y = projected_keypoints_1st_order_y[ev.start_id]
          - projected_keypoints_1st_order_y[ev.end_id];
        Vector6d adjusted_jacobi_edge_x = jacobi_edge_x * ev.inv_half_var(0, 0)
          + jacobi_edge_y * ev.inv_half_var(0, 1);
        Vector6d adjusted_jacobi_edge_y = jacobi_edge_x * ev.inv_half_var(1, 0)
          + jacobi_edge_y * ev.inv_half_var(1, 1);
        double w = para.beta_edge * weight_edges[edgeId];

        // Update the value of the objective function
        val_cur += w * adjusted_dif.squaredNorm();

        // Update the first-order approximation
        b += adjusted_jacobi_edge_x * (adjusted_dif[0] * w)
          + adjusted_jacobi_edge_y * (adjusted_dif[1] * w);

        // Update the second-order approximation
        for (int i = 0; i < 6; ++i)
          for (int j = i; j < 6; ++j)
            A(i, j) += (adjusted_jacobi_edge_x[i] * adjusted_jacobi_edge_x[j]
              + adjusted_jacobi_edge_y[i] * adjusted_jacobi_edge_y[j]) * w;
      }
      // Update symmetry correspondences
      for (unsigned corId = 0; corId < symcorres->size(); ++corId) {
        const SymmetryCorres& sc = (*symcorres)[corId];
        Symcorres_jacobi(sc, *rigid_pose, normal_gt,
          &sym_residual, &jacobi_sym);
        double w = para.beta_sym * weight_symcorres[corId];

        // Update the value of the objective function
        val_cur += w * sym_residual * sym_residual;

        for (int i = 0; i < 3; ++i) {
          // Update the first-order objective function
          b[i + 3] -= jacobi_sym[i] * sym_residual * w;
          // Update the second order approximation
          for (int j = i; j < 3; ++j)
            A(i + 3, j + 3) += (jacobi_sym[i] * jacobi_sym[j]) * w;
        }
      }
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < i; ++j)
          A(i, j) = A(j, i);

      // Perform Gauss-Newton step
      double velocity[6];
      Solve6x6(A, b, velocity);
      double norm2 = sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]
        + velocity[3] * velocity[3] + velocity[4] * velocity[4] + velocity[5] * velocity[5]);
      if (norm2 < 1e-7)
        break;
      AffineXform3d newMotion(velocity);
      AffineXform3d rigid_next = newMotion * (*rigid_pose);
      val_next = Objective_value(predictions, para,
        weight_keypts, weight_edges, weight_symcorres, rigid_next);

      if (val_next < val_cur) {
        *rigid_pose = rigid_next;
      } else {
        // Perform BFGS to search for the next pose to reduce the energy function
        bool success_update = false;
        double lambda = (A(0, 0) + A(1, 1) + A(2, 2) + A(3, 3) + A(4, 4) + A(5, 5)) / 6e3;
        for (unsigned iter2 = 0; iter2 < 10; ++iter2) {
          for (int i = 0; i < 6; ++i)
            A(i, i) += lambda;
          Solve6x6(A, b, velocity);
          AffineXform3d newMotion(velocity);
          AffineXform3d rigid_next = newMotion * (*rigid_pose);
          val_next = Objective_value(predictions, para,
            weight_keypts, weight_edges, weight_symcorres, rigid_next);
          if (val_next < val_cur) {
            *rigid_pose = rigid_next;
            success_update = true;
            break;
          }
          lambda = lambda * 2;
        }
        if (!success_update)
          break;
      }
    }
  
  }
}


double PoseRegression::Objective_value(const HybridPredictionContainer& predictions,
  const PoseRegressionPara& para,
  const vector<double>& weight_keypts,
  const vector<double>& weight_edges,
  const vector<double>& weight_symcorres,
  const AffineXform3d& rigid_pose) {
  double val_cur = 0.0;

  // Allocate space for the projected keypoints and first-order dervatives
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  vector<Vector2d> projected_keypoints;
  projected_keypoints.resize(keypoints->size());

  // Allocate space for each symmetry residual and its first-order derivative
  double sym_residual;

  // The ground-truth normal to the reflection plane in the world coordinate system
  Vector3d normal_gt = predictions.GetReflectionPlaneNormal();

  // Points to edge vectors and symmetry correspondences
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();

  for (unsigned ptId = 0; ptId < keypoints->size(); ++ptId)
    Projection((*keypoints)[ptId], rigid_pose,
      &projected_keypoints[ptId]);

  // Update Keypoints and Edges
  for (unsigned ptId = 0; ptId < keypoints->size(); ++ptId) {
    const Keypoint& kp = (*keypoints)[ptId];
    Vector2d adjusted_dif = kp.inv_half_var * (kp.point2D_pred - projected_keypoints[ptId]);
    double w = para.beta_kpts * weight_keypts[ptId];
    // Update objective function value
    val_cur += w * adjusted_dif.squaredNorm();
  }
  for (unsigned edgeId = 0; edgeId < edges->size(); ++edgeId) {
    const EdgeVector& ev = (*edges)[edgeId];
    Vector2d dif = ev.vec_pred - (projected_keypoints[ev.start_id] - projected_keypoints[ev.end_id]);
    Vector2d adjusted_dif = ev.inv_half_var * dif;
    double w = para.beta_edge * weight_edges[edgeId];
    // Update the value of the objective function
    val_cur += w * adjusted_dif.squaredNorm();
  }
  // Update symmetry correspondences
  for (unsigned corId = 0; corId < symcorres->size(); ++corId) {
    const SymmetryCorres& sc = (*symcorres)[corId];
    Symcorres_residual(sc, rigid_pose, normal_gt, &sym_residual);
    double w = para.beta_sym * weight_symcorres[corId];
    // Update the value of the objective function
    val_cur += w * sym_residual * sym_residual;
  }
  return val_cur;
}

void PoseRegression::Reweighting(const HybridPredictionContainer& predictions,
  const PoseRegressionPara& para,
  const AffineXform3d& rigid_pose,
  vector<double>* weight_keypts,
  vector<double>* weight_edge,
  vector<double>* weight_symcorres) {
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();
  // Buffer the projected keypoints
  vector<Vector2d> buffer_projections;
  buffer_projections.resize(keypoints->size());
  weight_keypts->resize(keypoints->size());

  double sigma2 = para.alpha_kpts * para.alpha_kpts;
  for (unsigned id = 0; id < buffer_projections.size(); ++id) {
    const Keypoint& kp = (*keypoints)[id];
    Projection(kp, rigid_pose, &buffer_projections[id]);
    Vector2d dif = kp.point2D_pred - buffer_projections[id];
    Vector2d adjusted_dif = kp.inv_half_var * dif;  
    (*weight_keypts)[id] = sigma2 / (sigma2 + adjusted_dif.squaredNorm());
  }

  weight_edge->resize(edges->size());
  sigma2 = para.alpha_edge * para.alpha_edge;
  for (unsigned id = 0; id < edges->size(); ++id) {
    const EdgeVector& ev = (*edges)[id];
    Vector2d dif = ev.vec_pred - (buffer_projections[ev.start_id] - buffer_projections[ev.end_id]);
    Vector2d adjusted_dif = ev.inv_half_var * dif;
    (*weight_edge)[id] = sigma2 / (sigma2 + adjusted_dif.squaredNorm());
  }

  weight_symcorres->resize(symcorres->size());
  sigma2 = para.alpha_sym * para.alpha_sym;
  const Vector3d &normal_gt = predictions.GetReflectionPlaneNormal();
  double residual = 0.0;
  for (unsigned id = 0; id < symcorres->size(); ++id) {
    const SymmetryCorres& sc = (*symcorres)[id];
    Symcorres_residual(sc, rigid_pose, normal_gt, &residual);
    (*weight_symcorres)[id] = sigma2*sc.weight / (sigma2 + residual * residual);
  }
}

void PoseRegression::Projection(const Keypoint& pt, const AffineXform3d& aff,
  Vector2d* p_cur) {
  Vector3d pos_camera = aff[0] + aff[1] * pt.point3D_gt[0]
    + aff[2] * pt.point3D_gt[1] + aff[3] * pt.point3D_gt[2];
  (*p_cur)[0] = pos_camera[0] / pos_camera[2];
  (*p_cur)[1] = pos_camera[1] / pos_camera[2];
}

// Compute the Jacobi of the keypoint term
void PoseRegression::Projection_jacobi(const Keypoint& pt, const AffineXform3d& aff,
  Vector2d* p_cur, Vector6d* jacobi_x, Vector6d* jacobi_y) {
  Vector3d pos_camera = aff[0] + aff[1] * pt.point3D_gt[0]
    + aff[2] * pt.point3D_gt[1] + aff[3] * pt.point3D_gt[2];
  (*p_cur)[0] = pos_camera[0] / pos_camera[2];
  (*p_cur)[1] = pos_camera[1] / pos_camera[2];
  //              (1, 0, 0, 0,             pos_camera[2], -pos_camera[1]) 
  //- (*p_cur)[0]*(0, 0, 1,pos_camera[1], -pos_camera[0], 0)
  (*jacobi_x)[0] = 1;
  (*jacobi_x)[1] = 0;
  (*jacobi_x)[2] = -(*p_cur)[0];
  (*jacobi_x)[3] = -(*p_cur)[0] * pos_camera[1];
  (*jacobi_x)[4] = pos_camera[2] + (*p_cur)[0]* pos_camera[0];
  (*jacobi_x)[5] = -pos_camera[1];

  //              (0, 1, 0,-pos_camera[2], 0,              pos_camera[0]) 
  //- (*p_cur)[1]*(0, 0, 1, pos_camera[1], -pos_camera[0], 0)
  (*jacobi_y)[0] = 0;
  (*jacobi_y)[1] = 1;
  (*jacobi_y)[2] = -(*p_cur)[1];
  (*jacobi_y)[3] = -pos_camera[2] -(*p_cur)[1] * pos_camera[1];
  (*jacobi_y)[4] = (*p_cur)[1] * pos_camera[0];
  (*jacobi_y)[5] = pos_camera[0];

  (*jacobi_x) /= pos_camera[2];
  (*jacobi_y) /= pos_camera[2];
}

void PoseRegression::Symcorres_residual(const SymmetryCorres& sc, const AffineXform3d& aff, const Vector3d& normal_gt,
  double* sym_residual) {
  Vector3d tp = aff[1] * normal_gt[0] + aff[2] * normal_gt[1] + aff[3] * normal_gt[2];
  *sym_residual = sc.qs1_cross_qs2.dot(tp);
}

void PoseRegression::Symcorres_jacobi(const SymmetryCorres& sc, const AffineXform3d& aff, const Vector3d& normal_gt,
  double* sym_residual, Vector3d* jacobi_sym) {
  Vector3d normal_cur = aff[1] * normal_gt[0] + aff[2] * normal_gt[1] + aff[3] * normal_gt[2];
  *sym_residual = sc.qs1_cross_qs2.dot(normal_cur);
  *jacobi_sym = normal_cur.cross(sc.qs1_cross_qs2);
}


bool PoseRegression::Solve6x6(const Matrix6d& hessian, Vector6d& gradient,
  double* velocity) {
  // Using LLT factorization to solve the symmetric linear system
  const double fTolerance = 1e-20;
  double afV[6], Lower[6][6];
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j)
      Lower[i][j] = hessian(i, j);
    velocity[i] = gradient[i];
  }

  for (int i1 = 0; i1 < 6; ++i1) {
    for (int i0 = 0; i0 < i1; ++i0)
      afV[i0] = Lower[i1][i0] * Lower[i0][i0];

    afV[i1] = Lower[i1][i1];
    for (int i0 = 0; i0 < i1; ++i0)
      afV[i1] -= Lower[i1][i0] * afV[i0];

    Lower[i1][i1] = afV[i1];
    if (fabs(afV[i1]) <= fTolerance) //singular
      return false;

    double fInv = 1.0f / afV[i1];
    for (int i0 = i1 + 1; i0 < 6; ++i0) {
      for (int i2 = 0; i2 < i1; ++i2)
        Lower[i0][i1] -= Lower[i0][i2] * afV[i2];
      Lower[i0][i1] *= fInv;
    }
  }

  // Solve Ax = B.
  // Forward substitution
  for (int i0 = 0; i0 < 6; ++i0) {
    for (int i1 = 0; i1 < i0; ++i1)
      velocity[i0] -= Lower[i0][i1] * velocity[i1];
  }

  // Diagonal division:  Let y = L^t x, then Dy = z.  Algorithm stores
  // y terms in B vector.
  for (int i0 = 0; i0 < 6; ++i0) {
    if (fabs(Lower[i0][i0]) <= fTolerance)
      return false;
    velocity[i0] /= Lower[i0][i0];
  }

  // Back substitution:  Solve L^t x = y.  Algorithm stores x terms in
  // B vector.
  for (int i0 = 4; i0 >= 0; i0--) {
    for (int i1 = i0 + 1; i1 < 6; ++i1)
      velocity[i0] -= Lower[i1][i0] * velocity[i1];
  }
  return true;
}
