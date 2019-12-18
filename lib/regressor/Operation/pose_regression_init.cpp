#include "pose_regression.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cmath>

void PoseRegression::InitializePose(const HybridPredictionContainer& predictions,
                                    const PoseRegressionPara& para,
                                    AffineXform3d* rigid_pose) {
 
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();

  // initial weights of different representations 
  vector<double> weight_keypts (keypoints->size()), weight_edges (edges->size()), weight_symcorres (symcorres->size());
  for (unsigned id = 0; id < keypoints->size(); ++id)
    weight_keypts[id] = 1;
  for (unsigned id = 0; id < edges->size(); ++id)
    weight_edges[id] = 1;
  for (unsigned id = 0; id < symcorres->size(); ++id)
    weight_symcorres[id] = 1;

  // Generate 12x12 Data matrix from different representations
  Matrix<double, 12, 12> DataMatrix;
  GenerateDataMatrix(predictions, weight_keypts, weight_edges, weight_symcorres,
    para, &DataMatrix);

  // Calculate Pose initial
  vector<Vector12d> eigenVectors;
  const unsigned numEigs = 4;
  //LeadingEigenSpace(DataMatrix, numEigs, predictions, &eigenVectors);
  LeadingEigenSpace(DataMatrix, numEigs, predictions, &eigenVectors, rigid_pose); 
}
/*
Compute the 12x12 matrix 
*/
void PoseRegression::GenerateDataMatrix(const HybridPredictionContainer& predictions,
                                        const vector<double>& weight_keypts,
                                        const vector<double>& weight_edges,
                                        const vector<double>& weight_symcorres,
                                        const PoseRegressionPara& para,
                                        Matrix12d* data_matrix) {
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  const vector<EdgeVector>* edges = predictions.GetEdgeVectors();
  const vector<SymmetryCorres>* symcorres = predictions.GetSymmetryCorres();

  // Step1: add contribution of keypoint to data matrix
  unsigned numKpts = keypoints->size();
  Matrix<double, 12, Dynamic> J_kpts;
  Matrix<double, Dynamic, Dynamic> W_kpts;
  J_kpts.resize(12, numKpts*3);
  W_kpts.resize(numKpts*3, numKpts*3);
  Matrix<double, 12, 12> M_kpts; 
  
  J_kpts.fill(0.0);
  W_kpts.fill(0.0);
  M_kpts.fill(0.0);

  for (unsigned ptId = 0; ptId < numKpts; ++ptId) {  
    const Keypoint& kp = (*keypoints)[ptId];
    //kp.point3D_gt
    //kp.point2D_pred
    double x_hc, y_hc, z_hc;
    x_hc = kp.point2D_pred[0];  
    y_hc = kp.point2D_pred[1]; 
    z_hc = 1.0;

    double x_3d, y_3d, z_3d;
    x_3d = kp.point3D_gt[0];
    y_3d = kp.point3D_gt[1];
    z_3d = kp.point3D_gt[2];
    
    J_kpts(0, ptId * 3) = 0 * x_3d;     J_kpts(0, ptId * 3 + 1) = -z_hc * x_3d; J_kpts(0, ptId * 3 + 2) = y_hc * x_3d;
    J_kpts(1, ptId * 3) = z_hc * x_3d;  J_kpts(1, ptId * 3 + 1) = 0 * x_3d;     J_kpts(1, ptId * 3 + 2) = -x_hc * x_3d;
    J_kpts(2, ptId * 3) = -y_hc * x_3d; J_kpts(2, ptId * 3 + 1) = x_hc * x_3d;  J_kpts(2, ptId * 3 + 2) = 0 * x_3d;

    J_kpts(3, ptId * 3) = 0 * y_3d;     J_kpts(3, ptId * 3 + 1) = -z_hc * y_3d; J_kpts(3, ptId * 3 + 2) = y_hc * y_3d;
    J_kpts(4, ptId * 3) = z_hc * y_3d;  J_kpts(4, ptId * 3 + 1) = 0 * y_3d;     J_kpts(4, ptId * 3 + 2) = -x_hc * y_3d;
    J_kpts(5, ptId * 3) = -y_hc * y_3d; J_kpts(5, ptId * 3 + 1) = x_hc * y_3d;  J_kpts(5, ptId * 3 + 2) = 0 * y_3d;

    J_kpts(6, ptId * 3) = 0 * z_3d;     J_kpts(6, ptId * 3 + 1) = -z_hc * z_3d; J_kpts(6, ptId * 3 + 2) = y_hc * z_3d;
    J_kpts(7, ptId * 3) = z_hc * z_3d;  J_kpts(7, ptId * 3 + 1) = 0 * z_3d;     J_kpts(7, ptId * 3 + 2) = -x_hc * z_3d;
    J_kpts(8, ptId * 3) = -y_hc * z_3d; J_kpts(8, ptId * 3 + 1) = x_hc * z_3d;  J_kpts(8, ptId * 3 + 2) = 0 * z_3d;

    J_kpts(9, ptId * 3) = 0 * 1.0;      J_kpts(9, ptId * 3 + 1) = -z_hc * 1.0; J_kpts(9, ptId * 3 + 2) = y_hc * 1.0;
    J_kpts(10, ptId * 3) = z_hc * 1.0;  J_kpts(10, ptId * 3 + 1) = 0 * 1.0;    J_kpts(10, ptId * 3 + 2) = -x_hc * 1.0;
    J_kpts(11, ptId * 3) = -y_hc * 1.0; J_kpts(11, ptId * 3 + 1) = x_hc * 1.0; J_kpts(11, ptId * 3 + 2) = 0 * 1.0;
    
    // calculate weight for keypoints
    //kp.inv_half_var
    EigenSolver<Matrix2d> es_kp(kp.inv_half_var);   
    /*
    W_kpts(ptId * 3, ptId * 3) = weight_keypts[ptId];
    W_kpts(ptId * 3 + 1, ptId * 3 + 1) = weight_keypts[ptId];
    W_kpts(ptId * 3 + 2, ptId * 3 + 2) = weight_keypts[ptId];   
    */
    W_kpts(ptId * 3, ptId * 3) = es_kp.eigenvalues().norm() * 0.02;
    W_kpts(ptId * 3 + 1, ptId * 3 + 1) = es_kp.eigenvalues().norm() * 0.02;
    W_kpts(ptId * 3 + 2, ptId * 3 + 2) = es_kp.eigenvalues().norm() * 2; 
  }  
  // add J_kpts into Datamatrix
  M_kpts = J_kpts * W_kpts * J_kpts.transpose();  
  for (unsigned row_id = 0; row_id < 12; ++row_id) {
    for (unsigned com_id = 0; com_id < 12; ++com_id) {
      (*data_matrix)(row_id, com_id) = M_kpts(row_id, com_id) * para.gamma_kpts;     
    }
  }  
  //  Step2: add contribution of edge for data matrix
  unsigned numedges = edges->size();
  Matrix<double, 12, Dynamic> J_edge(12, numedges*3);
  Matrix<double, Dynamic, Dynamic> W_edge(numedges*3, numedges*3);
  Matrix<double, 12, 12> M_edge;
  J_edge.fill(0.0);
  W_edge.fill(0.0);
  M_edge.fill(0.0);
 
  for (unsigned edgeId = 0; edgeId < numedges; ++edgeId) {
    const EdgeVector& ev = (*edges)[edgeId];
    const Keypoint& kp_start = (*keypoints)[ev.start_id]; // edge = s - e
    const Keypoint& kp_end = (*keypoints)[ev.end_id];
    // ev.vec_pred
    // pts2d_s
    // pts3d_t
    // edge3d

    // Part1 of step2
    double x_hc, y_hc, z_hc; // denotes pts2d_s
    x_hc = kp_end.point2D_pred[0];  
    y_hc = kp_end.point2D_pred[1]; 
    z_hc = 1.0;

    double x_3d, y_3d, z_3d; // denotes edge3d
    x_3d = kp_start.point3D_gt[0] - kp_end.point3D_gt[0];
    y_3d = kp_start.point3D_gt[1] - kp_end.point3D_gt[1];
    z_3d = kp_start.point3D_gt[2] - kp_end.point3D_gt[2];

    J_edge(0, edgeId * 3) = 0 * x_3d;     J_edge(0, edgeId * 3 + 1) = -z_hc * x_3d; J_edge(0, edgeId * 3 + 2) = y_hc * x_3d;
    J_edge(1, edgeId * 3) = z_hc * x_3d;  J_edge(1, edgeId * 3 + 1) = 0 * x_3d;     J_edge(1, edgeId * 3 + 2) = -x_hc * x_3d;
    J_edge(2, edgeId * 3) = -y_hc * x_3d; J_edge(2, edgeId * 3 + 1) = x_hc * x_3d;  J_edge(2, edgeId * 3 + 2) = 0 * x_3d;

    J_edge(3, edgeId * 3) = 0 * y_3d;     J_edge(3, edgeId * 3 + 1) = -z_hc * y_3d; J_edge(3, edgeId * 3 + 2) = y_hc * y_3d;
    J_edge(4, edgeId * 3) = z_hc * y_3d;  J_edge(4, edgeId * 3 + 1) = 0 * y_3d;     J_edge(4, edgeId * 3 + 2) = -x_hc * y_3d;
    J_edge(5, edgeId * 3) = -y_hc * y_3d; J_edge(5, edgeId * 3 + 1) = x_hc * y_3d;  J_edge(5, edgeId * 3 + 2) = 0 * y_3d;

    J_edge(6, edgeId * 3) = 0 * z_3d;     J_edge(6, edgeId * 3 + 1) = -z_hc * z_3d; J_edge(6, edgeId * 3 + 2) = y_hc * z_3d;
    J_edge(7, edgeId * 3) = z_hc * z_3d;  J_edge(7, edgeId * 3 + 1) = 0 * z_3d;     J_edge(7, edgeId * 3 + 2) = -x_hc * z_3d;
    J_edge(8, edgeId * 3) = -y_hc * z_3d; J_edge(8, edgeId * 3 + 1) = x_hc * z_3d;  J_edge(8, edgeId * 3 + 2) = 0 * z_3d;

    //Part2 of step2   
    x_hc = ev.vec_pred[0];   // denote edge_pred_2d
    y_hc = ev.vec_pred[1]; 
    z_hc = 0.0;    
    
    x_3d = kp_start.point3D_gt[0];// denotes pts3d_t
    y_3d = kp_start.point3D_gt[1];
    z_3d = kp_start.point3D_gt[2];

    J_edge(0, edgeId * 3) += 0 * x_3d;     J_edge(0, edgeId * 3 + 1) += -z_hc * x_3d; J_edge(0, edgeId * 3 + 2) += y_hc * x_3d;
    J_edge(1, edgeId * 3) += z_hc * x_3d;  J_edge(1, edgeId * 3 + 1) += 0 * x_3d;     J_edge(1, edgeId * 3 + 2) += -x_hc * x_3d;
    J_edge(2, edgeId * 3) += -y_hc * x_3d; J_edge(2, edgeId * 3 + 1) += x_hc * x_3d;  J_edge(2, edgeId * 3 + 2) += 0 * x_3d;

    J_edge(3, edgeId * 3) += 0 * y_3d;     J_edge(3, edgeId * 3 + 1) += -z_hc * y_3d; J_edge(3, edgeId * 3 + 2) += y_hc * y_3d;
    J_edge(4, edgeId * 3) += z_hc * y_3d;  J_edge(4, edgeId * 3 + 1) += 0 * y_3d;     J_edge(4, edgeId * 3 + 2) += -x_hc * y_3d;
    J_edge(5, edgeId * 3) += -y_hc * y_3d; J_edge(5, edgeId * 3 + 1) += x_hc * y_3d;  J_edge(5, edgeId * 3 + 2) += 0 * y_3d;

    J_edge(6, edgeId * 3) += 0 * z_3d;     J_edge(6, edgeId * 3 + 1) += -z_hc * z_3d; J_edge(6, edgeId * 3 + 2) += y_hc * z_3d;
    J_edge(7, edgeId * 3) += z_hc * z_3d;  J_edge(7, edgeId * 3 + 1) += 0 * z_3d;     J_edge(7, edgeId * 3 + 2) += -x_hc * z_3d;
    J_edge(8, edgeId * 3) += -y_hc * z_3d; J_edge(8, edgeId * 3 + 1) += x_hc * z_3d;  J_edge(8, edgeId * 3 + 2) += 0 * z_3d;

    J_edge(9, edgeId * 3) = 0 * 1.0;      J_edge(9, edgeId * 3 + 1) = -z_hc * 1.0;  J_edge(9, edgeId * 3 + 2) = y_hc * 1.0;
    J_edge(10, edgeId * 3) = z_hc * 1.0;  J_edge(10, edgeId * 3 + 1) = 0 * 1.0;     J_edge(10, edgeId * 3 + 2) = -x_hc * 1.0;
    J_edge(11, edgeId * 3) = -y_hc * 1.0; J_edge(11, edgeId * 3 + 1) = x_hc * 1.0;  J_edge(11, edgeId * 3 + 2) = 0 * 1.0;
  
    W_edge(edgeId * 3, edgeId * 3) = weight_edges[edgeId];
    W_edge(edgeId * 3 + 1, edgeId * 3 + 1) = weight_edges[edgeId];
    W_edge(edgeId * 3 + 2, edgeId * 3 + 2) = weight_edges[edgeId];    
  }
  // add J_edge into Datamatrix
  M_edge = J_edge * W_edge * J_edge.transpose();  
  for (unsigned row_id = 0; row_id < 12; ++row_id) {
    for (unsigned com_id = 0; com_id < 12; ++com_id) {
      (*data_matrix)(row_id, com_id) += M_edge(row_id, com_id) * para.gamma_edge;       
    }   
  }

  // Step3: add contribution of symmetry for data matrix
  unsigned numSym = symcorres->size();
  Matrix<double, 9, Dynamic> J_sym(9, numSym);
  Matrix<double, Dynamic, Dynamic> W_sym(numSym, numSym);
  Matrix<double, 9, 9> M_sym; 

  J_sym.fill(0.0);
  W_sym.fill(0.0);
  M_sym.fill(0.0);
  
  const Vector3d &normal_gt = predictions.GetReflectionPlaneNormal();
  for (unsigned symId = 0; symId < numSym; ++symId) {
    const SymmetryCorres& sc = (*symcorres)[symId];
    
    double x_hc, y_hc, z_hc; // denotes sym_pred_2d
    x_hc = sc.qs1_cross_qs2[0];
    y_hc = sc.qs1_cross_qs2[1];
    z_hc = sc.qs1_cross_qs2[2];
    
    J_sym(0, symId) = x_hc * normal_gt[0]; J_sym(1, symId) = y_hc * normal_gt[0]; J_sym(2, symId) = z_hc * normal_gt[0];
    J_sym(3, symId) = x_hc * normal_gt[1]; J_sym(4, symId) = y_hc * normal_gt[1]; J_sym(5, symId) = z_hc * normal_gt[1];
    J_sym(6, symId) = x_hc * normal_gt[2]; J_sym(7, symId) = y_hc * normal_gt[2]; J_sym(8, symId) = z_hc * normal_gt[2];
    W_sym(symId, symId) = weight_symcorres[symId];
  }  // add J_kpts into Datamatrix
  M_sym = J_sym * W_sym * J_sym.transpose();
  for (unsigned row_id = 0; row_id < 12; ++row_id) {
    for (unsigned com_id = 0; com_id < 12; ++com_id) {
      (*data_matrix)(row_id, com_id) += M_sym(row_id, com_id) * para.gamma_sym;      
    }
  } 
}


// Leading eigen-space computation
void PoseRegression::LeadingEigenSpace(Matrix12d& data_matrix, const unsigned& numEigs, 
                                       const HybridPredictionContainer& predictions, vector<Vector12d>* eigenVectors,
                                       AffineXform3d* rigid_pose) {
  Matrix<double, 12, 4> eigenvector;
  // SVD of data_matrix svd.singularValues()[0]
  JacobiSVD<Matrix12d> svd(data_matrix, ComputeFullV | ComputeFullU);
  for (unsigned com_id = 0; com_id < numEigs; ++com_id) {
    for (unsigned row_id = 0; row_id < 12; ++row_id)    
      eigenvector(row_id, com_id) = svd.matrixV().col(11 - com_id)[row_id];    
  }  
  // Calculate initial coefficient(beta) of different eigenvectors  
  Map<MatrixXd> A1(eigenvector.col(0).head(9).data(), 3,3); 
  Map<MatrixXd> A2(eigenvector.col(1).head(9).data(), 3,3);
  Map<MatrixXd> A3(eigenvector.col(2).head(9).data(), 3,3);
  Map<MatrixXd> A4(eigenvector.col(3).head(9).data(), 3,3);

  Matrix3d B1 = A1.transpose() * A1;
  Matrix3d B2 = A1.transpose() * A2 + A2.transpose() * A1;
  Matrix3d B3 = A1.transpose() * A3 + A3.transpose() * A1;
  Matrix3d B4 = A2.transpose() * A2;
  Matrix3d B5 = A2.transpose() * A3 + A3.transpose() * A2;
  Matrix3d B6 = A3.transpose() * A3;

  // form linear system Cx = y to solve gamma(x) which contains coefficient 
  Matrix6d C;
  Vector6d y(1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
  Vector6d gamma;
  unsigned row_id;
  for(row_id = 0; row_id < 3; ++row_id) {   
    C(row_id, 0) = B1(0, row_id); C(row_id, 1) = B2(0, row_id); C(row_id, 2) = B3(0, row_id);
    C(row_id, 3) = B4(0, row_id); C(row_id, 4) = B5(0, row_id); C(row_id, 5) = B6(0, row_id);
  }
  for(row_id = 3; row_id < 5; ++row_id) {   
    C(row_id, 0) = B1(1, row_id - 2); C(row_id, 1) = B2(1, row_id - 2); C(row_id, 2) = B3(1, row_id - 2);
    C(row_id, 3) = B4(1, row_id - 2); C(row_id, 4) = B5(1, row_id - 2); C(row_id, 5) = B6(1, row_id - 2);
  }
  row_id = 5;
  C(row_id, 0) = B1(2,row_id - 3); C(row_id, 1) = B2(2, row_id - 3); C(row_id, 2) = B3(2, row_id - 3);
  C(row_id, 3) = B4(2,row_id - 3); C(row_id, 4) = B5(2, row_id - 3); C(row_id, 5) = B6(2, row_id - 3);
  gamma = C.lu().solve(y);
  //project gamma in to valid coefficient space
  Matrix3d Gamma;
  Gamma(0, 0) = gamma[0];  Gamma(0, 1) = gamma[1];  Gamma(0, 2) = gamma[2]; 
  Gamma(1, 0) = gamma[1];  Gamma(1, 1) = gamma[3];  Gamma(1, 2) = gamma[4]; 
  Gamma(2, 0) = gamma[2];  Gamma(2, 1) = gamma[4];  Gamma(2, 2) = gamma[5]; 
  EigenSolver<Matrix3d> es(Gamma);
  MatrixXcd D = es.eigenvalues().asDiagonal();
  D(1, 1) = 0;
  D(2, 2) = 0;
  MatrixXcd V = es.eigenvectors();
  Gamma = (V * D * V.inverse()).real();  
  // assign new gamma
  gamma[0] = Gamma(0, 0); gamma[1] = Gamma(0, 1); gamma[2] = Gamma(0, 2);
  gamma[3] = Gamma(1, 1); gamma[4] = Gamma(1, 2); gamma[5] = Gamma(2, 2);
  if (gamma[0] < 0)
    gamma *= -1.0;

  // recover initial coefficient beta from gamma with following equations:
  // gamma[0] = beta[0]^2; gamma[1] = beta[0] * beta[1]; gamma[2] = beta[0]*beta[2]
  // gamma[3] = beta[1]^2; gamma[4] = beta[1] * beta[2]; gamma[5] = beta[2]^2;
  Vector4d beta;
  double temp;
  beta[3] = 0;
  beta[0] = sqrt(gamma[0]);  
  beta[1] = gamma[1] / beta[0];
  temp = sqrt(gamma[3]);
  if (gamma[1] < 0)
    temp = temp * -1.0;
  beta[1] = 0.5 * (temp + beta[1]);
  beta[2] = gamma[4] / beta[1];
  temp = gamma[2] / beta[0];
  beta[2] = 0.5 * (beta[2] + temp);
  
  // check the validation of betas
  Matrix3d R_init;
  Vector3d t_init;
  Matrix<double, 12, 9> At1;
  Matrix<double, 12, 3> At2;
  At1 = data_matrix.leftCols(9);
  At2 = data_matrix.rightCols(3);   
  R_init = beta[0] * A1 + beta[1] * A2 + beta[2] * A3 + beta[3] * A4;  

  // check1 det(R) > 0
  if (R_init.determinant() < 0) {
    beta = beta * -1.0;
    R_init = beta[0] * A1 + beta[1] * A2 + beta[2] * A3 + beta[3] * A4;    
  }
  
  Map<VectorXd> R_vec(R_init.data(), 9, 1); 
  t_init = -1.0 * (At2.transpose() * At2).lu().solve(At2.transpose() * At1 * R_vec);

 // check2 depth > 0
  const vector<Keypoint>* keypoints = predictions.GetKeypoints();
  double z = 0;
  unsigned numKpts = keypoints->size();
  
  for (unsigned ptId = 0; ptId < numKpts; ++ptId) {  
    const Keypoint& kp = (*keypoints)[ptId];
    z = z + (R_init * kp.point3D_gt + t_init)[2];
  }
  
  if (z < 0){
    t_init = t_init * -1.0;
    R_vec = -1.0 * (At1.transpose() * At1).lu().solve(At1.transpose() * At2 * t_init);
    Map<Matrix3d> R_temp(R_vec.data(), 3,3);
    // map R_init into SO(3)
    JacobiSVD<Matrix3d> svd(R_temp, ComputeFullV | ComputeFullU);
    R_init = svd.matrixU() * svd.matrixV().transpose();
    if (R_init.determinant() < 0){
      Vector3d dia(1.0,1.0,-1.0);
      R_init = svd.matrixU() * dia.asDiagonal() * svd.matrixV().transpose();
    }
    t_init = -1.0 * (At2.transpose() * At2).lu().solve(At2.transpose() * At1 * R_vec);
  }

  // optimize beta and R_init simutaneously
  Matrix3d A = beta[0] * A1 + beta[1] * A2 + beta[2] * A3 + beta[3] * A4;
  Matrix3d R;
  for(unsigned iter = 0; iter < 10; ++iter) {
    // optimize R_init
    JacobiSVD<Matrix3d> svd(A, ComputeFullV | ComputeFullU);
    if (A.determinant() > 0) {
      R = svd.matrixU() * svd.matrixV().transpose();
    }else {
      Vector3d dia(1.0, 1.0, -1.0);
      R = svd.matrixU() * dia.asDiagonal() * svd.matrixV().transpose();
    }
    // optimize betas by forming a linear system for beta
    Matrix<double, 9, 4> D;    
    unsigned row_id;
    for(row_id = 0; row_id < 3; ++row_id) {   
      D(row_id, 0) = A1(0, row_id); D(row_id, 1) = A2(0, row_id); 
      D(row_id, 2) = A3(0, row_id); D(row_id, 3) = A4(0, row_id); 
    }
    for(row_id = 3; row_id < 6; ++row_id) {   
      D(row_id, 0) = A1(1, row_id - 3); D(row_id, 1) = A2(1, row_id - 3); 
      D(row_id, 2) = A3(1, row_id - 3); D(row_id, 3) = A4(1, row_id - 3); 
    }
    for(row_id = 6; row_id < 9; ++row_id) {   
      D(row_id, 0) = A1(2, row_id - 6); D(row_id, 1) = A2(2, row_id - 6); 
      D(row_id, 2) = A3(2, row_id - 6); D(row_id, 3) = A4(2, row_id - 6); 
    }   

    Map<VectorXd> dy(R.transpose().data(), 9, 1);
    beta = D.bdcSvd(ComputeFullV | ComputeFullU).solve(dy);    
    A = beta[0] * A1 + beta[1] * A2 + beta[2] * A3 + beta[3] * A4;
  }

  R_init = R;  
  Map<Matrix<double, 9, 1>> R_vec1(R.data(), 9, 1); 
  t_init = -1.0 * (At2.transpose() * At2).lu().solve(At2.transpose() * At1 * R_vec1); 
  // check R_init and t_init again
  z = 0;
  for (unsigned ptId = 0; ptId < numKpts; ++ptId) {  
    const Keypoint& kp = (*keypoints)[ptId];
    z = z + (R_init * kp.point3D_gt + t_init)[2];
  }
  if (z < 0) {
    t_init = t_init * -1.0;
    R_vec = -1.0 * (At1.transpose() * At1).lu().solve(At1.transpose() * At2 * t_init);
    Map<Matrix3d> R_temp1(R_vec.data(), 3, 3);

    // map R_init into SO(3)   
    JacobiSVD<Matrix3d> svd(R_temp1, ComputeFullV | ComputeFullU);
    R_init = svd.matrixU() * svd.matrixV().transpose();
    if (R_init.determinant() < 0) {
      Vector3d dia1(1.0, 1.0, -1.0);
      R_init = svd.matrixU() * dia1.asDiagonal() * svd.matrixV().transpose();
    }
    t_init = -1.0 * (At2.transpose() * At2).lu().solve(At2.transpose() * At1 * R_vec);
  }

 // Assign pose initial to rigid_pose
  for (unsigned row_id = 0; row_id < 3; ++row_id) {
    (*rigid_pose)[0][row_id] = t_init[row_id];
    for (unsigned com_id = 0; com_id < 3; ++com_id)
      (*rigid_pose)[com_id + 1][row_id] = R_init(row_id, com_id);    
  }
}
