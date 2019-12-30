#include "pose_regression_parameter_search.h"
#include <algorithm>

float rand_val() { 
  return (float) rand() / RAND_MAX;
}

void PoseRegressionParameterSearch::Refinement(
  vector<HybridPredictionContainer>& training_data,
  vector<AffineXform3d>& label_data,
  const ParameterSearchConfig& para_config,
  PRRefinePara* para) {
  // Encode the differences between the values of the samples
  // and the current sample configuration  
  Matrix<double, 5, Dynamic> mat_x_dif(5, para_config.num_samples);
  mat_x_dif.fill(0.0);
  // Encode the differences between the values of the objective functions
  // under these samples and the current objective value 
  VectorXd vec_val_dif(para_config.num_samples);
  vec_val_dif.fill(0.0);  
  VectorXd vec_val_dif_median(para_config.num_samples);
  vec_val_dif_median.fill(0.0);
  // Store the current configuration 
  VectorXd x_cur(5);
  x_cur.fill(0.0);

  
  for (unsigned iter = 0; iter < para_config.max_num_iterations; ++iter) {
    PoseRegressionPara pr_para;
    PoseRegression pr;
    pr_para.beta_edge = para->beta_edge;
    pr_para.beta_sym = para->beta_sym;
    pr_para.alpha_kpts = para->alpha_kpts;
    pr_para.alpha_edge = para->alpha_edge;
    pr_para.alpha_sym = para->alpha_sym;

    double val_cur = 0;
    VectorXd temp(training_data.size());
    temp.fill(0.0);
    for (unsigned j = 0; j < training_data.size(); ++j) {
      AffineXform3d pose = *training_data[j].GetRigidPose(); 
      pr.RefinePose(training_data[j], pr_para, &pose);
      AffineXform3d pose_gt = label_data[j];
      double loss = para_config.lambda_trans
        * (pose[0] - pose_gt[0]).squaredNorm()
        + (pose[1] - pose_gt[1]).squaredNorm()
        + (pose[2] - pose_gt[2]).squaredNorm()
        + (pose[3] - pose_gt[3]).squaredNorm();
      val_cur += sqrt(loss);
      temp[j] = loss;
    }
    sort(temp.begin(), temp.end());
    double val_cur_median = temp[temp.size()/2];
    
    x_cur[0] = pr_para.beta_edge;
    x_cur[1] = pr_para.beta_sym;
    x_cur[2] = pr_para.alpha_kpts;
    x_cur[3] = pr_para.alpha_edge;
    x_cur[4] = pr_para.alpha_sym;
    for (unsigned i = 0; i < para_config.num_samples; ++i) {
      pr_para.beta_edge = para->beta_edge * (1 + para_config.max_perturb * rand_val());
      pr_para.beta_sym = para->beta_sym * (1 + para_config.max_perturb * rand_val());
      pr_para.alpha_kpts = para->alpha_kpts * (1 + para_config.max_perturb * rand_val());
      pr_para.alpha_edge = para->alpha_edge * (1 + para_config.max_perturb * rand_val());
      pr_para.alpha_sym = para->alpha_sym * (1 + para_config.max_perturb * rand_val());

      mat_x_dif(0, i) = pr_para.beta_edge - x_cur[0];
      mat_x_dif(1, i) = pr_para.beta_sym - x_cur[1];
      mat_x_dif(2, i) = pr_para.alpha_kpts - x_cur[2];
      mat_x_dif(3, i) = pr_para.alpha_edge - x_cur[3];
      mat_x_dif(4, i) = pr_para.alpha_sym - x_cur[4];      

      vec_val_dif[i] = - val_cur;
      for (unsigned j = 0; j < training_data.size(); ++j) {
        AffineXform3d pose = *training_data[j].GetRigidPose();
        pr.RefinePose(training_data[j], pr_para, &pose);
        AffineXform3d pose_gt = label_data[j];
        double loss = para_config.lambda_trans * (pose[0] - pose_gt[0]).squaredNorm()
          + (pose[1] - pose_gt[1]).squaredNorm()
          + (pose[2] - pose_gt[2]).squaredNorm()
          + (pose[3] - pose_gt[3]).squaredNorm();
        vec_val_dif[i] += sqrt(loss);
        temp[j] = loss;
      }
      sort(temp.begin(), temp.end());
      vec_val_dif_median[i] = temp[temp.size()/2]-val_cur_median;
    }
    // Fit a qudratic surface to the samples
    unsigned dim = 20;
    //DMatrixD matJ(para_config.num_samples, dim);
    Matrix<double, Dynamic, Dynamic> matJ(dim, para_config.num_samples);
    matJ.fill(0.0);
    for (unsigned i = 0; i < para_config.num_samples; ++i) {
      for (int j = 0; j < 5; ++j) {      
        matJ(j, i) = mat_x_dif(j, i);
      }
      int off = 5;
      for (int j = 0; j < 5; ++j) {
        for (int k = j; k < 5; ++k) {       
          matJ(off, i) = mat_x_dif(j, i) * mat_x_dif(k, i);  
          off = off + 1;
        }
      }
    }  
    Matrix<double, Dynamic, Dynamic> matA(dim, dim);
    matA.fill(0.0);
    matA = matJ * matJ.transpose(); 
    VectorXd vecb(para_config.num_samples);
    vecb.fill(0.0);
    vecb = matJ * vec_val_dif_median;   
    VectorXd quad_coeffs(20);
    quad_coeffs.fill(0.0);
    Solve(matA, vecb, &quad_coeffs);
    // Parse out the quadratic approximation   
    VectorXd vecg(5);
    vecg.fill(0.0);
    Matrix<double, Dynamic, Dynamic> matH(5, 5);
    matH.fill(0.0);
    for (int i = 0; i < 5; ++i) {
      vecg[i] = -quad_coeffs[i];
    }
    int off = 5;
    for (int j = 0; j < 5; ++j) {
      for (int k = j; k < 5; ++k) {
        if (j == k) {
          matH(j, j) = 2 * quad_coeffs[off];
        } else {
          matH(k, j) = quad_coeffs[off];
          matH(j, k) = quad_coeffs[off];
        }
        off = off + 1;
      }
    }
    VectorXd dpara(5);
    dpara.fill(0.0);
    Solve(matH, vecg, &dpara);
    bool updated = false;
    double lambda = 0.0;
    
    for (unsigned search_id = 0; search_id < 20; ++search_id) {
      for (int j = 0; j < 5; ++j)
        matH(j, j) += lambda;

      Solve(matH, vecg, &dpara);
      double max_ratio = 0;
      for (int j = 0; j < 5; ++j) {
        double ratio = dpara[j] / x_cur[j];
        if (ratio < 0)
          ratio = -ratio;
        if (ratio > max_ratio)
          max_ratio = ratio;
      }
      if (max_ratio > para_config.max_perturb) {
        if (search_id == 0) {
          for (int j = 0; j < 5; ++j)
            if (abs(matH(j, j)) > lambda)
              lambda = abs(matH(j, j));
        }
        lambda = lambda * 2;
        continue;
      }
      // Check whether we will have a reduction in the objective function
      pr_para.beta_edge = para->beta_edge + dpara[0];
      pr_para.beta_sym = para->beta_sym + dpara[1];
      pr_para.alpha_kpts = para->alpha_kpts + dpara[2];
      pr_para.alpha_edge = para->alpha_edge + dpara[3];
      pr_para.alpha_sym = para->alpha_sym + dpara[4];
      double val_dif_next = -val_cur;
      for (unsigned j = 0; j < training_data.size(); ++j) {
        AffineXform3d pose = *training_data[j].GetRigidPose(); 
        pr.RefinePose(training_data[j], pr_para, &pose);
        AffineXform3d pose_gt = label_data[j];
        double loss = para_config.lambda_trans * (pose[0] - pose_gt[0]).squaredNorm()
          + (pose[1] - pose_gt[1]).squaredNorm()
          + (pose[2] - pose_gt[2]).squaredNorm()
          + (pose[3] - pose_gt[3]).squaredNorm();
        val_dif_next += sqrt(loss);
        temp[j] = loss;
      }
      sort(temp.begin(), temp.end());
      double val_dif_next_median = temp[temp.size() / 2] - val_cur_median;
      if (val_dif_next_median >= 0) {
        if (search_id == 0) {
          for (int j = 0; j < 5; ++j)
            if (abs(matH(j, j)) > lambda)
              lambda = abs(matH(j, j));
        }
        lambda = lambda * 2;
        continue;
      }
      // Now it is safe to update the parameters
      para->beta_edge += dpara[0];
      para->beta_sym += dpara[1];
      para->alpha_kpts += dpara[2];
      para->alpha_edge += dpara[3];
      para->alpha_sym += dpara[4];
      printf("pose refine search_iter = %d, loss_reduction = %f\n", iter, val_dif_next_median);
      updated = true;
      break;
    }
    
    if (updated == false) {
      printf("Parameter search for refinement module is terminated.\n");
      break;
    }
  }
}

void PoseRegressionParameterSearch::Initialization(
  vector<HybridPredictionContainer>& training_data,
  vector<AffineXform3d>& label_data,
  const ParameterSearchConfig& para_config,
  PRInitPara* para) {
// Encode the differences between the values of the samples
  // and the current sample configuration 
  Matrix<double, 2, Dynamic> mat_x_dif(2, para_config.num_samples);
  mat_x_dif.fill(0.0);
  // Encode the differences between the values of the objective functions
  // under these samples and the current objective value 
  VectorXd vec_val_dif(para_config.num_samples);
  vec_val_dif.fill(0.0);
  VectorXd vec_val_dif_median(para_config.num_samples);
  vec_val_dif_median.fill(0.0);
  // Store the current configuration
  VectorXd x_cur(2);
  x_cur.fill(0.0);

  
  for (unsigned iter = 0; iter < para_config.max_num_iterations; ++iter) {
    PoseRegressionPara pr_para;
    PoseRegression pr;
    pr_para.gamma_edge = para->gamma_edge;
    pr_para.gamma_sym = para->gamma_sym;

    double val_cur = 0;
    VectorXd temp(training_data.size());
    temp.fill(0.0);
    for (unsigned j = 0; j < training_data.size(); ++j) {
      AffineXform3d pose;
      pr.InitializePose(training_data[j], pr_para, &pose);
      AffineXform3d pose_gt = label_data[j];
      double loss = para_config.lambda_trans
        * (pose[0] - pose_gt[0]).squaredNorm() 
        + (pose[1] - pose_gt[1]).squaredNorm()
        + (pose[2] - pose_gt[2]).squaredNorm()
        + (pose[3] - pose_gt[3]).squaredNorm();
      val_cur += sqrt(loss);
      temp[j] = loss;
    }
    sort(temp.begin(), temp.end());
    double val_cur_median = temp[temp.size()/2];
    
    x_cur[0] = pr_para.gamma_edge;
    x_cur[1] = pr_para.gamma_sym;
    for (unsigned i = 0; i < para_config.num_samples; ++i) {
      pr_para.gamma_edge = para->gamma_edge * (1 + para_config.max_perturb * rand_val());
      pr_para.gamma_sym = para->gamma_sym * (1 + para_config.max_perturb * rand_val());
     
      mat_x_dif(0, i) = pr_para.gamma_edge - x_cur[0];
      mat_x_dif(1, i) = pr_para.gamma_sym - x_cur[1];     

      vec_val_dif[i] = - val_cur;
      for (unsigned j = 0; j < training_data.size(); ++j) {
        AffineXform3d pose;
        pr.InitializePose(training_data[j], pr_para, &pose);
        AffineXform3d pose_gt = label_data[j];
        double loss = para_config.lambda_trans * (pose[0] - pose_gt[0]).squaredNorm()
          + (pose[1] - pose_gt[1]).squaredNorm()
          + (pose[2] - pose_gt[2]).squaredNorm()
          + (pose[3] - pose_gt[3]).squaredNorm();
        vec_val_dif[i] += sqrt(loss);
        temp[j] = loss;
      }
      sort(temp.begin(), temp.end());
      vec_val_dif_median[i] = temp[temp.size()/2]-val_cur_median;
    }
    // Fit a qudratic surface to the samples
    unsigned dim = 5; 
    Matrix<double, Dynamic, Dynamic> matJ(dim, para_config.num_samples);
    matJ.fill(0.0);
    for (unsigned i = 0; i < para_config.num_samples; ++i) {
      for (int j = 0; j < 2; ++j) {        
        matJ(j, i) = mat_x_dif(j, i);
      }
      int off = 2;
      for (int j = 0; j < 2; ++j) {
        for (int k = j; k < 2; ++k) {  
          matJ(off, i) = mat_x_dif(j, i) * mat_x_dif(k, i); 
          off = off + 1;
        }
      }
    }
    Matrix<double, Dynamic, Dynamic> matA(dim, dim);
    matA.fill(0.0);
    matA = matJ * matJ.transpose();
    VectorXd vecb(para_config.num_samples);
    vecb.fill(0.0);
    vecb = matJ * vec_val_dif_median;
    VectorXd quad_coeffs(5);
    quad_coeffs.fill(0.0);
    Solve(matA, vecb, &quad_coeffs);
    // Parse out the quadratic approximation
    VectorXd vecg(2);
    vecg.fill(0.0);
    Matrix<double, Dynamic, Dynamic> matH(2, 2);
    matH.fill(0.0);
    for (int i = 0; i < 2; ++i) {
      vecg[i] = -quad_coeffs[i];
    }
    int off = 2;
    for (int j = 0; j < 2; ++j) {
      for (int k = j; k < 2; ++k) {
        if (j == k) {
          matH(j, j) = 2 * quad_coeffs[off];
        } else {
          matH(k, j) = quad_coeffs[off];
          matH(j, k) = quad_coeffs[off];
        }
        off = off + 1;
      }
    }
    VectorXd dpara(2);
    dpara.fill(0.0);
    Solve(matH, vecg, &dpara);
    bool updated = false;
    double lambda = 0.0;
    
    for (unsigned search_id = 0; search_id < 5; ++search_id) {
      for (int j = 0; j < 2; ++j)
        matH(j, j) += lambda;

      Solve(matH, vecg, &dpara);
      double max_ratio = 0;
      for (int j = 0; j < 2; ++j) {
        double ratio = dpara[j] / x_cur[j];
        if (ratio < 0)
          ratio = -ratio;
        if (ratio > max_ratio)
          max_ratio = ratio;
      }
      if (max_ratio > para_config.max_perturb) {
        if (search_id == 0) {
          for (int j = 0; j < 2; ++j)
            if (abs(matH(j, j)) > lambda)
              lambda = abs(matH(j, j));
        }
        lambda = lambda * 2;
        continue;
      }
      // Check whether we will have a reduction in the objective function
      pr_para.gamma_edge = para->gamma_edge + dpara[0];
      pr_para.gamma_sym = para->gamma_sym + dpara[1];
      double val_dif_next = -val_cur;
      for (unsigned j = 0; j < training_data.size(); ++j) {
        AffineXform3d pose; 
        pr.InitializePose(training_data[j], pr_para, &pose);
        AffineXform3d pose_gt = label_data[j];
        double loss = para_config.lambda_trans * (pose[0] - pose_gt[0]).squaredNorm()
          + (pose[1] - pose_gt[1]).squaredNorm()
          + (pose[2] - pose_gt[2]).squaredNorm()
          + (pose[3] - pose_gt[3]).squaredNorm();
        val_dif_next += sqrt(loss);
        temp[j] = loss;
      }
      sort(temp.begin(), temp.end());
      double val_dif_next_median = temp[temp.size() / 2] - val_cur_median;
      if (val_dif_next_median >= 0) {
        if (search_id == 0) {
          for (int j = 0; j < 2; ++j)
            if (abs(matH(j, j)) > lambda)
              lambda = abs(matH(j, j));
        }
        lambda = lambda * 2;
        continue;
      }
      // Now it is safe to update the parameters
      para->gamma_edge += dpara[0];
      para->gamma_sym += dpara[1];
      printf("pose initial search_iter = %d, loss_reduction = %f\n", iter, val_dif_next_median);
      updated = true;
      break;
    }    

    if (updated == false) {
      printf("Parameter search for initialization module is terminated.\n");
      break;
    }
  }
}

bool PoseRegressionParameterSearch::Solve(
  const Matrix<double, Dynamic, Dynamic>& A, const VectorXd& b, VectorXd* x) {
  // Using LLT factorization to solve the symmetric linear system 
  if (b.size() != x->size() || b.size() != A.cols()) 
    return false;

  //int dim = b.GetDim();
  int dim = b.size();
  const double fTolerance = 1e-20;
  //DVectorD afV;
  //afV.SetDim(dim);
  VectorXd afV(dim);
  afV.fill(0.0);
  
  //DMatrixD Lower;
  Matrix<double, Dynamic, Dynamic> Lower(dim, dim);
  Lower.fill(0.0);
  
  //Lower.SetDimension(dim, dim);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j)
      Lower(i, j) = A(i, j);
    (*x)[i] = b[i];
  }

  for (int i1 = 0; i1 < dim; ++i1) {
    for (int i0 = 0; i0 < i1; ++i0)
      //afV[i0] = Lower[i1][i0] * Lower[i0][i0];
      afV[i0] = Lower(i0, i1) * Lower(i0, i0);
    //afV[i1] = Lower[i1][i1];
    afV[i1] = Lower(i1, i1);
    for (int i0 = 0; i0 < i1; ++i0)
      //afV[i1] -= Lower[i1][i0] * afV[i0];
      afV[i1] -= Lower(i0, i1) * afV[i0];

    //Lower[i1][i1] = afV[i1];
    Lower(i1, i1) = afV[i1];
    if (fabs(afV[i1]) <= fTolerance) //singular
      return false;

    double fInv = 1.0f / afV[i1];
    for (int i0 = i1 + 1; i0 < dim; ++i0) {
      for (int i2 = 0; i2 < i1; ++i2)
        //Lower[i0][i1] -= Lower[i0][i2] * afV[i2];
        Lower(i1, i0) -= Lower(i2, i0) * afV[i2];
      //Lower[i0][i1] *= fInv;
      Lower(i1, i0) *= fInv;
    }
  }

  // Solve Ax = B.
  // Forward substitution
  for (int i0 = 0; i0 < dim; ++i0) {
    for (int i1 = 0; i1 < i0; ++i1)
      //(*x)[i0] -= Lower[i0][i1] * (*x)[i1];
      (*x)[i0] -= Lower(i1, i0) * (*x)[i1];
  }

  // Diagonal division:  Let y = L^t x, then Dy = z.  Algorithm stores
  // y terms in B vector.
  for (int i0 = 0; i0 < dim; ++i0) {
    //if (fabs(Lower[i0][i0]) <= fTolerance)
    if (fabs(Lower(i0, i0)) <= fTolerance)
      return false;
    //(*x)[i0] /= Lower[i0][i0];
    (*x)[i0] /= Lower(i0, i0);
  }

  // Back substitution:  Solve L^t x = y.  Algorithm stores x terms in
  // B vector.
  for (int i0 = 4; i0 >= 0; i0--) {
    for (int i1 = i0 + 1; i1 < dim; ++i1)
      //(*x)[i0] -= Lower[i1][i0] * (*x)[i1];
      (*x)[i0] -= Lower(i0, i1) * (*x)[i1];
  }
  return true;
}