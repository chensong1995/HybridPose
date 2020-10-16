#ifndef pose_regression_parameter_search_h_
#define pose_regression_parameter_search_h_
/**************************************************************************/
// This file implements the data structure for a bounding box of a surface
// represented as a point cloud or a triangular mesh
/**************************************************************************/


#include "pose_regression.h"
//#include "dynamic_linear_algebra.h"
//#include "dynamic_linear_algebra_templcode.h"
#include <cmath>

struct ParameterSearchConfig {
public:
  ParameterSearchConfig() {
    lambda_trans = 200;
    num_samples = 300;
    max_num_iterations = 100;
    max_perturb = 0.7;
  }
  ~ParameterSearchConfig() {
  }
  double lambda_trans;
  unsigned num_samples;
  unsigned max_num_iterations;
  double max_perturb;
};

struct PRRefinePara {
  public:
   PRRefinePara() {
     beta_edge = 0.2;
     beta_sym = 1e-3;
     alpha_kpts = 1;
     alpha_edge = 0.2;
     alpha_sym = 1e-3;

   }
   ~PRRefinePara() {
   }
   double beta_edge;
   double beta_sym;
   double alpha_kpts;
   double alpha_edge;
   double alpha_sym;
};

struct PRInitPara {
 public:
  PRInitPara() {
    gamma_edge = 0.2;
    gamma_sym = 1e-3;

  }
  ~PRInitPara() {
  }  
  double gamma_edge;
  double gamma_sym;
};

class PoseRegressionParameterSearch {
 public:
   PoseRegressionParameterSearch() {
  }
  ~PoseRegressionParameterSearch() {
  }
  void Refinement(vector<HybridPredictionContainer>& training_data,
    vector<AffineXform3d>& label_data,
    const ParameterSearchConfig& para_config,
    PRRefinePara *para);
  void Initialization(vector<HybridPredictionContainer>& training_data,
    vector<AffineXform3d>& label_data,
    const ParameterSearchConfig& para_config,
    PRInitPara* para);
 private:
  bool Solve(const Matrix<double, Dynamic, Dynamic>& A, const VectorXd& b, VectorXd* x);
};
#endif
