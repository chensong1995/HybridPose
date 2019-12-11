#ifndef symmetry_corres_h_
#define symmetry_corres_h_

#include <Eigen/Dense>
using namespace Eigen;

struct SymmetryCorres {
 public:
   SymmetryCorres();
   ~SymmetryCorres();
   // The cross product between qs1 and qs2
   // The constraint is qs1_cross_qs2'*R*n_gt = 0
   Vector3d qs1_cross_qs2;
   double weight;
};
#endif
