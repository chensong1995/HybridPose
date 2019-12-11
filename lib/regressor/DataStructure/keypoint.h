#ifndef keypoint_h_
#define keypoint_h_

#include <Eigen/Dense>
using namespace Eigen;

struct Keypoint {
 public:
   Keypoint();
   ~Keypoint();
   // Ground-truth 3D position
   Vector3d point3D_gt;
   // Predicted 2D position
   Vector2d point2D_pred;
   // Predicted inverse half variance
   Matrix2d inv_half_var;
};
#endif
