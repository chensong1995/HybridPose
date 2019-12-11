#include "keypoint.h"

Keypoint::Keypoint() {
  point2D_pred[0] = point2D_pred[1] = 0.;
  point3D_gt[0] = point3D_gt[1] = point3D_gt[2] = 0.;
  inv_half_var(0, 0) = inv_half_var(1, 1) = 1.;
  inv_half_var(0, 1) = inv_half_var(1, 0) = 0.;
}

Keypoint::~Keypoint() {

}
