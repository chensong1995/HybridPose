#ifndef affine_transformation_h_
#define affine_transformation_h_

#include <Eigen/Dense>
using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 12, 12> Matrix12d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 12, 1> Vector12d;

class AffineXform3d {   
 private:
	// Translation is the vector
	// followed by a dim x dim matrix
	Vector3d v[4];
 public:
	// ---- constructors
	inline AffineXform3d();
	inline AffineXform3d(double *velocity);

	inline Vector3d& operator[](const unsigned &index);
	inline const Vector3d& operator[](const unsigned &index) const;

	//  ----  operators +, -, *, /
	inline AffineXform3d operator+(const AffineXform3d &op) const;
	inline AffineXform3d operator-(const AffineXform3d &op) const;
	inline AffineXform3d operator-() const;
	inline AffineXform3d operator*(const double &s) const;
	inline AffineXform3d operator*(const AffineXform3d &op) const ;

	inline AffineXform3d operator/(const double &s) const;

	//  ---- operators +=, -=, *=, /=
	inline AffineXform3d operator+=(const AffineXform3d &op);
	inline AffineXform3d operator-=(const AffineXform3d &op);
	inline AffineXform3d operator*=(const double &op);
	inline AffineXform3d operator*=(const AffineXform3d &op);
	inline AffineXform3d operator/=(const double &op);
	inline AffineXform3d operator=(const AffineXform3d &op);
	inline AffineXform3d	Inverse();

	inline void Initialize();
	inline void SetZero();
	inline double Det();
  inline void GetData(Vector12d* vec);
  inline void SetData(const Vector12d& vec);
	// ---- multiplying with vectors
	inline Vector3d operator*(const Vector3d &v) const;
};

#include "affine_transformation_templcode.h"

#endif
