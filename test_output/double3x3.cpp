// file was generated by kissmath.py at <TODO: add github link>
#include "double3x3.hpp"

#include "double2x2.hpp"
#include "double4x4.hpp"
#include "double2x3.hpp"
#include "double3x4.hpp"
#include "float3x3.hpp"

namespace kissmath {
	
	//// Accessors
	
	
	// get cell with row, column indecies
	double const& double3x3::get (int r, int c) const {
		return arr[c][r];
	}
	
	// get matrix column
	double3 const& double3x3::get_column (int indx) const {
		return arr[indx];
	}
	
	// get matrix row
	double3 double3x3::get_row (int indx) const {
		return double3(arr[0][indx], arr[1][indx], arr[2][indx]);
	}
	
	//// Constructors
	
	
	// uninitialized constructor
	double3x3::double3x3 () {
		
	}
	
	// supply one value for all cells
	double3x3::double3x3 (double all): 
	arr{double3(all, all, all),
		double3(all, all, all),
		double3(all, all, all)} {
		
	}
	
	// supply all cells, in row major order for readability -> c<row><column>
	double3x3::double3x3 (double c00, double c01, double c02,
						  double c10, double c11, double c12,
						  double c20, double c21, double c22): 
	arr{double3(c00, c10, c20),
		double3(c01, c11, c21),
		double3(c02, c12, c22)} {
		
	}
	
	// static rows() and columns() methods are preferred over constructors, to avoid confusion if column or row vectors are supplied to the constructor
	
	// supply all row vectors
	double3x3 double3x3::rows (double3 row0, double3 row1, double3 row2) {
		return double3x3(row0[0], row0[1], row0[2],
						 row1[0], row1[1], row1[2],
						 row2[0], row2[1], row2[2]);
	}
	
	// supply all cells in row major order
	double3x3 double3x3::rows (double c00, double c01, double c02,
							   double c10, double c11, double c12,
							   double c20, double c21, double c22) {
		return double3x3(c00, c01, c02,
						 c10, c11, c12,
						 c20, c21, c22);
	}
	
	// supply all column vectors
	double3x3 double3x3::columns (double3 col0, double3 col1, double3 col2) {
		return double3x3(col0[0], col1[0], col2[0],
						 col0[1], col1[1], col2[1],
						 col0[2], col1[2], col2[2]);
	}
	
	// supply all cells in column major order
	double3x3 double3x3::columns (double c00, double c10, double c20,
								  double c01, double c11, double c21,
								  double c02, double c12, double c22) {
		return double3x3(c00, c01, c02,
						 c10, c11, c12,
						 c20, c21, c22);
	}
	
	
	// identity matrix
	double3x3 double3x3::identity () {
		return double3x3(1,0,0,
						 0,1,0,
						 0,0,1);
	}
	
	// Casting operators
	
	
	// extend/truncate matrix of other size
	double3x3::operator double2x2 () const {
		return double2x2(arr[0][0], arr[1][0],
						 arr[0][1], arr[1][1]);
	}
	
	// extend/truncate matrix of other size
	double3x3::operator double4x4 () const {
		return double4x4(arr[0][0], arr[1][0], arr[2][0],         0,
						 arr[0][1], arr[1][1], arr[2][1],         0,
						 arr[0][2], arr[1][2], arr[2][2],         0,
						         0,         0,         0,         1);
	}
	
	// extend/truncate matrix of other size
	double3x3::operator double2x3 () const {
		return double2x3(arr[0][0], arr[1][0], arr[2][0],
						 arr[0][1], arr[1][1], arr[2][1]);
	}
	
	// extend/truncate matrix of other size
	double3x3::operator double3x4 () const {
		return double3x4(arr[0][0], arr[1][0], arr[2][0],         0,
						 arr[0][1], arr[1][1], arr[2][1],         0,
						 arr[0][2], arr[1][2], arr[2][2],         0);
	}
	
	// typecast
	double3x3::operator float3x3 () const {
		return float3x3((float)arr[0][0], (float)arr[0][1], (float)arr[0][2],
						(float)arr[1][0], (float)arr[1][1], (float)arr[1][2],
						(float)arr[2][0], (float)arr[2][1], (float)arr[2][2]);
	}
	
	// Componentwise operators; These might be useful in some cases
	
	
	// add scalar to all matrix cells
	double3x3& double3x3::operator+= (double r) {
		*this = *this + r;
		return *this;
	}
	
	// substract scalar from all matrix cells
	double3x3& double3x3::operator-= (double r) {
		*this = *this - r;
		return *this;
	}
	
	// multiply scalar with all matrix cells
	double3x3& double3x3::operator*= (double r) {
		*this = *this * r;
		return *this;
	}
	
	// divide all matrix cells by scalar
	double3x3& double3x3::operator/= (double r) {
		*this = *this / r;
		return *this;
	}
	
	// Matrix multiplication
	
	
	// matrix-matrix muliplication
	double3x3& double3x3::operator*= (double3x3 const& r) {
		*this = *this * r;
		return *this;
	}
	
	// Componentwise operators; These might be useful in some cases
	
	
	
	// componentwise matrix_cell + matrix_cell
	double3x3 operator+ (double3x3 const& l, double3x3 const& r) {
		return double3x3(l.arr[0][0] + r.arr[0][0], l.arr[1][0] + r.arr[1][0], l.arr[2][0] + r.arr[2][0],
						 l.arr[0][1] + r.arr[0][1], l.arr[1][1] + r.arr[1][1], l.arr[2][1] + r.arr[2][1],
						 l.arr[0][2] + r.arr[0][2], l.arr[1][2] + r.arr[1][2], l.arr[2][2] + r.arr[2][2]);
	}
	
	// componentwise matrix_cell + scalar
	double3x3 operator+ (double3x3 const& l, double r) {
		return double3x3(l.arr[0][0] + r, l.arr[1][0] + r, l.arr[2][0] + r,
						 l.arr[0][1] + r, l.arr[1][1] + r, l.arr[2][1] + r,
						 l.arr[0][2] + r, l.arr[1][2] + r, l.arr[2][2] + r);
	}
	
	// componentwise scalar + matrix_cell
	double3x3 operator+ (double l, double3x3 const& r) {
		return double3x3(l + r.arr[0][0], l + r.arr[1][0], l + r.arr[2][0],
						 l + r.arr[0][1], l + r.arr[1][1], l + r.arr[2][1],
						 l + r.arr[0][2], l + r.arr[1][2], l + r.arr[2][2]);
	}
	
	
	// componentwise matrix_cell - matrix_cell
	double3x3 operator- (double3x3 const& l, double3x3 const& r) {
		return double3x3(l.arr[0][0] - r.arr[0][0], l.arr[1][0] - r.arr[1][0], l.arr[2][0] - r.arr[2][0],
						 l.arr[0][1] - r.arr[0][1], l.arr[1][1] - r.arr[1][1], l.arr[2][1] - r.arr[2][1],
						 l.arr[0][2] - r.arr[0][2], l.arr[1][2] - r.arr[1][2], l.arr[2][2] - r.arr[2][2]);
	}
	
	// componentwise matrix_cell - scalar
	double3x3 operator- (double3x3 const& l, double r) {
		return double3x3(l.arr[0][0] - r, l.arr[1][0] - r, l.arr[2][0] - r,
						 l.arr[0][1] - r, l.arr[1][1] - r, l.arr[2][1] - r,
						 l.arr[0][2] - r, l.arr[1][2] - r, l.arr[2][2] - r);
	}
	
	// componentwise scalar - matrix_cell
	double3x3 operator- (double l, double3x3 const& r) {
		return double3x3(l - r.arr[0][0], l - r.arr[1][0], l - r.arr[2][0],
						 l - r.arr[0][1], l - r.arr[1][1], l - r.arr[2][1],
						 l - r.arr[0][2], l - r.arr[1][2], l - r.arr[2][2]);
	}
	
	
	// componentwise matrix_cell * matrix_cell
	double3x3 mul_componentwise (double3x3 const& l, double3x3 const& r) {
		return double3x3(l.arr[0][0] * r.arr[0][0], l.arr[1][0] * r.arr[1][0], l.arr[2][0] * r.arr[2][0],
						 l.arr[0][1] * r.arr[0][1], l.arr[1][1] * r.arr[1][1], l.arr[2][1] * r.arr[2][1],
						 l.arr[0][2] * r.arr[0][2], l.arr[1][2] * r.arr[1][2], l.arr[2][2] * r.arr[2][2]);
	}
	
	// componentwise matrix_cell * scalar
	double3x3 operator* (double3x3 const& l, double r) {
		return double3x3(l.arr[0][0] * r, l.arr[1][0] * r, l.arr[2][0] * r,
						 l.arr[0][1] * r, l.arr[1][1] * r, l.arr[2][1] * r,
						 l.arr[0][2] * r, l.arr[1][2] * r, l.arr[2][2] * r);
	}
	
	// componentwise scalar * matrix_cell
	double3x3 operator* (double l, double3x3 const& r) {
		return double3x3(l * r.arr[0][0], l * r.arr[1][0], l * r.arr[2][0],
						 l * r.arr[0][1], l * r.arr[1][1], l * r.arr[2][1],
						 l * r.arr[0][2], l * r.arr[1][2], l * r.arr[2][2]);
	}
	
	
	// componentwise matrix_cell / matrix_cell
	double3x3 div_componentwise (double3x3 const& l, double3x3 const& r) {
		return double3x3(l.arr[0][0] / r.arr[0][0], l.arr[1][0] / r.arr[1][0], l.arr[2][0] / r.arr[2][0],
						 l.arr[0][1] / r.arr[0][1], l.arr[1][1] / r.arr[1][1], l.arr[2][1] / r.arr[2][1],
						 l.arr[0][2] / r.arr[0][2], l.arr[1][2] / r.arr[1][2], l.arr[2][2] / r.arr[2][2]);
	}
	
	// componentwise matrix_cell / scalar
	double3x3 operator/ (double3x3 const& l, double r) {
		return double3x3(l.arr[0][0] / r, l.arr[1][0] / r, l.arr[2][0] / r,
						 l.arr[0][1] / r, l.arr[1][1] / r, l.arr[2][1] / r,
						 l.arr[0][2] / r, l.arr[1][2] / r, l.arr[2][2] / r);
	}
	
	// componentwise scalar / matrix_cell
	double3x3 operator/ (double l, double3x3 const& r) {
		return double3x3(l / r.arr[0][0], l / r.arr[1][0], l / r.arr[2][0],
						 l / r.arr[0][1], l / r.arr[1][1], l / r.arr[2][1],
						 l / r.arr[0][2], l / r.arr[1][2], l / r.arr[2][2]);
	}
	
	// Matrix ops
	
	
	// matrix-matrix multiply
	double3x3 operator* (double3x3 const& l, double3x3 const& r) {
		double3x3 ret;
		ret.arr[0] = l * r.arr[0];
		ret.arr[1] = l * r.arr[1];
		ret.arr[2] = l * r.arr[2];
		return ret;
	}
	
	// matrix-matrix multiply
	double3x4 operator* (double3x3 const& l, double3x4 const& r) {
		double3x4 ret;
		ret.arr[0] = l * r.arr[0];
		ret.arr[1] = l * r.arr[1];
		ret.arr[2] = l * r.arr[2];
		ret.arr[3] = l * r.arr[3];
		return ret;
	}
	
	// matrix-vector multiply
	double3 operator* (double3x3 const& l, double3 r) {
		double3 ret;
		ret[0] = l.arr[0].x * r.x + l.arr[1].x * r.y + l.arr[2].x * r.z;
		ret[1] = l.arr[0].y * r.x + l.arr[1].y * r.y + l.arr[2].y * r.z;
		ret[2] = l.arr[0].z * r.x + l.arr[1].z * r.y + l.arr[2].z * r.z;
		return ret;
	}
	
	// vector-matrix multiply
	double3 operator* (double3 l, double3x3 const& r) {
		double3 ret;
		ret[0] = l.x * r.arr[0].x + l.y * r.arr[0].y + l.z * r.arr[0].z;
		ret[1] = l.x * r.arr[1].x + l.y * r.arr[1].y + l.z * r.arr[1].z;
		ret[2] = l.x * r.arr[2].x + l.y * r.arr[2].y + l.z * r.arr[2].z;
		return ret;
	}
	
	double3x3 transpose (double3x3 const& m) {
		return double3x3::rows(m.arr[0], m.arr[1], m.arr[2]);
	}
	
	#define LETTERIFY \
	double a = mat.arr[0][0]; \
	double b = mat.arr[0][1]; \
	double c = mat.arr[0][2]; \
	double d = mat.arr[1][0]; \
	double e = mat.arr[1][1]; \
	double f = mat.arr[1][2]; \
	double g = mat.arr[2][0]; \
	double h = mat.arr[2][1]; \
	double i = mat.arr[2][2];
	
	double determinant (double3x3 const& mat) {
		LETTERIFY
		
		return +a*(e*i - f*h) -b*(d*i - f*g) +c*(d*h - e*g);
	}
	
	double3x3 inverse (double3x3 const& mat) {
		LETTERIFY
		
		double det;
		{ // clac determinate
			det = +a*(e*i - f*h) -b*(d*i - f*g) +c*(d*h - e*g);
		}
		// calc cofactor matrix
		
		double cofac_00 = e*i - f*h;
		double cofac_01 = d*i - f*g;
		double cofac_02 = d*h - e*g;
		double cofac_10 = b*i - c*h;
		double cofac_11 = a*i - c*g;
		double cofac_12 = a*h - b*g;
		double cofac_20 = b*f - c*e;
		double cofac_21 = a*f - c*d;
		double cofac_22 = a*e - b*d;
		
		double3x3 ret;
		
		double inv_det = double(1) / det;
		double ninv_det = -inv_det;
		
		ret.arr[0][0] = cofac_00 *  inv_det;
		ret.arr[0][1] = cofac_10 * ninv_det;
		ret.arr[0][2] = cofac_20 *  inv_det;
		ret.arr[1][0] = cofac_01 * ninv_det;
		ret.arr[1][1] = cofac_11 *  inv_det;
		ret.arr[1][2] = cofac_21 * ninv_det;
		ret.arr[2][0] = cofac_02 *  inv_det;
		ret.arr[2][1] = cofac_12 * ninv_det;
		ret.arr[2][2] = cofac_22 *  inv_det;
		
		return ret;
	}
	
	#undef LETTERIFY
	
}

