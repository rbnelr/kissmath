// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "double4.hpp"
#include "double3.hpp"

namespace kissmath {
	
	//// matrix forward declarations
	struct double2x2;
	struct double3x3;
	struct double4x4;
	struct double2x3;
	struct float3x4;
	
	struct double3x4 {
		double3 arr[4]; // column major for compatibility with OpenGL
		
		//// Accessors
		
		// get cell with row, column indecies
		double const& get (int r, int c) const;
		
		// get matrix column
		double3 const& get_column (int indx) const;
		
		// get matrix row
		double4 get_row (int indx) const;
		
		
		//// Constructors
		
		// uninitialized constructor
		double3x4 ();
		
		// supply one value for all cells
		explicit double3x4 (double all);
		
		// supply all cells, in row major order for readability -> c<row><column>
		explicit double3x4 (double c00, double c01, double c02, double c03,
							double c10, double c11, double c12, double c13,
							double c20, double c21, double c22, double c23);
		
		
		// static rows() and columns() methods are preferred over constructors, to avoid confusion if column or row vectors are supplied to the constructor
		// supply all row vectors
		static double3x4 rows (double4 row0, double4 row1, double4 row2);
		
		// supply all cells in row major order
		static double3x4 rows (double c00, double c01, double c02, double c03,
							   double c10, double c11, double c12, double c13,
							   double c20, double c21, double c22, double c23);
		
		// supply all column vectors
		static double3x4 columns (double3 col0, double3 col1, double3 col2, double3 col3);
		
		// supply all cells in column major order
		static double3x4 columns (double c00, double c10, double c20,
								  double c01, double c11, double c21,
								  double c02, double c12, double c22,
								  double c03, double c13, double c23);
		
		
		// identity matrix
		static double3x4 identity ();
		
		
		// Casting operators
		
		// extend/truncate matrix of other size
		explicit operator double2x2 () const;
		
		// extend/truncate matrix of other size
		explicit operator double3x3 () const;
		
		// extend/truncate matrix of other size
		explicit operator double4x4 () const;
		
		// extend/truncate matrix of other size
		explicit operator double2x3 () const;
		
		// typecast
		explicit operator float3x4 () const;
		
		
		// Componentwise operators; These might be useful in some cases
		
		// add scalar to all matrix cells
		double3x4& operator+= (double r);
		
		// substract scalar from all matrix cells
		double3x4& operator-= (double r);
		
		// multiply scalar with all matrix cells
		double3x4& operator*= (double r);
		
		// divide all matrix cells by scalar
		double3x4& operator/= (double r);
		
		
		// Matrix multiplication
		
		// matrix-matrix muliplication
		double3x4& operator*= (double3x4 const& r);
		
	};
	
	// Componentwise operators; These might be useful in some cases
	
	
	// componentwise matrix_cell + matrix_cell
	double3x4 operator+ (double3x4 const& l, double3x4 const& r);
	
	// componentwise matrix_cell + scalar
	double3x4 operator+ (double3x4 const& l, double r);
	
	// componentwise scalar + matrix_cell
	double3x4 operator+ (double l, double3x4 const& r);
	
	
	// componentwise matrix_cell - matrix_cell
	double3x4 operator- (double3x4 const& l, double3x4 const& r);
	
	// componentwise matrix_cell - scalar
	double3x4 operator- (double3x4 const& l, double r);
	
	// componentwise scalar - matrix_cell
	double3x4 operator- (double l, double3x4 const& r);
	
	
	// componentwise matrix_cell * matrix_cell
	double3x4 mul_componentwise (double3x4 const& l, double3x4 const& r);
	
	// componentwise matrix_cell * scalar
	double3x4 operator* (double3x4 const& l, double r);
	
	// componentwise scalar * matrix_cell
	double3x4 operator* (double l, double3x4 const& r);
	
	
	// componentwise matrix_cell / matrix_cell
	double3x4 div_componentwise (double3x4 const& l, double3x4 const& r);
	
	// componentwise matrix_cell / scalar
	double3x4 operator/ (double3x4 const& l, double r);
	
	// componentwise scalar / matrix_cell
	double3x4 operator/ (double l, double3x4 const& r);
	
	
	// Matrix ops
	
	// matrix-matrix multiply
	double3x4 operator* (double3x4 const& l, double4x4 const& r);
	
	// matrix-vector multiply
	double3 operator* (double3x4 const& l, double4 r);
	
	// vector-matrix multiply
	double4 operator* (double3 l, double3x4 const& r);
	
	
	// Matrix operation shortforms so that you can treat a 3x4 matrix as a 3x3 matrix plus translation
	
	// shortform for double3x4 * (double4x4)double3x3
	double3x4 operator* (double3x4 const& l, double3x3 const& r);
	
	// shortform for double3x4 * (double4x4)double3x4
	double3x4 operator* (double3x4 const& l, double3x4 const& r);
	
	// shortform for double3x4 * double4(double3, 1)
	double3 operator* (double3x4 const& l, double3 r);
	
}
