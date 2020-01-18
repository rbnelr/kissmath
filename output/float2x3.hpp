// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "float2.hpp"
#include "float3.hpp"

namespace kissmath {
	
	//// matrix forward declarations
	struct float2x2;
	struct float3x3;
	struct float4x4;
	struct float3x4;
	struct double2x3;
	
	struct float2x3 {
		float2 arr[3]; // column major for compatibility with OpenGL
		
		//// Accessors
		
		// get cell with row, column indecies
		float const& get (int r, int c) const;
		
		// get matrix column
		float2 const& get_column (int indx) const;
		
		// get matrix row
		float3 get_row (int indx) const;
		
		
		//// Constructors
		
		// uninitialized constructor
		float2x3 ();
		
		// supply one value for all cells
		explicit float2x3 (float all);
		
		// supply all cells, in row major order for readability -> c<row><column>
		explicit float2x3 (float c00, float c01, float c02,
						   float c10, float c11, float c12);
		
		
		// static rows() and columns() methods are preferred over constructors, to avoid confusion if column or row vectors are supplied to the constructor
		// supply all row vectors
		static float2x3 rows (float3 row0, float3 row1);
		
		// supply all cells in row major order
		static float2x3 rows (float c00, float c01, float c02,
							  float c10, float c11, float c12);
		
		// supply all column vectors
		static float2x3 columns (float2 col0, float2 col1, float2 col2);
		
		// supply all cells in column major order
		static float2x3 columns (float c00, float c10,
								 float c01, float c11,
								 float c02, float c12);
		
		
		// identity matrix
		static float2x3 identity ();
		
		
		// Casting operators
		
		// extend/truncate matrix of other size
		explicit operator float2x2 () const;
		
		// extend/truncate matrix of other size
		explicit operator float3x3 () const;
		
		// extend/truncate matrix of other size
		explicit operator float4x4 () const;
		
		// extend/truncate matrix of other size
		explicit operator float3x4 () const;
		
		// typecast
		explicit operator double2x3 () const;
		
		
		// Componentwise operators; These might be useful in some cases
		
		// add scalar to all matrix cells
		float2x3& operator+= (float r);
		
		// substract scalar from all matrix cells
		float2x3& operator-= (float r);
		
		// multiply scalar with all matrix cells
		float2x3& operator*= (float r);
		
		// divide all matrix cells by scalar
		float2x3& operator/= (float r);
		
		
		// Matrix multiplication
		
		// matrix-matrix muliplication
		float2x3& operator*= (float2x3 const& r);
		
	};
	
	// Componentwise operators; These might be useful in some cases
	
	
	// componentwise matrix_cell + matrix_cell
	float2x3 operator+ (float2x3 const& l, float2x3 const& r);
	
	// componentwise matrix_cell + scalar
	float2x3 operator+ (float2x3 const& l, float r);
	
	// componentwise scalar + matrix_cell
	float2x3 operator+ (float l, float2x3 const& r);
	
	
	// componentwise matrix_cell - matrix_cell
	float2x3 operator- (float2x3 const& l, float2x3 const& r);
	
	// componentwise matrix_cell - scalar
	float2x3 operator- (float2x3 const& l, float r);
	
	// componentwise scalar - matrix_cell
	float2x3 operator- (float l, float2x3 const& r);
	
	
	// componentwise matrix_cell * matrix_cell
	float2x3 mul_componentwise (float2x3 const& l, float2x3 const& r);
	
	// componentwise matrix_cell * scalar
	float2x3 operator* (float2x3 const& l, float r);
	
	// componentwise scalar * matrix_cell
	float2x3 operator* (float l, float2x3 const& r);
	
	
	// componentwise matrix_cell / matrix_cell
	float2x3 div_componentwise (float2x3 const& l, float2x3 const& r);
	
	// componentwise matrix_cell / scalar
	float2x3 operator/ (float2x3 const& l, float r);
	
	// componentwise scalar / matrix_cell
	float2x3 operator/ (float l, float2x3 const& r);
	
	
	// Matrix ops
	
	// matrix-matrix multiply
	float2x3 operator* (float2x2 const& l, float2x3 const& r);
	
	// matrix-matrix multiply
	float2x3 operator* (float2x3 const& l, float3x3 const& r);
	
	// matrix-vector multiply
	float2 operator* (float2x3 const& l, float3 r);
	
	// vector-matrix multiply
	float3 operator* (float2 l, float2x3 const& r);
	
	
	// Matrix operation shortforms so that you can treat a 2x3 matrix as a 2x2 matrix plus translation
	
	// shortform for float2x3 * (float3x3)float2x2
	float2x3 operator* (float2x3 const& l, float2x2 const& r);
	
	// shortform for float2x3 * (float3x3)float2x3
	float2x3 operator* (float2x3 const& l, float2x3 const& r);
	
	// shortform for float2x3 * float3(float2, 1)
	float2 operator* (float2x3 const& l, float2 r);
	
}
