// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "int.hpp"

namespace kissmath {
	//// forward declarations
	
	struct uint64v3;
	struct uint16v3;
	struct double3;
	struct bool3;
	struct int8v3;
	struct int16v3;
	struct uint8v3;
	struct uint3;
	struct float3;
	struct int2;
	struct int4;
	struct int64v3;
	
	struct int3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				int	x, y, z;
			};
			int		arr[3];
		};
		
		// Component indexing operator
		int& operator[] (int i);
		
		// Component indexing operator
		int const& operator[] (int i) const;
		
		
		// uninitialized constructor
		int3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		int3 (int all);
		
		// supply all components
		int3 (int x, int y, int z);
		
		// extend vector
		int3 (int2 xy, int z);
		
		// truncate vector
		int3 (int4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator int2 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool3 () const;
		
		// type cast operator
		explicit operator float3 () const;
		
		// type cast operator
		explicit operator double3 () const;
		
		// type cast operator
		explicit operator int8v3 () const;
		
		// type cast operator
		explicit operator int16v3 () const;
		
		// type cast operator
		explicit operator int64v3 () const;
		
		// type cast operator
		explicit operator uint8v3 () const;
		
		// type cast operator
		explicit operator uint16v3 () const;
		
		// type cast operator
		explicit operator uint3 () const;
		
		// type cast operator
		explicit operator uint64v3 () const;
		
		
		// componentwise arithmetic operator
		int3 operator+= (int3 r);
		
		// componentwise arithmetic operator
		int3 operator-= (int3 r);
		
		// componentwise arithmetic operator
		int3 operator*= (int3 r);
		
		// componentwise arithmetic operator
		int3 operator/= (int3 r);
		
	};
	
	//// arthmethic ops
	
	int3 operator+ (int3 v);
	
	int3 operator- (int3 v);
	
	int3 operator+ (int3 l, int3 r);
	
	int3 operator- (int3 l, int3 r);
	
	int3 operator* (int3 l, int3 r);
	
	int3 operator/ (int3 l, int3 r);
	
	
	//// bitwise ops
	
	int3 operator~ (int3 v);
	
	int3 operator& (int3 l, int3 r);
	
	int3 operator| (int3 l, int3 r);
	
	int3 operator^ (int3 l, int3 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (int3 l, int3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (int3 l, int3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (int3 l, int3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (int3 l, int3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (int3 l, int3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (int3 l, int3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (int3 l, int3 r);
	
	// componentwise ternary (c ? l : r)
	int3 select (bool3 c, int3 l, int3 r);
	
	
	//// misc ops
	// componentwise absolute
	int3 abs (int3 v);
	
	// componentwise minimum
	int3 min (int3 l, int3 r);
	
	// componentwise maximum
	int3 max (int3 l, int3 r);
	
	// componentwise clamp into range [a,b]
	int3 clamp (int3 x, int3 a, int3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	int3 clamp (int3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	int min_component (int3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	int max_component (int3 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	int3 wrap (int3 v, int3 range);
	
	// componentwise wrap
	int3 wrap (int3 v, int3 a, int3 b);
	
	
	
	//// Vector math
	
	// magnitude of vector
	float length (int3 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	int length_sqr (int3 v);
	
	// distance between points, equivalent to length(a - b)
	float distance (int3 a, int3 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float3 normalize (int3 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float3 normalizesafe (int3 v);
	
	// dot product
	int dot (int3 l, int3 r);
	
	// 3d cross product
	int3 cross (int3 l, int3 r);
	
}

