// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "int8.hpp"

namespace kissmath {
	//// forward declarations
	
	struct uint3;
	struct int16v3;
	struct double3;
	struct int8v4;
	struct uint64v3;
	struct uint16v3;
	struct int8v2;
	struct bool3;
	struct int64v3;
	struct float3;
	struct int3;
	struct uint8v3;
	
	struct int8v3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				int8	x, y, z;
			};
			int8		arr[3];
		};
		
		// Component indexing operator
		int8& operator[] (int i);
		
		// Component indexing operator
		int8 const& operator[] (int i) const;
		
		
		// uninitialized constructor
		int8v3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		int8v3 (int8 all);
		
		// supply all components
		int8v3 (int8 x, int8 y, int8 z);
		
		// extend vector
		int8v3 (int8v2 xy, int8 z);
		
		// truncate vector
		int8v3 (int8v4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator int8v2 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool3 () const;
		
		// type cast operator
		explicit operator float3 () const;
		
		// type cast operator
		explicit operator double3 () const;
		
		// type cast operator
		explicit operator int16v3 () const;
		
		// type cast operator
		explicit operator int3 () const;
		
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
		int8v3 operator+= (int8v3 r);
		
		// componentwise arithmetic operator
		int8v3 operator-= (int8v3 r);
		
		// componentwise arithmetic operator
		int8v3 operator*= (int8v3 r);
		
		// componentwise arithmetic operator
		int8v3 operator/= (int8v3 r);
		
	};
	
	//// arthmethic ops
	
	int8v3 operator+ (int8v3 v);
	
	int8v3 operator- (int8v3 v);
	
	int8v3 operator+ (int8v3 l, int8v3 r);
	
	int8v3 operator- (int8v3 l, int8v3 r);
	
	int8v3 operator* (int8v3 l, int8v3 r);
	
	int8v3 operator/ (int8v3 l, int8v3 r);
	
	
	//// bitwise ops
	
	int8v3 operator~ (int8v3 v);
	
	int8v3 operator& (int8v3 l, int8v3 r);
	
	int8v3 operator| (int8v3 l, int8v3 r);
	
	int8v3 operator^ (int8v3 l, int8v3 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (int8v3 l, int8v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (int8v3 l, int8v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (int8v3 l, int8v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (int8v3 l, int8v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (int8v3 l, int8v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (int8v3 l, int8v3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (int8v3 l, int8v3 r);
	
	// componentwise ternary (c ? l : r)
	int8v3 select (bool3 c, int8v3 l, int8v3 r);
	
	
	//// misc ops
	// componentwise absolute
	int8v3 abs (int8v3 v);
	
	// componentwise minimum
	int8v3 min (int8v3 l, int8v3 r);
	
	// componentwise maximum
	int8v3 max (int8v3 l, int8v3 r);
	
	// componentwise clamp into range [a,b]
	int8v3 clamp (int8v3 x, int8v3 a, int8v3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	int8v3 clamp (int8v3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	int8 min_component (int8v3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	int8 max_component (int8v3 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	int8v3 wrap (int8v3 v, int8v3 range);
	
	// componentwise wrap
	int8v3 wrap (int8v3 v, int8v3 a, int8v3 b);
	
	
	
	//// Vector math
	
	// magnitude of vector
	float length (int8v3 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	int8 length_sqr (int8v3 v);
	
	// distance between points, equivalent to length(a - b)
	float distance (int8v3 a, int8v3 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float3 normalize (int8v3 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float3 normalizesafe (int8v3 v);
	
	// dot product
	int8 dot (int8v3 l, int8v3 r);
	
	// 3d cross product
	int8v3 cross (int8v3 l, int8v3 r);
	
}

