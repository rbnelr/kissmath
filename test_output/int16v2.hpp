// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "int16.hpp"

namespace kissmath {
	//// forward declarations
	
	struct int64v2;
	struct uint64v2;
	struct int8v2;
	struct int16v3;
	struct uint2;
	struct uint8v2;
	struct double2;
	struct float2;
	struct int16v4;
	struct int2;
	struct bool2;
	struct uint16v2;
	
	struct int16v2 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				int16	x, y;
			};
			int16		arr[2];
		};
		
		// Component indexing operator
		int16& operator[] (int i);
		
		// Component indexing operator
		int16 const& operator[] (int i) const;
		
		
		// uninitialized constructor
		int16v2 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		int16v2 (int16 all);
		
		// supply all components
		int16v2 (int16 x, int16 y);
		
		// truncate vector
		int16v2 (int16v3 v);
		
		// truncate vector
		int16v2 (int16v4 v);
		
		
		//// Truncating cast operators
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool2 () const;
		
		// type cast operator
		explicit operator float2 () const;
		
		// type cast operator
		explicit operator double2 () const;
		
		// type cast operator
		explicit operator int8v2 () const;
		
		// type cast operator
		explicit operator int2 () const;
		
		// type cast operator
		explicit operator int64v2 () const;
		
		// type cast operator
		explicit operator uint8v2 () const;
		
		// type cast operator
		explicit operator uint16v2 () const;
		
		// type cast operator
		explicit operator uint2 () const;
		
		// type cast operator
		explicit operator uint64v2 () const;
		
		
		// componentwise arithmetic operator
		int16v2 operator+= (int16v2 r);
		
		// componentwise arithmetic operator
		int16v2 operator-= (int16v2 r);
		
		// componentwise arithmetic operator
		int16v2 operator*= (int16v2 r);
		
		// componentwise arithmetic operator
		int16v2 operator/= (int16v2 r);
		
	};
	
	//// arthmethic ops
	
	int16v2 operator+ (int16v2 v);
	
	int16v2 operator- (int16v2 v);
	
	int16v2 operator+ (int16v2 l, int16v2 r);
	
	int16v2 operator- (int16v2 l, int16v2 r);
	
	int16v2 operator* (int16v2 l, int16v2 r);
	
	int16v2 operator/ (int16v2 l, int16v2 r);
	
	
	//// bitwise ops
	
	int16v2 operator~ (int16v2 v);
	
	int16v2 operator& (int16v2 l, int16v2 r);
	
	int16v2 operator| (int16v2 l, int16v2 r);
	
	int16v2 operator^ (int16v2 l, int16v2 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool2 operator< (int16v2 l, int16v2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator<= (int16v2 l, int16v2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator> (int16v2 l, int16v2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator>= (int16v2 l, int16v2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator== (int16v2 l, int16v2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator!= (int16v2 l, int16v2 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (int16v2 l, int16v2 r);
	
	// componentwise ternary (c ? l : r)
	int16v2 select (bool2 c, int16v2 l, int16v2 r);
	
	
	//// misc ops
	// componentwise absolute
	int16v2 abs (int16v2 v);
	
	// componentwise minimum
	int16v2 min (int16v2 l, int16v2 r);
	
	// componentwise maximum
	int16v2 max (int16v2 l, int16v2 r);
	
	// componentwise clamp into range [a,b]
	int16v2 clamp (int16v2 x, int16v2 a, int16v2 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	int16v2 clamp (int16v2 x);
	
	// get minimum component of vector, optionally get component index via min_index
	int16 min_component (int16v2 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	int16 max_component (int16v2 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	int16v2 wrap (int16v2 v, int16v2 range);
	
	// componentwise wrap
	int16v2 wrap (int16v2 v, int16v2 a, int16v2 b);
	
	
	
	//// Vector math
	
	// magnitude of vector
	float length (int16v2 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	int16 length_sqr (int16v2 v);
	
	// distance between points, equivalent to length(a - b)
	float distance (int16v2 a, int16v2 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float2 normalize (int16v2 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float2 normalizesafe (int16v2 v);
	
	// dot product
	int16 dot (int16v2 l, int16v2 r);
	
	// 2d cross product hack for convenient 2d stuff
	// same as cross({T.name[:-2]}3(l, 0), {T.name[:-2]}3(r, 0)).z,
	// ie. the cross product of the 2d vectors on the z=0 plane in 3d space and then return the z coord of that (signed mag of cross product)
	int16 cross (int16v2 l, int16v2 r);
	
	// rotate 2d vector counterclockwise 90 deg, ie. int16v2(-y, x) which is fast
	int16v2 rotate90 (int16v2 v);
	
}

