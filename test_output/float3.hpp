// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "float.hpp"

namespace kissmath {
	//// forward declarations
	
	struct uint8v3;
	struct uint3;
	struct float2;
	struct double3;
	struct int64v3;
	struct float4;
	struct uint16v3;
	struct int3;
	struct uint64v3;
	struct int16v3;
	struct int8v3;
	struct bool3;
	
	struct float3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				float	x, y, z;
			};
			float		arr[3];
		};
		
		// Component indexing operator
		float& operator[] (int i);
		
		// Component indexing operator
		float const& operator[] (int i) const;
		
		
		// uninitialized constructor
		float3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		float3 (float all);
		
		// supply all components
		float3 (float x, float y, float z);
		
		// extend vector
		float3 (float2 xy, float z);
		
		// truncate vector
		float3 (float4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator float2 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool3 () const;
		
		// type cast operator
		explicit operator double3 () const;
		
		// type cast operator
		explicit operator int8v3 () const;
		
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
		float3 operator+= (float3 r);
		
		// componentwise arithmetic operator
		float3 operator-= (float3 r);
		
		// componentwise arithmetic operator
		float3 operator*= (float3 r);
		
		// componentwise arithmetic operator
		float3 operator/= (float3 r);
		
	};
	
	//// arthmethic ops
	
	float3 operator+ (float3 v);
	
	float3 operator- (float3 v);
	
	float3 operator+ (float3 l, float3 r);
	
	float3 operator- (float3 l, float3 r);
	
	float3 operator* (float3 l, float3 r);
	
	float3 operator/ (float3 l, float3 r);
	
	
	//// bitwise ops
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (float3 l, float3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (float3 l, float3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (float3 l, float3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (float3 l, float3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (float3 l, float3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (float3 l, float3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (float3 l, float3 r);
	
	// componentwise ternary (c ? l : r)
	float3 select (bool3 c, float3 l, float3 r);
	
	
	//// misc ops
	// componentwise absolute
	float3 abs (float3 v);
	
	// componentwise minimum
	float3 min (float3 l, float3 r);
	
	// componentwise maximum
	float3 max (float3 l, float3 r);
	
	// componentwise clamp into range [a,b]
	float3 clamp (float3 x, float3 a, float3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	float3 clamp (float3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	float min_component (float3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	float max_component (float3 v, int* max_index=nullptr);
	
	
	// componentwise floor
	float3 floor (float3 v);
	
	// componentwise ceil
	float3 ceil (float3 v);
	
	// componentwise round
	float3 round (float3 v);
	
	// componentwise floor to int
	int3 floori (float3 v);
	
	// componentwise ceil to int
	int3 ceili (float3 v);
	
	// componentwise round to int
	int3 roundi (float3 v);
	
	// componentwise pow
	float3 pow (float3 v, float3 e);
	
	// componentwise wrap
	float3 wrap (float3 v, float3 range);
	
	// componentwise wrap
	float3 wrap (float3 v, float3 a, float3 b);
	
	
	
	//// Angle conversion
	
	// converts degrees to radiants
	float3 radians (float3 deg);
	
	// converts radiants to degrees
	float3 degrees (float3 deg);
	
	//// Linear interpolation
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	float3 lerp (float3 a, float3 b, float3 t);
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	float3 map (float3 x, float3 in_a, float3 in_b);
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	float3 map (float3 x, float3 in_a, float3 in_b, float3 out_a, float3 out_b);
	
	
	//// Various interpolation
	
	// standard smoothstep interpolation
	float3 smoothstep (float3 x);
	
	// 3 point bezier interpolation
	float3 bezier (float3 a, float3 b, float3 c, float t);
	
	// 4 point bezier interpolation
	float3 bezier (float3 a, float3 b, float3 c, float3 d, float t);
	
	// 5 point bezier interpolation
	float3 bezier (float3 a, float3 b, float3 c, float3 d, float3 e, float t);
	
	
	//// Vector math
	
	// magnitude of vector
	float length (float3 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	float length_sqr (float3 v);
	
	// distance between points, equivalent to length(a - b)
	float distance (float3 a, float3 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float3 normalize (float3 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float3 normalizesafe (float3 v);
	
	// dot product
	float dot (float3 l, float3 r);
	
	// 3d cross product
	float3 cross (float3 l, float3 r);
	
}

