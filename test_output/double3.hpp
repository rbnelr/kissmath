// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "double.hpp"

namespace kissmath {
	//// forward declarations
	
	struct double4;
	struct double2;
	struct uint8v3;
	struct uint3;
	struct float3;
	struct int64v3;
	struct uint16v3;
	struct int3;
	struct uint64v3;
	struct int16v3;
	struct int8v3;
	struct bool3;
	
	struct double3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				double	x, y, z;
			};
			double		arr[3];
		};
		
		// Component indexing operator
		double& operator[] (int i);
		
		// Component indexing operator
		double const& operator[] (int i) const;
		
		
		// uninitialized constructor
		double3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		double3 (double all);
		
		// supply all components
		double3 (double x, double y, double z);
		
		// extend vector
		double3 (double2 xy, double z);
		
		// truncate vector
		double3 (double4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator double2 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool3 () const;
		
		// type cast operator
		explicit operator float3 () const;
		
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
		double3 operator+= (double3 r);
		
		// componentwise arithmetic operator
		double3 operator-= (double3 r);
		
		// componentwise arithmetic operator
		double3 operator*= (double3 r);
		
		// componentwise arithmetic operator
		double3 operator/= (double3 r);
		
	};
	
	//// arthmethic ops
	
	double3 operator+ (double3 v);
	
	double3 operator- (double3 v);
	
	double3 operator+ (double3 l, double3 r);
	
	double3 operator- (double3 l, double3 r);
	
	double3 operator* (double3 l, double3 r);
	
	double3 operator/ (double3 l, double3 r);
	
	
	//// bitwise ops
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (double3 l, double3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (double3 l, double3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (double3 l, double3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (double3 l, double3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (double3 l, double3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (double3 l, double3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (double3 l, double3 r);
	
	// componentwise ternary (c ? l : r)
	double3 select (bool3 c, double3 l, double3 r);
	
	
	//// misc ops
	// componentwise absolute
	double3 abs (double3 v);
	
	// componentwise minimum
	double3 min (double3 l, double3 r);
	
	// componentwise maximum
	double3 max (double3 l, double3 r);
	
	// componentwise clamp into range [a,b]
	double3 clamp (double3 x, double3 a, double3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	double3 clamp (double3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	double min_component (double3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	double max_component (double3 v, int* max_index=nullptr);
	
	
	// componentwise floor
	double3 floor (double3 v);
	
	// componentwise ceil
	double3 ceil (double3 v);
	
	// componentwise round
	double3 round (double3 v);
	
	// componentwise floor to int
	int64v3 floori (double3 v);
	
	// componentwise ceil to int
	int64v3 ceili (double3 v);
	
	// componentwise round to int
	int64v3 roundi (double3 v);
	
	// componentwise pow
	double3 pow (double3 v, double3 e);
	
	// componentwise wrap
	double3 wrap (double3 v, double3 range);
	
	// componentwise wrap
	double3 wrap (double3 v, double3 a, double3 b);
	
	
	
	//// Angle conversion
	
	// converts degrees to radiants
	double3 radians (double3 deg);
	
	// converts radiants to degrees
	double3 degrees (double3 deg);
	
	//// Linear interpolation
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	double3 lerp (double3 a, double3 b, double3 t);
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	double3 map (double3 x, double3 in_a, double3 in_b);
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	double3 map (double3 x, double3 in_a, double3 in_b, double3 out_a, double3 out_b);
	
	
	//// Various interpolation
	
	// standard smoothstep interpolation
	double3 smoothstep (double3 x);
	
	// 3 point bezier interpolation
	double3 bezier (double3 a, double3 b, double3 c, double t);
	
	// 4 point bezier interpolation
	double3 bezier (double3 a, double3 b, double3 c, double3 d, double t);
	
	// 5 point bezier interpolation
	double3 bezier (double3 a, double3 b, double3 c, double3 d, double3 e, double t);
	
	
	//// Vector math
	
	// magnitude of vector
	double length (double3 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	double length_sqr (double3 v);
	
	// distance between points, equivalent to length(a - b)
	double distance (double3 a, double3 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	double3 normalize (double3 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	double3 normalizesafe (double3 v);
	
	// dot product
	double dot (double3 l, double3 r);
	
	// 3d cross product
	double3 cross (double3 l, double3 r);
	
}

