// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "double.hpp"

namespace kissmath {
	//// forward declarations
	
	struct float4;
	struct double3;
	struct uint64v4;
	struct uint8v4;
	struct double2;
	struct uint16v4;
	struct int64v4;
	struct uint4;
	struct int16v4;
	struct int8v4;
	struct int4;
	struct bool4;
	
	struct double4 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				double	x, y, z, w;
			};
			double		arr[4];
		};
		
		// Component indexing operator
		double& operator[] (int i);
		
		// Component indexing operator
		double const& operator[] (int i) const;
		
		
		// uninitialized constructor
		double4 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		double4 (double all);
		
		// supply all components
		double4 (double x, double y, double z, double w);
		
		// extend vector
		double4 (double2 xy, double z, double w);
		
		// extend vector
		double4 (double3 xyz, double w);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator double2 () const;
		
		// truncating cast operator
		explicit operator double3 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool4 () const;
		
		// type cast operator
		explicit operator float4 () const;
		
		// type cast operator
		explicit operator int8v4 () const;
		
		// type cast operator
		explicit operator int16v4 () const;
		
		// type cast operator
		explicit operator int4 () const;
		
		// type cast operator
		explicit operator int64v4 () const;
		
		// type cast operator
		explicit operator uint8v4 () const;
		
		// type cast operator
		explicit operator uint16v4 () const;
		
		// type cast operator
		explicit operator uint4 () const;
		
		// type cast operator
		explicit operator uint64v4 () const;
		
		
		// componentwise arithmetic operator
		double4 operator+= (double4 r);
		
		// componentwise arithmetic operator
		double4 operator-= (double4 r);
		
		// componentwise arithmetic operator
		double4 operator*= (double4 r);
		
		// componentwise arithmetic operator
		double4 operator/= (double4 r);
		
	};
	
	//// arthmethic ops
	
	double4 operator+ (double4 v);
	
	double4 operator- (double4 v);
	
	double4 operator+ (double4 l, double4 r);
	
	double4 operator- (double4 l, double4 r);
	
	double4 operator* (double4 l, double4 r);
	
	double4 operator/ (double4 l, double4 r);
	
	
	//// bitwise ops
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool4 operator< (double4 l, double4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator<= (double4 l, double4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator> (double4 l, double4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator>= (double4 l, double4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator== (double4 l, double4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (double4 l, double4 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (double4 l, double4 r);
	
	// componentwise ternary (c ? l : r)
	double4 select (bool4 c, double4 l, double4 r);
	
	
	//// misc ops
	// componentwise absolute
	double4 abs (double4 v);
	
	// componentwise minimum
	double4 min (double4 l, double4 r);
	
	// componentwise maximum
	double4 max (double4 l, double4 r);
	
	// componentwise clamp into range [a,b]
	double4 clamp (double4 x, double4 a, double4 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	double4 clamp (double4 x);
	
	// get minimum component of vector, optionally get component index via min_index
	double min_component (double4 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	double max_component (double4 v, int* max_index=nullptr);
	
	
	// componentwise floor
	double4 floor (double4 v);
	
	// componentwise ceil
	double4 ceil (double4 v);
	
	// componentwise round
	double4 round (double4 v);
	
	// componentwise floor to int
	int64v4 floori (double4 v);
	
	// componentwise ceil to int
	int64v4 ceili (double4 v);
	
	// componentwise round to int
	int64v4 roundi (double4 v);
	
	// componentwise pow
	double4 pow (double4 v, double4 e);
	
	// componentwise wrap
	double4 wrap (double4 v, double4 range);
	
	// componentwise wrap
	double4 wrap (double4 v, double4 a, double4 b);
	
	
	
	//// Angle conversion
	
	// converts degrees to radiants
	double4 to_radians (double4 deg);
	
	// converts radiants to degrees
	double4 to_degrees (double4 rad);
	
	// converts degrees to radiants
	// shortform to make degree literals more readable
	double4 deg (double4 deg);
	
	//// Linear interpolation
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	double4 lerp (double4 a, double4 b, double4 t);
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	double4 map (double4 x, double4 in_a, double4 in_b);
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	double4 map (double4 x, double4 in_a, double4 in_b, double4 out_a, double4 out_b);
	
	
	//// Various interpolation
	
	// standard smoothstep interpolation
	double4 smoothstep (double4 x);
	
	// 3 point bezier interpolation
	double4 bezier (double4 a, double4 b, double4 c, double t);
	
	// 4 point bezier interpolation
	double4 bezier (double4 a, double4 b, double4 c, double4 d, double t);
	
	// 5 point bezier interpolation
	double4 bezier (double4 a, double4 b, double4 c, double4 d, double4 e, double t);
	
	
	//// Vector math
	
	// magnitude of vector
	double length (double4 v);
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	double length_sqr (double4 v);
	
	// distance between points, equivalent to length(a - b)
	double distance (double4 a, double4 b);
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	double4 normalize (double4 v);
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	double4 normalizesafe (double4 v);
	
	// dot product
	double dot (double4 l, double4 r);
	
}

