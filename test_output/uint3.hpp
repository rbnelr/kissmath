// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "uint.hpp"

namespace kissmath {
	//// forward declarations
	
	struct uint4;
	struct int8v3;
	struct int16v3;
	struct double3;
	struct uint64v3;
	struct uint16v3;
	struct bool3;
	struct uint2;
	struct int64v3;
	struct float3;
	struct int3;
	struct uint8v3;
	
	struct uint3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				uint	x, y, z;
			};
			uint		arr[3];
		};
		
		// Component indexing operator
		uint& operator[] (int i);
		
		// Component indexing operator
		uint const& operator[] (int i) const;
		
		
		// uninitialized constructor
		uint3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		uint3 (uint all);
		
		// supply all components
		uint3 (uint x, uint y, uint z);
		
		// extend vector
		uint3 (uint2 xy, uint z);
		
		// truncate vector
		uint3 (uint4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator uint2 () const;
		
		
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
		explicit operator int3 () const;
		
		// type cast operator
		explicit operator int64v3 () const;
		
		// type cast operator
		explicit operator uint8v3 () const;
		
		// type cast operator
		explicit operator uint16v3 () const;
		
		// type cast operator
		explicit operator uint64v3 () const;
		
		
		// componentwise arithmetic operator
		uint3 operator+= (uint3 r);
		
		// componentwise arithmetic operator
		uint3 operator-= (uint3 r);
		
		// componentwise arithmetic operator
		uint3 operator*= (uint3 r);
		
		// componentwise arithmetic operator
		uint3 operator/= (uint3 r);
		
	};
	
	//// arthmethic ops
	
	uint3 operator+ (uint3 v);
	
	uint3 operator+ (uint3 l, uint3 r);
	
	uint3 operator- (uint3 l, uint3 r);
	
	uint3 operator* (uint3 l, uint3 r);
	
	uint3 operator/ (uint3 l, uint3 r);
	
	
	//// bitwise ops
	
	uint3 operator~ (uint3 v);
	
	uint3 operator& (uint3 l, uint3 r);
	
	uint3 operator| (uint3 l, uint3 r);
	
	uint3 operator^ (uint3 l, uint3 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (uint3 l, uint3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (uint3 l, uint3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (uint3 l, uint3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (uint3 l, uint3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (uint3 l, uint3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (uint3 l, uint3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint3 l, uint3 r);
	
	// componentwise ternary (c ? l : r)
	uint3 select (bool3 c, uint3 l, uint3 r);
	
	
	//// misc ops
	// componentwise minimum
	uint3 min (uint3 l, uint3 r);
	
	// componentwise maximum
	uint3 max (uint3 l, uint3 r);
	
	// componentwise clamp into range [a,b]
	uint3 clamp (uint3 x, uint3 a, uint3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint3 clamp (uint3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	uint min_component (uint3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	uint max_component (uint3 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	uint3 wrap (uint3 v, uint3 range);
	
	// componentwise wrap
	uint3 wrap (uint3 v, uint3 a, uint3 b);
	
	
	
	//// Vector math
	
}

