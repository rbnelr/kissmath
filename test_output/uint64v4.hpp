// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "uint64.hpp"

namespace kissmath {
	//// forward declarations
	
	struct double4;
	struct uint16v4;
	struct int64v4;
	struct int8v4;
	struct int16v4;
	struct bool4;
	struct uint4;
	struct int4;
	struct float4;
	struct uint64v2;
	struct uint64v3;
	struct uint8v4;
	
	struct uint64v4 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				uint64	x, y, z, w;
			};
			uint64		arr[4];
		};
		
		// Component indexing operator
		uint64& operator[] (int i);
		
		// Component indexing operator
		uint64 const& operator[] (int i) const;
		
		
		// uninitialized constructor
		uint64v4 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		uint64v4 (uint64 all);
		
		// supply all components
		uint64v4 (uint64 x, uint64 y, uint64 z, uint64 w);
		
		// extend vector
		uint64v4 (uint64v2 xy, uint64 z, uint64 w);
		
		// extend vector
		uint64v4 (uint64v3 xyz, uint64 w);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator uint64v2 () const;
		
		// truncating cast operator
		explicit operator uint64v3 () const;
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator bool4 () const;
		
		// type cast operator
		explicit operator float4 () const;
		
		// type cast operator
		explicit operator double4 () const;
		
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
		
		
		// componentwise arithmetic operator
		uint64v4 operator+= (uint64v4 r);
		
		// componentwise arithmetic operator
		uint64v4 operator-= (uint64v4 r);
		
		// componentwise arithmetic operator
		uint64v4 operator*= (uint64v4 r);
		
		// componentwise arithmetic operator
		uint64v4 operator/= (uint64v4 r);
		
	};
	
	//// arthmethic ops
	
	uint64v4 operator+ (uint64v4 v);
	
	uint64v4 operator+ (uint64v4 l, uint64v4 r);
	
	uint64v4 operator- (uint64v4 l, uint64v4 r);
	
	uint64v4 operator* (uint64v4 l, uint64v4 r);
	
	uint64v4 operator/ (uint64v4 l, uint64v4 r);
	
	
	//// bitwise ops
	
	uint64v4 operator~ (uint64v4 v);
	
	uint64v4 operator& (uint64v4 l, uint64v4 r);
	
	uint64v4 operator| (uint64v4 l, uint64v4 r);
	
	uint64v4 operator^ (uint64v4 l, uint64v4 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool4 operator< (uint64v4 l, uint64v4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator<= (uint64v4 l, uint64v4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator> (uint64v4 l, uint64v4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator>= (uint64v4 l, uint64v4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator== (uint64v4 l, uint64v4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (uint64v4 l, uint64v4 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint64v4 l, uint64v4 r);
	
	// componentwise ternary (c ? l : r)
	uint64v4 select (bool4 c, uint64v4 l, uint64v4 r);
	
	
	//// misc ops
	// componentwise minimum
	uint64v4 min (uint64v4 l, uint64v4 r);
	
	// componentwise maximum
	uint64v4 max (uint64v4 l, uint64v4 r);
	
	// componentwise clamp into range [a,b]
	uint64v4 clamp (uint64v4 x, uint64v4 a, uint64v4 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint64v4 clamp (uint64v4 x);
	
	// get minimum component of vector, optionally get component index via min_index
	uint64 min_component (uint64v4 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	uint64 max_component (uint64v4 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	uint64v4 wrap (uint64v4 v, uint64v4 range);
	
	// componentwise wrap
	uint64v4 wrap (uint64v4 v, uint64v4 a, uint64v4 b);
	
	
	
	//// Vector math
	
}

