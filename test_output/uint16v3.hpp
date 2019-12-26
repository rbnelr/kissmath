// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include "uint16.hpp"

namespace kissmath {
	//// forward declarations
	
	struct int16v3;
	struct int8v3;
	struct bool3;
	struct uint16v4;
	struct uint64v3;
	struct int64v3;
	struct uint8v3;
	struct uint3;
	struct float3;
	struct uint16v2;
	struct int3;
	struct double3;
	
	struct uint16v3 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				uint16	x, y, z;
			};
			uint16		arr[3];
		};
		
		// Component indexing operator
		uint16& operator[] (int i);
		
		// Component indexing operator
		uint16 const& operator[] (int i) const;
		
		
		// uninitialized constructor
		uint16v3 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		uint16v3 (uint16 all);
		
		// supply all components
		uint16v3 (uint16 x, uint16 y, uint16 z);
		
		// extend vector
		uint16v3 (uint16v2 xy, uint16 z);
		
		// truncate vector
		uint16v3 (uint16v4 v);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator uint16v2 () const;
		
		
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
		explicit operator uint3 () const;
		
		// type cast operator
		explicit operator uint64v3 () const;
		
		
		// componentwise arithmetic operator
		uint16v3 operator+= (uint16v3 r);
		
		// componentwise arithmetic operator
		uint16v3 operator-= (uint16v3 r);
		
		// componentwise arithmetic operator
		uint16v3 operator*= (uint16v3 r);
		
		// componentwise arithmetic operator
		uint16v3 operator/= (uint16v3 r);
		
	};
	
	//// arthmethic ops
	
	uint16v3 operator+ (uint16v3 v);
	
	uint16v3 operator+ (uint16v3 l, uint16v3 r);
	
	uint16v3 operator- (uint16v3 l, uint16v3 r);
	
	uint16v3 operator* (uint16v3 l, uint16v3 r);
	
	uint16v3 operator/ (uint16v3 l, uint16v3 r);
	
	
	//// bitwise ops
	
	uint16v3 operator~ (uint16v3 v);
	
	uint16v3 operator& (uint16v3 l, uint16v3 r);
	
	uint16v3 operator| (uint16v3 l, uint16v3 r);
	
	uint16v3 operator^ (uint16v3 l, uint16v3 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool3 operator< (uint16v3 l, uint16v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (uint16v3 l, uint16v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator> (uint16v3 l, uint16v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (uint16v3 l, uint16v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator== (uint16v3 l, uint16v3 r);
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (uint16v3 l, uint16v3 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint16v3 l, uint16v3 r);
	
	// componentwise ternary (c ? l : r)
	uint16v3 select (bool3 c, uint16v3 l, uint16v3 r);
	
	
	//// misc ops
	// componentwise minimum
	uint16v3 min (uint16v3 l, uint16v3 r);
	
	// componentwise maximum
	uint16v3 max (uint16v3 l, uint16v3 r);
	
	// componentwise clamp into range [a,b]
	uint16v3 clamp (uint16v3 x, uint16v3 a, uint16v3 b);
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint16v3 clamp (uint16v3 x);
	
	// get minimum component of vector, optionally get component index via min_index
	uint16 min_component (uint16v3 v, int* min_index=nullptr);
	
	// get maximum component of vector, optionally get component index via max_index
	uint16 max_component (uint16v3 v, int* max_index=nullptr);
	
	
	// componentwise wrap
	uint16v3 wrap (uint16v3 v, uint16v3 range);
	
	// componentwise wrap
	uint16v3 wrap (uint16v3 v, uint16v3 a, uint16v3 b);
	
	
	
	//// Vector math
	
}

