// file was generated by kissmath.py at <TODO: add github link>
#pragma once

namespace kissmath {
	//// forward declarations
	
	struct float4;
	struct bool3;
	struct uint64v4;
	struct uint8v4;
	struct double4;
	struct uint16v4;
	struct int64v4;
	struct uint4;
	struct int16v4;
	struct int8v4;
	struct int4;
	struct bool4;
	struct bool2;
	
	struct bool4 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				bool	x, y, z, w;
			};
			bool		arr[4];
		};
		
		// Component indexing operator
		bool& operator[] (int i);
		
		// Component indexing operator
		bool const& operator[] (int i) const;
		
		
		// uninitialized constructor
		bool4 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		bool4 (bool all);
		
		// supply all components
		bool4 (bool x, bool y, bool z, bool w);
		
		// extend vector
		bool4 (bool2 xy, bool z, bool w);
		
		// extend vector
		bool4 (bool3 xyz, bool w);
		
		
		//// Truncating cast operators
		
		// truncating cast operator
		explicit operator bool2 () const;
		
		// truncating cast operator
		explicit operator bool3 () const;
		
		
		//// Type cast operators
		
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
		
		// type cast operator
		explicit operator uint64v4 () const;
		
	};
	
	//// reducing ops
	
	// all components are true
	bool all (bool4 v);
	
	// any component is true
	bool any (bool4 v);
	
	
	//// boolean ops
	
	bool4 operator! (bool4 v);
	
	bool4 operator&& (bool4 l, bool4 r);
	
	bool4 operator|| (bool4 l, bool4 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool4 operator== (bool4 l, bool4 r);
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (bool4 l, bool4 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (bool4 l, bool4 r);
	
	// componentwise ternary (c ? l : r)
	bool4 select (bool4 c, bool4 l, bool4 r);
	
}

