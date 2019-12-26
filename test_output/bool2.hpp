// file was generated by kissmath.py at <TODO: add github link>
#pragma once

namespace kissmath {
	//// forward declarations
	
	struct uint8v2;
	struct double2;
	struct float2;
	struct bool4;
	struct int2;
	struct int16v2;
	struct int64v2;
	struct uint16v2;
	struct bool2;
	struct uint2;
	struct int8v2;
	struct uint64v2;
	struct bool3;
	
	struct bool2 {
		union { // Union with named members and array members to allow vector[] operator, not 100% sure that this is not undefined behavoir, but I think all compilers definitely don't screw up this use case
			struct {
				bool	x, y;
			};
			bool		arr[2];
		};
		
		// Component indexing operator
		bool& operator[] (int i);
		
		// Component indexing operator
		bool const& operator[] (int i) const;
		
		
		// uninitialized constructor
		bool2 ();
		
		// sets all components to one value
		// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
		// and short initialization like float3 a = 0; works
		bool2 (bool all);
		
		// supply all components
		bool2 (bool x, bool y);
		
		// truncate vector
		bool2 (bool3 v);
		
		// truncate vector
		bool2 (bool4 v);
		
		
		//// Truncating cast operators
		
		
		//// Type cast operators
		
		// type cast operator
		explicit operator float2 () const;
		
		// type cast operator
		explicit operator double2 () const;
		
		// type cast operator
		explicit operator int8v2 () const;
		
		// type cast operator
		explicit operator int16v2 () const;
		
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
		
	};
	
	//// reducing ops
	
	// all components are true
	bool all (bool2 v);
	
	// any component is true
	bool any (bool2 v);
	
	
	//// boolean ops
	
	bool2 operator! (bool2 v);
	
	bool2 operator&& (bool2 l, bool2 r);
	
	bool2 operator|| (bool2 l, bool2 r);
	
	
	//// comparison ops
	
	// componentwise comparison returns a bool vector
	bool2 operator== (bool2 l, bool2 r);
	
	// componentwise comparison returns a bool vector
	bool2 operator!= (bool2 l, bool2 r);
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (bool2 l, bool2 r);
	
	// componentwise ternary (c ? l : r)
	bool2 select (bool2 c, bool2 l, bool2 r);
	
}
