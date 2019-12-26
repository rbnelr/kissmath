// file was generated by kissmath.py at <TODO: add github link>
#include "uint8v2.hpp"

#include "int16v2.hpp"
#include "int64v2.hpp"
#include "uint64v2.hpp"
#include "int8v2.hpp"
#include "uint2.hpp"
#include "uint8v4.hpp"
#include "uint8v3.hpp"
#include "double2.hpp"
#include "float2.hpp"
#include "int2.hpp"
#include "bool2.hpp"
#include "uint16v2.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef int16_t int16;
	typedef int64_t int64;
	typedef uint64_t uint64;
	typedef int8_t int8;
	typedef unsigned int uint;
	typedef uint8_t uint8;
	typedef uint8_t uint8;
	typedef uint16_t uint16;
	
	// Component indexing operator
	uint8& uint8v2::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	uint8 const& uint8v2::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	uint8v2::uint8v2 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	uint8v2::uint8v2 (uint8 all): x{all}, y{all} {
		
	}
	
	// supply all components
	uint8v2::uint8v2 (uint8 x, uint8 y): x{x}, y{y} {
		
	}
	
	// truncate vector
	uint8v2::uint8v2 (uint8v3 v): x{v.x}, y{v.y} {
		
	}
	
	// truncate vector
	uint8v2::uint8v2 (uint8v4 v): x{v.x}, y{v.y} {
		
	}
	
	//// Truncating cast operators
	
	
	//// Type cast operators
	
	
	// type cast operator
	uint8v2::operator bool2 () const {
		return bool2((bool)x, (bool)y);
	}
	
	// type cast operator
	uint8v2::operator float2 () const {
		return float2((float)x, (float)y);
	}
	
	// type cast operator
	uint8v2::operator double2 () const {
		return double2((double)x, (double)y);
	}
	
	// type cast operator
	uint8v2::operator int8v2 () const {
		return int8v2((int8)x, (int8)y);
	}
	
	// type cast operator
	uint8v2::operator int16v2 () const {
		return int16v2((int16)x, (int16)y);
	}
	
	// type cast operator
	uint8v2::operator int2 () const {
		return int2((int)x, (int)y);
	}
	
	// type cast operator
	uint8v2::operator int64v2 () const {
		return int64v2((int64)x, (int64)y);
	}
	
	// type cast operator
	uint8v2::operator uint16v2 () const {
		return uint16v2((uint16)x, (uint16)y);
	}
	
	// type cast operator
	uint8v2::operator uint2 () const {
		return uint2((uint)x, (uint)y);
	}
	
	// type cast operator
	uint8v2::operator uint64v2 () const {
		return uint64v2((uint64)x, (uint64)y);
	}
	
	
	// componentwise arithmetic operator
	uint8v2 uint8v2::operator+= (uint8v2 r) {
		x += r.x;
		y += r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v2 uint8v2::operator-= (uint8v2 r) {
		x -= r.x;
		y -= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v2 uint8v2::operator*= (uint8v2 r) {
		x *= r.x;
		y *= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v2 uint8v2::operator/= (uint8v2 r) {
		x /= r.x;
		y /= r.y;
		return *this;
	}
	
	//// arthmethic ops
	
	
	uint8v2 operator+ (uint8v2 v) {
		return uint8v2(+v.x, +v.y);
	}
	
	uint8v2 operator+ (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x + r.x, l.y + r.y);
	}
	
	uint8v2 operator- (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x - r.x, l.y - r.y);
	}
	
	uint8v2 operator* (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x * r.x, l.y * r.y);
	}
	
	uint8v2 operator/ (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x / r.x, l.y / r.y);
	}
	
	//// bitwise ops
	
	
	uint8v2 operator~ (uint8v2 v) {
		return uint8v2(~v.x, ~v.y);
	}
	
	uint8v2 operator& (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x & r.x, l.y & r.y);
	}
	
	uint8v2 operator| (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x | r.x, l.y | r.y);
	}
	
	uint8v2 operator^ (uint8v2 l, uint8v2 r) {
		return uint8v2(l.x ^ r.x, l.y ^ r.y);
	}
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool2 operator< (uint8v2 l, uint8v2 r) {
		return bool2(l.x < r.x, l.y < r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator<= (uint8v2 l, uint8v2 r) {
		return bool2(l.x <= r.x, l.y <= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator> (uint8v2 l, uint8v2 r) {
		return bool2(l.x > r.x, l.y > r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator>= (uint8v2 l, uint8v2 r) {
		return bool2(l.x >= r.x, l.y >= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator== (uint8v2 l, uint8v2 r) {
		return bool2(l.x == r.x, l.y == r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator!= (uint8v2 l, uint8v2 r) {
		return bool2(l.x != r.x, l.y != r.y);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint8v2 l, uint8v2 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	uint8v2 select (bool2 c, uint8v2 l, uint8v2 r) {
		return uint8v2(c.x ? l.x : r.x, c.y ? l.y : r.y);
	}
	
	//// misc ops
	
	// componentwise minimum
	uint8v2 min (uint8v2 l, uint8v2 r) {
		return uint8v2(min(l.x,r.x), min(l.y,r.y));
	}
	
	// componentwise maximum
	uint8v2 max (uint8v2 l, uint8v2 r) {
		return uint8v2(max(l.x,r.x), max(l.y,r.y));
	}
	
	// componentwise clamp into range [a,b]
	uint8v2 clamp (uint8v2 x, uint8v2 a, uint8v2 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint8v2 clamp (uint8v2 x) {
		return min(max(x, uint8(0)), uint8(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	uint8 min_component (uint8v2 v, int* min_index) {
		int index = 0;
		uint8 min_val = v.x;	
		for (int i=1; i<2; ++i) {
			if (v.arr[i] <= min_val) {
				index = i;
				min_val = v.arr[i];
			}
		}
		if (min_index) *min_index = index;
		return min_val;
	}
	
	// get maximum component of vector, optionally get component index via max_index
	uint8 max_component (uint8v2 v, int* max_index) {
		int index = 0;
		uint8 max_val = v.x;	
		for (int i=1; i<2; ++i) {
			if (v.arr[i] >= max_val) {
				index = i;
				max_val = v.arr[i];
			}
		}
		if (max_index) *max_index = index;
		return max_val;
	}
	
	
	// componentwise wrap
	uint8v2 wrap (uint8v2 v, uint8v2 range) {
		return uint8v2(wrap(v.x,range.x), wrap(v.y,range.y));
	}
	
	// componentwise wrap
	uint8v2 wrap (uint8v2 v, uint8v2 a, uint8v2 b) {
		return uint8v2(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y));
	}
	
	
	//// Vector math
	
}

