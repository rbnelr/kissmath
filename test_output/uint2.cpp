// file was generated by kissmath.py at <TODO: add github link>
#include "uint2.hpp"

#include "uint8v2.hpp"
#include "double2.hpp"
#include "uint3.hpp"
#include "float2.hpp"
#include "uint4.hpp"
#include "int2.hpp"
#include "int16v2.hpp"
#include "int64v2.hpp"
#include "uint16v2.hpp"
#include "bool2.hpp"
#include "uint64v2.hpp"
#include "int8v2.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef uint8_t uint8;
	typedef unsigned int uint;
	typedef unsigned int uint;
	typedef int16_t int16;
	typedef int64_t int64;
	typedef uint16_t uint16;
	typedef uint64_t uint64;
	typedef int8_t int8;
	
	// Component indexing operator
	uint& uint2::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	uint const& uint2::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	uint2::uint2 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	uint2::uint2 (uint all): x{all}, y{all} {
		
	}
	
	// supply all components
	uint2::uint2 (uint x, uint y): x{x}, y{y} {
		
	}
	
	// truncate vector
	uint2::uint2 (uint3 v): x{v.x}, y{v.y} {
		
	}
	
	// truncate vector
	uint2::uint2 (uint4 v): x{v.x}, y{v.y} {
		
	}
	
	//// Truncating cast operators
	
	
	//// Type cast operators
	
	
	// type cast operator
	uint2::operator bool2 () const {
		return bool2((bool)x, (bool)y);
	}
	
	// type cast operator
	uint2::operator float2 () const {
		return float2((float)x, (float)y);
	}
	
	// type cast operator
	uint2::operator double2 () const {
		return double2((double)x, (double)y);
	}
	
	// type cast operator
	uint2::operator int8v2 () const {
		return int8v2((int8)x, (int8)y);
	}
	
	// type cast operator
	uint2::operator int16v2 () const {
		return int16v2((int16)x, (int16)y);
	}
	
	// type cast operator
	uint2::operator int2 () const {
		return int2((int)x, (int)y);
	}
	
	// type cast operator
	uint2::operator int64v2 () const {
		return int64v2((int64)x, (int64)y);
	}
	
	// type cast operator
	uint2::operator uint8v2 () const {
		return uint8v2((uint8)x, (uint8)y);
	}
	
	// type cast operator
	uint2::operator uint16v2 () const {
		return uint16v2((uint16)x, (uint16)y);
	}
	
	// type cast operator
	uint2::operator uint64v2 () const {
		return uint64v2((uint64)x, (uint64)y);
	}
	
	
	// componentwise arithmetic operator
	uint2 uint2::operator+= (uint2 r) {
		x += r.x;
		y += r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint2 uint2::operator-= (uint2 r) {
		x -= r.x;
		y -= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint2 uint2::operator*= (uint2 r) {
		x *= r.x;
		y *= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint2 uint2::operator/= (uint2 r) {
		x /= r.x;
		y /= r.y;
		return *this;
	}
	
	//// arthmethic ops
	
	
	uint2 operator+ (uint2 v) {
		return uint2(+v.x, +v.y);
	}
	
	uint2 operator+ (uint2 l, uint2 r) {
		return uint2(l.x + r.x, l.y + r.y);
	}
	
	uint2 operator- (uint2 l, uint2 r) {
		return uint2(l.x - r.x, l.y - r.y);
	}
	
	uint2 operator* (uint2 l, uint2 r) {
		return uint2(l.x * r.x, l.y * r.y);
	}
	
	uint2 operator/ (uint2 l, uint2 r) {
		return uint2(l.x / r.x, l.y / r.y);
	}
	
	//// bitwise ops
	
	
	uint2 operator~ (uint2 v) {
		return uint2(~v.x, ~v.y);
	}
	
	uint2 operator& (uint2 l, uint2 r) {
		return uint2(l.x & r.x, l.y & r.y);
	}
	
	uint2 operator| (uint2 l, uint2 r) {
		return uint2(l.x | r.x, l.y | r.y);
	}
	
	uint2 operator^ (uint2 l, uint2 r) {
		return uint2(l.x ^ r.x, l.y ^ r.y);
	}
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool2 operator< (uint2 l, uint2 r) {
		return bool2(l.x < r.x, l.y < r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator<= (uint2 l, uint2 r) {
		return bool2(l.x <= r.x, l.y <= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator> (uint2 l, uint2 r) {
		return bool2(l.x > r.x, l.y > r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator>= (uint2 l, uint2 r) {
		return bool2(l.x >= r.x, l.y >= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator== (uint2 l, uint2 r) {
		return bool2(l.x == r.x, l.y == r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator!= (uint2 l, uint2 r) {
		return bool2(l.x != r.x, l.y != r.y);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint2 l, uint2 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	uint2 select (bool2 c, uint2 l, uint2 r) {
		return uint2(c.x ? l.x : r.x, c.y ? l.y : r.y);
	}
	
	//// misc ops
	
	// componentwise minimum
	uint2 min (uint2 l, uint2 r) {
		return uint2(min(l.x,r.x), min(l.y,r.y));
	}
	
	// componentwise maximum
	uint2 max (uint2 l, uint2 r) {
		return uint2(max(l.x,r.x), max(l.y,r.y));
	}
	
	// componentwise clamp into range [a,b]
	uint2 clamp (uint2 x, uint2 a, uint2 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint2 clamp (uint2 x) {
		return min(max(x, uint(0)), uint(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	uint min_component (uint2 v, int* min_index) {
		int index = 0;
		uint min_val = v.x;	
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
	uint max_component (uint2 v, int* max_index) {
		int index = 0;
		uint max_val = v.x;	
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
	uint2 wrap (uint2 v, uint2 range) {
		return uint2(wrap(v.x,range.x), wrap(v.y,range.y));
	}
	
	// componentwise wrap
	uint2 wrap (uint2 v, uint2 a, uint2 b) {
		return uint2(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y));
	}
	
	
	//// Vector math
	
}
