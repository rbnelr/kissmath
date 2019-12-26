// file was generated by kissmath.py at <TODO: add github link>
#include "uint8v3.hpp"

#include "uint8v2.hpp"
#include "uint3.hpp"
#include "float3.hpp"
#include "double3.hpp"
#include "int64v3.hpp"
#include "uint16v3.hpp"
#include "int3.hpp"
#include "uint64v3.hpp"
#include "int16v3.hpp"
#include "int8v3.hpp"
#include "bool3.hpp"
#include "uint8v4.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef uint8_t uint8;
	typedef unsigned int uint;
	typedef int64_t int64;
	typedef uint16_t uint16;
	typedef uint64_t uint64;
	typedef int16_t int16;
	typedef int8_t int8;
	typedef uint8_t uint8;
	
	// Component indexing operator
	uint8& uint8v3::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	uint8 const& uint8v3::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	uint8v3::uint8v3 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	uint8v3::uint8v3 (uint8 all): x{all}, y{all}, z{all} {
		
	}
	
	// supply all components
	uint8v3::uint8v3 (uint8 x, uint8 y, uint8 z): x{x}, y{y}, z{z} {
		
	}
	
	// extend vector
	uint8v3::uint8v3 (uint8v2 xy, uint8 z): x{xy.x}, y{xy.y}, z{z} {
		
	}
	
	// truncate vector
	uint8v3::uint8v3 (uint8v4 v): x{v.x}, y{v.y}, z{v.z} {
		
	}
	
	//// Truncating cast operators
	
	
	// truncating cast operator
	uint8v3::operator uint8v2 () const {
		return uint8v2(x, y);
	}
	
	//// Type cast operators
	
	
	// type cast operator
	uint8v3::operator bool3 () const {
		return bool3((bool)x, (bool)y, (bool)z);
	}
	
	// type cast operator
	uint8v3::operator float3 () const {
		return float3((float)x, (float)y, (float)z);
	}
	
	// type cast operator
	uint8v3::operator double3 () const {
		return double3((double)x, (double)y, (double)z);
	}
	
	// type cast operator
	uint8v3::operator int8v3 () const {
		return int8v3((int8)x, (int8)y, (int8)z);
	}
	
	// type cast operator
	uint8v3::operator int16v3 () const {
		return int16v3((int16)x, (int16)y, (int16)z);
	}
	
	// type cast operator
	uint8v3::operator int3 () const {
		return int3((int)x, (int)y, (int)z);
	}
	
	// type cast operator
	uint8v3::operator int64v3 () const {
		return int64v3((int64)x, (int64)y, (int64)z);
	}
	
	// type cast operator
	uint8v3::operator uint16v3 () const {
		return uint16v3((uint16)x, (uint16)y, (uint16)z);
	}
	
	// type cast operator
	uint8v3::operator uint3 () const {
		return uint3((uint)x, (uint)y, (uint)z);
	}
	
	// type cast operator
	uint8v3::operator uint64v3 () const {
		return uint64v3((uint64)x, (uint64)y, (uint64)z);
	}
	
	
	// componentwise arithmetic operator
	uint8v3 uint8v3::operator+= (uint8v3 r) {
		x += r.x;
		y += r.y;
		z += r.z;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v3 uint8v3::operator-= (uint8v3 r) {
		x -= r.x;
		y -= r.y;
		z -= r.z;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v3 uint8v3::operator*= (uint8v3 r) {
		x *= r.x;
		y *= r.y;
		z *= r.z;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint8v3 uint8v3::operator/= (uint8v3 r) {
		x /= r.x;
		y /= r.y;
		z /= r.z;
		return *this;
	}
	
	//// arthmethic ops
	
	
	uint8v3 operator+ (uint8v3 v) {
		return uint8v3(+v.x, +v.y, +v.z);
	}
	
	uint8v3 operator+ (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x + r.x, l.y + r.y, l.z + r.z);
	}
	
	uint8v3 operator- (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x - r.x, l.y - r.y, l.z - r.z);
	}
	
	uint8v3 operator* (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x * r.x, l.y * r.y, l.z * r.z);
	}
	
	uint8v3 operator/ (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x / r.x, l.y / r.y, l.z / r.z);
	}
	
	//// bitwise ops
	
	
	uint8v3 operator~ (uint8v3 v) {
		return uint8v3(~v.x, ~v.y, ~v.z);
	}
	
	uint8v3 operator& (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x & r.x, l.y & r.y, l.z & r.z);
	}
	
	uint8v3 operator| (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x | r.x, l.y | r.y, l.z | r.z);
	}
	
	uint8v3 operator^ (uint8v3 l, uint8v3 r) {
		return uint8v3(l.x ^ r.x, l.y ^ r.y, l.z ^ r.z);
	}
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool3 operator< (uint8v3 l, uint8v3 r) {
		return bool3(l.x < r.x, l.y < r.y, l.z < r.z);
	}
	
	// componentwise comparison returns a bool vector
	bool3 operator<= (uint8v3 l, uint8v3 r) {
		return bool3(l.x <= r.x, l.y <= r.y, l.z <= r.z);
	}
	
	// componentwise comparison returns a bool vector
	bool3 operator> (uint8v3 l, uint8v3 r) {
		return bool3(l.x > r.x, l.y > r.y, l.z > r.z);
	}
	
	// componentwise comparison returns a bool vector
	bool3 operator>= (uint8v3 l, uint8v3 r) {
		return bool3(l.x >= r.x, l.y >= r.y, l.z >= r.z);
	}
	
	// componentwise comparison returns a bool vector
	bool3 operator== (uint8v3 l, uint8v3 r) {
		return bool3(l.x == r.x, l.y == r.y, l.z == r.z);
	}
	
	// componentwise comparison returns a bool vector
	bool3 operator!= (uint8v3 l, uint8v3 r) {
		return bool3(l.x != r.x, l.y != r.y, l.z != r.z);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint8v3 l, uint8v3 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	uint8v3 select (bool3 c, uint8v3 l, uint8v3 r) {
		return uint8v3(c.x ? l.x : r.x, c.y ? l.y : r.y, c.z ? l.z : r.z);
	}
	
	//// misc ops
	
	// componentwise minimum
	uint8v3 min (uint8v3 l, uint8v3 r) {
		return uint8v3(min(l.x,r.x), min(l.y,r.y), min(l.z,r.z));
	}
	
	// componentwise maximum
	uint8v3 max (uint8v3 l, uint8v3 r) {
		return uint8v3(max(l.x,r.x), max(l.y,r.y), max(l.z,r.z));
	}
	
	// componentwise clamp into range [a,b]
	uint8v3 clamp (uint8v3 x, uint8v3 a, uint8v3 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint8v3 clamp (uint8v3 x) {
		return min(max(x, uint8(0)), uint8(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	uint8 min_component (uint8v3 v, int* min_index) {
		int index = 0;
		uint8 min_val = v.x;	
		for (int i=1; i<3; ++i) {
			if (v.arr[i] <= min_val) {
				index = i;
				min_val = v.arr[i];
			}
		}
		if (min_index) *min_index = index;
		return min_val;
	}
	
	// get maximum component of vector, optionally get component index via max_index
	uint8 max_component (uint8v3 v, int* max_index) {
		int index = 0;
		uint8 max_val = v.x;	
		for (int i=1; i<3; ++i) {
			if (v.arr[i] >= max_val) {
				index = i;
				max_val = v.arr[i];
			}
		}
		if (max_index) *max_index = index;
		return max_val;
	}
	
	
	// componentwise wrap
	uint8v3 wrap (uint8v3 v, uint8v3 range) {
		return uint8v3(wrap(v.x,range.x), wrap(v.y,range.y), wrap(v.z,range.z));
	}
	
	// componentwise wrap
	uint8v3 wrap (uint8v3 v, uint8v3 a, uint8v3 b) {
		return uint8v3(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y), wrap(v.z,a.z,b.z));
	}
	
	
	//// Vector math
	
}

