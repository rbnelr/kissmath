// file was generated by kissmath.py at <TODO: add github link>
#include "uint4.hpp"

#include "int64v4.hpp"
#include "uint16v4.hpp"
#include "bool4.hpp"
#include "uint2.hpp"
#include "uint64v4.hpp"
#include "int16v4.hpp"
#include "uint8v4.hpp"
#include "uint3.hpp"
#include "int8v4.hpp"
#include "int4.hpp"
#include "double4.hpp"
#include "float4.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef int64_t int64;
	typedef uint16_t uint16;
	typedef unsigned int uint;
	typedef uint64_t uint64;
	typedef int16_t int16;
	typedef uint8_t uint8;
	typedef unsigned int uint;
	typedef int8_t int8;
	
	// Component indexing operator
	uint& uint4::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	uint const& uint4::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	uint4::uint4 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	uint4::uint4 (uint all): x{all}, y{all}, z{all}, w{all} {
		
	}
	
	// supply all components
	uint4::uint4 (uint x, uint y, uint z, uint w): x{x}, y{y}, z{z}, w{w} {
		
	}
	
	// extend vector
	uint4::uint4 (uint2 xy, uint z, uint w): x{xy.x}, y{xy.y}, z{z}, w{w} {
		
	}
	
	// extend vector
	uint4::uint4 (uint3 xyz, uint w): x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {
		
	}
	
	//// Truncating cast operators
	
	
	// truncating cast operator
	uint4::operator uint2 () const {
		return uint2(x, y);
	}
	
	// truncating cast operator
	uint4::operator uint3 () const {
		return uint3(x, y, z);
	}
	
	//// Type cast operators
	
	
	// type cast operator
	uint4::operator bool4 () const {
		return bool4((bool)x, (bool)y, (bool)z, (bool)w);
	}
	
	// type cast operator
	uint4::operator float4 () const {
		return float4((float)x, (float)y, (float)z, (float)w);
	}
	
	// type cast operator
	uint4::operator double4 () const {
		return double4((double)x, (double)y, (double)z, (double)w);
	}
	
	// type cast operator
	uint4::operator int8v4 () const {
		return int8v4((int8)x, (int8)y, (int8)z, (int8)w);
	}
	
	// type cast operator
	uint4::operator int16v4 () const {
		return int16v4((int16)x, (int16)y, (int16)z, (int16)w);
	}
	
	// type cast operator
	uint4::operator int4 () const {
		return int4((int)x, (int)y, (int)z, (int)w);
	}
	
	// type cast operator
	uint4::operator int64v4 () const {
		return int64v4((int64)x, (int64)y, (int64)z, (int64)w);
	}
	
	// type cast operator
	uint4::operator uint8v4 () const {
		return uint8v4((uint8)x, (uint8)y, (uint8)z, (uint8)w);
	}
	
	// type cast operator
	uint4::operator uint16v4 () const {
		return uint16v4((uint16)x, (uint16)y, (uint16)z, (uint16)w);
	}
	
	// type cast operator
	uint4::operator uint64v4 () const {
		return uint64v4((uint64)x, (uint64)y, (uint64)z, (uint64)w);
	}
	
	
	// componentwise arithmetic operator
	uint4 uint4::operator+= (uint4 r) {
		x += r.x;
		y += r.y;
		z += r.z;
		w += r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint4 uint4::operator-= (uint4 r) {
		x -= r.x;
		y -= r.y;
		z -= r.z;
		w -= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint4 uint4::operator*= (uint4 r) {
		x *= r.x;
		y *= r.y;
		z *= r.z;
		w *= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	uint4 uint4::operator/= (uint4 r) {
		x /= r.x;
		y /= r.y;
		z /= r.z;
		w /= r.w;
		return *this;
	}
	
	//// arthmethic ops
	
	
	uint4 operator+ (uint4 v) {
		return uint4(+v.x, +v.y, +v.z, +v.w);
	}
	
	uint4 operator+ (uint4 l, uint4 r) {
		return uint4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
	}
	
	uint4 operator- (uint4 l, uint4 r) {
		return uint4(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
	}
	
	uint4 operator* (uint4 l, uint4 r) {
		return uint4(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
	}
	
	uint4 operator/ (uint4 l, uint4 r) {
		return uint4(l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w);
	}
	
	//// bitwise ops
	
	
	uint4 operator~ (uint4 v) {
		return uint4(~v.x, ~v.y, ~v.z, ~v.w);
	}
	
	uint4 operator& (uint4 l, uint4 r) {
		return uint4(l.x & r.x, l.y & r.y, l.z & r.z, l.w & r.w);
	}
	
	uint4 operator| (uint4 l, uint4 r) {
		return uint4(l.x | r.x, l.y | r.y, l.z | r.z, l.w | r.w);
	}
	
	uint4 operator^ (uint4 l, uint4 r) {
		return uint4(l.x ^ r.x, l.y ^ r.y, l.z ^ r.z, l.w ^ r.w);
	}
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool4 operator< (uint4 l, uint4 r) {
		return bool4(l.x < r.x, l.y < r.y, l.z < r.z, l.w < r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator<= (uint4 l, uint4 r) {
		return bool4(l.x <= r.x, l.y <= r.y, l.z <= r.z, l.w <= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator> (uint4 l, uint4 r) {
		return bool4(l.x > r.x, l.y > r.y, l.z > r.z, l.w > r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator>= (uint4 l, uint4 r) {
		return bool4(l.x >= r.x, l.y >= r.y, l.z >= r.z, l.w >= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator== (uint4 l, uint4 r) {
		return bool4(l.x == r.x, l.y == r.y, l.z == r.z, l.w == r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (uint4 l, uint4 r) {
		return bool4(l.x != r.x, l.y != r.y, l.z != r.z, l.w != r.w);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (uint4 l, uint4 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	uint4 select (bool4 c, uint4 l, uint4 r) {
		return uint4(c.x ? l.x : r.x, c.y ? l.y : r.y, c.z ? l.z : r.z, c.w ? l.w : r.w);
	}
	
	//// misc ops
	
	// componentwise minimum
	uint4 min (uint4 l, uint4 r) {
		return uint4(min(l.x,r.x), min(l.y,r.y), min(l.z,r.z), min(l.w,r.w));
	}
	
	// componentwise maximum
	uint4 max (uint4 l, uint4 r) {
		return uint4(max(l.x,r.x), max(l.y,r.y), max(l.z,r.z), max(l.w,r.w));
	}
	
	// componentwise clamp into range [a,b]
	uint4 clamp (uint4 x, uint4 a, uint4 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	uint4 clamp (uint4 x) {
		return min(max(x, uint(0)), uint(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	uint min_component (uint4 v, int* min_index) {
		int index = 0;
		uint min_val = v.x;	
		for (int i=1; i<4; ++i) {
			if (v.arr[i] <= min_val) {
				index = i;
				min_val = v.arr[i];
			}
		}
		if (min_index) *min_index = index;
		return min_val;
	}
	
	// get maximum component of vector, optionally get component index via max_index
	uint max_component (uint4 v, int* max_index) {
		int index = 0;
		uint max_val = v.x;	
		for (int i=1; i<4; ++i) {
			if (v.arr[i] >= max_val) {
				index = i;
				max_val = v.arr[i];
			}
		}
		if (max_index) *max_index = index;
		return max_val;
	}
	
	
	// componentwise wrap
	uint4 wrap (uint4 v, uint4 range) {
		return uint4(wrap(v.x,range.x), wrap(v.y,range.y), wrap(v.z,range.z), wrap(v.w,range.w));
	}
	
	// componentwise wrap
	uint4 wrap (uint4 v, uint4 a, uint4 b) {
		return uint4(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y), wrap(v.z,a.z,b.z), wrap(v.w,a.w,b.w));
	}
	
	
	//// Vector math
	
}

