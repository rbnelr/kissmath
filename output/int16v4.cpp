// file was generated by kissmath.py at <TODO: add github link>
#include "int16v4.hpp"

#include "int16v3.hpp"
#include "float4.hpp"
#include "uint16v4.hpp"
#include "uint4.hpp"
#include "double4.hpp"
#include "int4.hpp"
#include "int8v4.hpp"
#include "int64v4.hpp"
#include "bool4.hpp"
#include "uint64v4.hpp"
#include "int16v2.hpp"
#include "uint8v4.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef int16_t int16;
	typedef uint16_t uint16;
	typedef unsigned int uint;
	typedef int8_t int8;
	typedef int64_t int64;
	typedef uint64_t uint64;
	typedef int16_t int16;
	typedef uint8_t uint8;
	
	// Component indexing operator
	int16& int16v4::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	int16 const& int16v4::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	int16v4::int16v4 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	int16v4::int16v4 (int16 all): x{all}, y{all}, z{all}, w{all} {
		
	}
	
	// supply all components
	int16v4::int16v4 (int16 x, int16 y, int16 z, int16 w): x{x}, y{y}, z{z}, w{w} {
		
	}
	
	// extend vector
	int16v4::int16v4 (int16v2 xy, int16 z, int16 w): x{xy.x}, y{xy.y}, z{z}, w{w} {
		
	}
	
	// extend vector
	int16v4::int16v4 (int16v3 xyz, int16 w): x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {
		
	}
	
	//// Truncating cast operators
	
	
	// truncating cast operator
	int16v4::operator int16v2 () const {
		return int16v2(x, y);
	}
	
	// truncating cast operator
	int16v4::operator int16v3 () const {
		return int16v3(x, y, z);
	}
	
	//// Type cast operators
	
	
	// type cast operator
	int16v4::operator bool4 () const {
		return bool4((bool)x, (bool)y, (bool)z, (bool)w);
	}
	
	// type cast operator
	int16v4::operator float4 () const {
		return float4((float)x, (float)y, (float)z, (float)w);
	}
	
	// type cast operator
	int16v4::operator double4 () const {
		return double4((double)x, (double)y, (double)z, (double)w);
	}
	
	// type cast operator
	int16v4::operator int8v4 () const {
		return int8v4((int8)x, (int8)y, (int8)z, (int8)w);
	}
	
	// type cast operator
	int16v4::operator int4 () const {
		return int4((int)x, (int)y, (int)z, (int)w);
	}
	
	// type cast operator
	int16v4::operator int64v4 () const {
		return int64v4((int64)x, (int64)y, (int64)z, (int64)w);
	}
	
	// type cast operator
	int16v4::operator uint8v4 () const {
		return uint8v4((uint8)x, (uint8)y, (uint8)z, (uint8)w);
	}
	
	// type cast operator
	int16v4::operator uint16v4 () const {
		return uint16v4((uint16)x, (uint16)y, (uint16)z, (uint16)w);
	}
	
	// type cast operator
	int16v4::operator uint4 () const {
		return uint4((uint)x, (uint)y, (uint)z, (uint)w);
	}
	
	// type cast operator
	int16v4::operator uint64v4 () const {
		return uint64v4((uint64)x, (uint64)y, (uint64)z, (uint64)w);
	}
	
	
	// componentwise arithmetic operator
	int16v4 int16v4::operator+= (int16v4 r) {
		x += r.x;
		y += r.y;
		z += r.z;
		w += r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	int16v4 int16v4::operator-= (int16v4 r) {
		x -= r.x;
		y -= r.y;
		z -= r.z;
		w -= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	int16v4 int16v4::operator*= (int16v4 r) {
		x *= r.x;
		y *= r.y;
		z *= r.z;
		w *= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	int16v4 int16v4::operator/= (int16v4 r) {
		x /= r.x;
		y /= r.y;
		z /= r.z;
		w /= r.w;
		return *this;
	}
	
	//// arthmethic ops
	
	
	int16v4 operator+ (int16v4 v) {
		return int16v4(+v.x, +v.y, +v.z, +v.w);
	}
	
	int16v4 operator- (int16v4 v) {
		return int16v4(-v.x, -v.y, -v.z, -v.w);
	}
	
	int16v4 operator+ (int16v4 l, int16v4 r) {
		return int16v4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
	}
	
	int16v4 operator- (int16v4 l, int16v4 r) {
		return int16v4(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
	}
	
	int16v4 operator* (int16v4 l, int16v4 r) {
		return int16v4(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
	}
	
	int16v4 operator/ (int16v4 l, int16v4 r) {
		return int16v4(l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w);
	}
	
	//// bitwise ops
	
	
	int16v4 operator~ (int16v4 v) {
		return int16v4(~v.x, ~v.y, ~v.z, ~v.w);
	}
	
	int16v4 operator& (int16v4 l, int16v4 r) {
		return int16v4(l.x & r.x, l.y & r.y, l.z & r.z, l.w & r.w);
	}
	
	int16v4 operator| (int16v4 l, int16v4 r) {
		return int16v4(l.x | r.x, l.y | r.y, l.z | r.z, l.w | r.w);
	}
	
	int16v4 operator^ (int16v4 l, int16v4 r) {
		return int16v4(l.x ^ r.x, l.y ^ r.y, l.z ^ r.z, l.w ^ r.w);
	}
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool4 operator< (int16v4 l, int16v4 r) {
		return bool4(l.x < r.x, l.y < r.y, l.z < r.z, l.w < r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator<= (int16v4 l, int16v4 r) {
		return bool4(l.x <= r.x, l.y <= r.y, l.z <= r.z, l.w <= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator> (int16v4 l, int16v4 r) {
		return bool4(l.x > r.x, l.y > r.y, l.z > r.z, l.w > r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator>= (int16v4 l, int16v4 r) {
		return bool4(l.x >= r.x, l.y >= r.y, l.z >= r.z, l.w >= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator== (int16v4 l, int16v4 r) {
		return bool4(l.x == r.x, l.y == r.y, l.z == r.z, l.w == r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (int16v4 l, int16v4 r) {
		return bool4(l.x != r.x, l.y != r.y, l.z != r.z, l.w != r.w);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (int16v4 l, int16v4 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	int16v4 select (bool4 c, int16v4 l, int16v4 r) {
		return int16v4(c.x ? l.x : r.x, c.y ? l.y : r.y, c.z ? l.z : r.z, c.w ? l.w : r.w);
	}
	
	//// misc ops
	
	// componentwise absolute
	int16v4 abs (int16v4 v) {
		return int16v4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
	}
	
	// componentwise minimum
	int16v4 min (int16v4 l, int16v4 r) {
		return int16v4(min(l.x,r.x), min(l.y,r.y), min(l.z,r.z), min(l.w,r.w));
	}
	
	// componentwise maximum
	int16v4 max (int16v4 l, int16v4 r) {
		return int16v4(max(l.x,r.x), max(l.y,r.y), max(l.z,r.z), max(l.w,r.w));
	}
	
	// componentwise clamp into range [a,b]
	int16v4 clamp (int16v4 x, int16v4 a, int16v4 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	int16v4 clamp (int16v4 x) {
		return min(max(x, int16(0)), int16(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	int16 min_component (int16v4 v, int* min_index) {
		int index = 0;
		int16 min_val = v.x;	
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
	int16 max_component (int16v4 v, int* max_index) {
		int index = 0;
		int16 max_val = v.x;	
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
	int16v4 wrap (int16v4 v, int16v4 range) {
		return int16v4(wrap(v.x,range.x), wrap(v.y,range.y), wrap(v.z,range.z), wrap(v.w,range.w));
	}
	
	// componentwise wrap
	int16v4 wrap (int16v4 v, int16v4 a, int16v4 b) {
		return int16v4(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y), wrap(v.z,a.z,b.z), wrap(v.w,a.w,b.w));
	}
	
	
	//// Vector math
	
	
	// magnitude of vector
	float length (int16v4 v) {
		return sqrt((float)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w));
	}
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	int16 length_sqr (int16v4 v) {
		return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	}
	
	// distance between points, equivalent to length(a - b)
	float distance (int16v4 a, int16v4 b) {
		return length(a - b);
	}
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float4 normalize (int16v4 v) {
		return float4(v) / length(v);
	}
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float4 normalizesafe (int16v4 v) {
		float len = length(v);
		if (len == float(0)) {
			return float(0);
		}
		return float4(v) / float4(len);
	}
	
	// dot product
	int16 dot (int16v4 l, int16v4 r) {
		return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
	}
}
