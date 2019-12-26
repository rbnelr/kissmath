// file was generated by kissmath.py at <TODO: add github link>
#include "float4.hpp"

#include "float2.hpp"
#include "uint8v4.hpp"
#include "uint4.hpp"
#include "int64v4.hpp"
#include "int8v4.hpp"
#include "bool4.hpp"
#include "uint16v4.hpp"
#include "uint64v4.hpp"
#include "int16v4.hpp"
#include "float3.hpp"
#include "double4.hpp"
#include "int4.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef uint8_t uint8;
	typedef unsigned int uint;
	typedef int64_t int64;
	typedef int8_t int8;
	typedef uint16_t uint16;
	typedef uint64_t uint64;
	typedef int16_t int16;
	
	// Component indexing operator
	float& float4::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	float const& float4::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	float4::float4 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	float4::float4 (float all): x{all}, y{all}, z{all}, w{all} {
		
	}
	
	// supply all components
	float4::float4 (float x, float y, float z, float w): x{x}, y{y}, z{z}, w{w} {
		
	}
	
	// extend vector
	float4::float4 (float2 xy, float z, float w): x{xy.x}, y{xy.y}, z{z}, w{w} {
		
	}
	
	// extend vector
	float4::float4 (float3 xyz, float w): x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {
		
	}
	
	//// Truncating cast operators
	
	
	// truncating cast operator
	float4::operator float2 () const {
		return float2(x, y);
	}
	
	// truncating cast operator
	float4::operator float3 () const {
		return float3(x, y, z);
	}
	
	//// Type cast operators
	
	
	// type cast operator
	float4::operator bool4 () const {
		return bool4((bool)x, (bool)y, (bool)z, (bool)w);
	}
	
	// type cast operator
	float4::operator double4 () const {
		return double4((double)x, (double)y, (double)z, (double)w);
	}
	
	// type cast operator
	float4::operator int8v4 () const {
		return int8v4((int8)x, (int8)y, (int8)z, (int8)w);
	}
	
	// type cast operator
	float4::operator int16v4 () const {
		return int16v4((int16)x, (int16)y, (int16)z, (int16)w);
	}
	
	// type cast operator
	float4::operator int4 () const {
		return int4((int)x, (int)y, (int)z, (int)w);
	}
	
	// type cast operator
	float4::operator int64v4 () const {
		return int64v4((int64)x, (int64)y, (int64)z, (int64)w);
	}
	
	// type cast operator
	float4::operator uint8v4 () const {
		return uint8v4((uint8)x, (uint8)y, (uint8)z, (uint8)w);
	}
	
	// type cast operator
	float4::operator uint16v4 () const {
		return uint16v4((uint16)x, (uint16)y, (uint16)z, (uint16)w);
	}
	
	// type cast operator
	float4::operator uint4 () const {
		return uint4((uint)x, (uint)y, (uint)z, (uint)w);
	}
	
	// type cast operator
	float4::operator uint64v4 () const {
		return uint64v4((uint64)x, (uint64)y, (uint64)z, (uint64)w);
	}
	
	
	// componentwise arithmetic operator
	float4 float4::operator+= (float4 r) {
		x += r.x;
		y += r.y;
		z += r.z;
		w += r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	float4 float4::operator-= (float4 r) {
		x -= r.x;
		y -= r.y;
		z -= r.z;
		w -= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	float4 float4::operator*= (float4 r) {
		x *= r.x;
		y *= r.y;
		z *= r.z;
		w *= r.w;
		return *this;
	}
	
	// componentwise arithmetic operator
	float4 float4::operator/= (float4 r) {
		x /= r.x;
		y /= r.y;
		z /= r.z;
		w /= r.w;
		return *this;
	}
	
	//// arthmethic ops
	
	
	float4 operator+ (float4 v) {
		return float4(+v.x, +v.y, +v.z, +v.w);
	}
	
	float4 operator- (float4 v) {
		return float4(-v.x, -v.y, -v.z, -v.w);
	}
	
	float4 operator+ (float4 l, float4 r) {
		return float4(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
	}
	
	float4 operator- (float4 l, float4 r) {
		return float4(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w);
	}
	
	float4 operator* (float4 l, float4 r) {
		return float4(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w);
	}
	
	float4 operator/ (float4 l, float4 r) {
		return float4(l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w);
	}
	
	//// bitwise ops
	
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool4 operator< (float4 l, float4 r) {
		return bool4(l.x < r.x, l.y < r.y, l.z < r.z, l.w < r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator<= (float4 l, float4 r) {
		return bool4(l.x <= r.x, l.y <= r.y, l.z <= r.z, l.w <= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator> (float4 l, float4 r) {
		return bool4(l.x > r.x, l.y > r.y, l.z > r.z, l.w > r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator>= (float4 l, float4 r) {
		return bool4(l.x >= r.x, l.y >= r.y, l.z >= r.z, l.w >= r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator== (float4 l, float4 r) {
		return bool4(l.x == r.x, l.y == r.y, l.z == r.z, l.w == r.w);
	}
	
	// componentwise comparison returns a bool vector
	bool4 operator!= (float4 l, float4 r) {
		return bool4(l.x != r.x, l.y != r.y, l.z != r.z, l.w != r.w);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (float4 l, float4 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	float4 select (bool4 c, float4 l, float4 r) {
		return float4(c.x ? l.x : r.x, c.y ? l.y : r.y, c.z ? l.z : r.z, c.w ? l.w : r.w);
	}
	
	//// misc ops
	
	// componentwise absolute
	float4 abs (float4 v) {
		return float4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
	}
	
	// componentwise minimum
	float4 min (float4 l, float4 r) {
		return float4(min(l.x,r.x), min(l.y,r.y), min(l.z,r.z), min(l.w,r.w));
	}
	
	// componentwise maximum
	float4 max (float4 l, float4 r) {
		return float4(max(l.x,r.x), max(l.y,r.y), max(l.z,r.z), max(l.w,r.w));
	}
	
	// componentwise clamp into range [a,b]
	float4 clamp (float4 x, float4 a, float4 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	float4 clamp (float4 x) {
		return min(max(x, float(0)), float(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	float min_component (float4 v, int* min_index) {
		int index = 0;
		float min_val = v.x;	
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
	float max_component (float4 v, int* max_index) {
		int index = 0;
		float max_val = v.x;	
		for (int i=1; i<4; ++i) {
			if (v.arr[i] >= max_val) {
				index = i;
				max_val = v.arr[i];
			}
		}
		if (max_index) *max_index = index;
		return max_val;
	}
	
	
	// componentwise floor
	float4 floor (float4 v) {
		return float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
	}
	
	// componentwise ceil
	float4 ceil (float4 v) {
		return float4(ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w));
	}
	
	// componentwise round
	float4 round (float4 v) {
		return float4(round(v.x), round(v.y), round(v.z), round(v.w));
	}
	
	// componentwise floor to int
	int4 floori (float4 v) {
		return int4(floori(v.x), floori(v.y), floori(v.z), floori(v.w));
	}
	
	// componentwise ceil to int
	int4 ceili (float4 v) {
		return int4(ceili(v.x), ceili(v.y), ceili(v.z), ceili(v.w));
	}
	
	// componentwise round to int
	int4 roundi (float4 v) {
		return int4(roundi(v.x), roundi(v.y), roundi(v.z), roundi(v.w));
	}
	
	// componentwise pow
	float4 pow (float4 v, float4 e) {
		return float4(pow(v.x,e.x), pow(v.y,e.y), pow(v.z,e.z), pow(v.w,e.w));
	}
	
	// componentwise wrap
	float4 wrap (float4 v, float4 range) {
		return float4(wrap(v.x,range.x), wrap(v.y,range.y), wrap(v.z,range.z), wrap(v.w,range.w));
	}
	
	// componentwise wrap
	float4 wrap (float4 v, float4 a, float4 b) {
		return float4(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y), wrap(v.z,a.z,b.z), wrap(v.w,a.w,b.w));
	}
	
	
	//// Angle conversion
	
	
	// converts degrees to radiants
	float4 radians (float4 deg) {
		return deg * DEG_TO_RAD;
	}
	
	// converts radiants to degrees
	float4 degrees (float4 deg) {
		return deg * DEG_TO_RAD;
	}
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	float4 lerp (float4 a, float4 b, float4 t) {
		return t * (b - a) + a;
	}
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	float4 map (float4 x, float4 in_a, float4 in_b) {
		return (x - in_a) / (in_b - in_a);
	}
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	float4 map (float4 x, float4 in_a, float4 in_b, float4 out_a, float4 out_b) {
		return lerp(out_a, out_b, map(x, in_a, in_b));
	}
	
	//// Various interpolation
	
	
	// standard smoothstep interpolation
	float4 smoothstep (float4 x) {
		float4 t = clamp(x);
		return t * t * (3.0f - 2.0f * t);
	}
	
	// 3 point bezier interpolation
	float4 bezier (float4 a, float4 b, float4 c, float t) {
		float4 d = lerp(a, b, t);
		float4 e = lerp(b, c, t);
		float4 f = lerp(d, e, t);
		return f;
	}
	
	// 4 point bezier interpolation
	float4 bezier (float4 a, float4 b, float4 c, float4 d, float t) {
		return bezier(
					  lerp(a, b, t),
					  lerp(b, c, t),
					  lerp(c, d, t),
					  t
			   );
	}
	
	// 5 point bezier interpolation
	float4 bezier (float4 a, float4 b, float4 c, float4 d, float4 e, float t) {
		return bezier(
					  lerp(a, b, t),
					  lerp(b, c, t),
					  lerp(c, d, t),
					  lerp(d, e, t),
					  t
			   );
	}
	
	//// Vector math
	
	
	// magnitude of vector
	float length (float4 v) {
		return sqrt((float)(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w));
	}
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	float length_sqr (float4 v) {
		return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	}
	
	// distance between points, equivalent to length(a - b)
	float distance (float4 a, float4 b) {
		return length(a - b);
	}
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float4 normalize (float4 v) {
		return float4(v) / length(v);
	}
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float4 normalizesafe (float4 v) {
		float len = length(v);
		if (len == float(0)) {
			return float(0);
		}
		return float4(v) / float4(len);
	}
	
	// dot product
	float dot (float4 l, float4 r) {
		return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
	}
}

