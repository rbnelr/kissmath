// file was generated by kissmath.py at <TODO: add github link>
#include "float2.hpp"

#include "int64v2.hpp"
#include "uint2.hpp"
#include "bool2.hpp"
#include "float3.hpp"
#include "uint16v2.hpp"
#include "int16v2.hpp"
#include "double2.hpp"
#include "int2.hpp"
#include "float4.hpp"
#include "uint64v2.hpp"
#include "int8v2.hpp"
#include "uint8v2.hpp"

namespace kissmath {
	//// forward declarations
	// typedef these because the _t suffix is kinda unwieldy when using these types often
	
	typedef int64_t int64;
	typedef unsigned int uint;
	typedef uint16_t uint16;
	typedef int16_t int16;
	typedef uint64_t uint64;
	typedef int8_t int8;
	typedef uint8_t uint8;
	
	// Component indexing operator
	float& float2::operator[] (int i) {
		return arr[i];
	}
	
	// Component indexing operator
	float const& float2::operator[] (int i) const {
		return arr[i];
	}
	
	
	// uninitialized constructor
	float2::float2 () {
		
	}
	
	// sets all components to one value
	// implicit constructor -> float3(x,y,z) * 5 will be turned into float3(x,y,z) * float3(5) by to compiler to be able to execute operator*(float3, float3), which is desirable
	// and short initialization like float3 a = 0; works
	float2::float2 (float all): x{all}, y{all} {
		
	}
	
	// supply all components
	float2::float2 (float x, float y): x{x}, y{y} {
		
	}
	
	// truncate vector
	float2::float2 (float3 v): x{v.x}, y{v.y} {
		
	}
	
	// truncate vector
	float2::float2 (float4 v): x{v.x}, y{v.y} {
		
	}
	
	//// Truncating cast operators
	
	
	//// Type cast operators
	
	
	// type cast operator
	float2::operator bool2 () const {
		return bool2((bool)x, (bool)y);
	}
	
	// type cast operator
	float2::operator double2 () const {
		return double2((double)x, (double)y);
	}
	
	// type cast operator
	float2::operator int8v2 () const {
		return int8v2((int8)x, (int8)y);
	}
	
	// type cast operator
	float2::operator int16v2 () const {
		return int16v2((int16)x, (int16)y);
	}
	
	// type cast operator
	float2::operator int2 () const {
		return int2((int)x, (int)y);
	}
	
	// type cast operator
	float2::operator int64v2 () const {
		return int64v2((int64)x, (int64)y);
	}
	
	// type cast operator
	float2::operator uint8v2 () const {
		return uint8v2((uint8)x, (uint8)y);
	}
	
	// type cast operator
	float2::operator uint16v2 () const {
		return uint16v2((uint16)x, (uint16)y);
	}
	
	// type cast operator
	float2::operator uint2 () const {
		return uint2((uint)x, (uint)y);
	}
	
	// type cast operator
	float2::operator uint64v2 () const {
		return uint64v2((uint64)x, (uint64)y);
	}
	
	
	// componentwise arithmetic operator
	float2 float2::operator+= (float2 r) {
		x += r.x;
		y += r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	float2 float2::operator-= (float2 r) {
		x -= r.x;
		y -= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	float2 float2::operator*= (float2 r) {
		x *= r.x;
		y *= r.y;
		return *this;
	}
	
	// componentwise arithmetic operator
	float2 float2::operator/= (float2 r) {
		x /= r.x;
		y /= r.y;
		return *this;
	}
	
	//// arthmethic ops
	
	
	float2 operator+ (float2 v) {
		return float2(+v.x, +v.y);
	}
	
	float2 operator- (float2 v) {
		return float2(-v.x, -v.y);
	}
	
	float2 operator+ (float2 l, float2 r) {
		return float2(l.x + r.x, l.y + r.y);
	}
	
	float2 operator- (float2 l, float2 r) {
		return float2(l.x - r.x, l.y - r.y);
	}
	
	float2 operator* (float2 l, float2 r) {
		return float2(l.x * r.x, l.y * r.y);
	}
	
	float2 operator/ (float2 l, float2 r) {
		return float2(l.x / r.x, l.y / r.y);
	}
	
	//// bitwise ops
	
	
	//// comparison ops
	
	
	// componentwise comparison returns a bool vector
	bool2 operator< (float2 l, float2 r) {
		return bool2(l.x < r.x, l.y < r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator<= (float2 l, float2 r) {
		return bool2(l.x <= r.x, l.y <= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator> (float2 l, float2 r) {
		return bool2(l.x > r.x, l.y > r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator>= (float2 l, float2 r) {
		return bool2(l.x >= r.x, l.y >= r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator== (float2 l, float2 r) {
		return bool2(l.x == r.x, l.y == r.y);
	}
	
	// componentwise comparison returns a bool vector
	bool2 operator!= (float2 l, float2 r) {
		return bool2(l.x != r.x, l.y != r.y);
	}
	
	// vectors are equal, equivalent to all(l == r)
	bool equal (float2 l, float2 r) {
		return all(l == r);
	}
	
	// componentwise ternary (c ? l : r)
	float2 select (bool2 c, float2 l, float2 r) {
		return float2(c.x ? l.x : r.x, c.y ? l.y : r.y);
	}
	
	//// misc ops
	
	// componentwise absolute
	float2 abs (float2 v) {
		return float2(abs(v.x), abs(v.y));
	}
	
	// componentwise minimum
	float2 min (float2 l, float2 r) {
		return float2(min(l.x,r.x), min(l.y,r.y));
	}
	
	// componentwise maximum
	float2 max (float2 l, float2 r) {
		return float2(max(l.x,r.x), max(l.y,r.y));
	}
	
	// componentwise clamp into range [a,b]
	float2 clamp (float2 x, float2 a, float2 b) {
		return min(max(x,a), b);
	}
	
	// componentwise clamp into range [0,1] also known as saturate in hlsl
	float2 clamp (float2 x) {
		return min(max(x, float(0)), float(1));
	}
	
	// get minimum component of vector, optionally get component index via min_index
	float min_component (float2 v, int* min_index) {
		int index = 0;
		float min_val = v.x;	
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
	float max_component (float2 v, int* max_index) {
		int index = 0;
		float max_val = v.x;	
		for (int i=1; i<2; ++i) {
			if (v.arr[i] >= max_val) {
				index = i;
				max_val = v.arr[i];
			}
		}
		if (max_index) *max_index = index;
		return max_val;
	}
	
	
	// componentwise floor
	float2 floor (float2 v) {
		return float2(floor(v.x), floor(v.y));
	}
	
	// componentwise ceil
	float2 ceil (float2 v) {
		return float2(ceil(v.x), ceil(v.y));
	}
	
	// componentwise round
	float2 round (float2 v) {
		return float2(round(v.x), round(v.y));
	}
	
	// componentwise floor to int
	int2 floori (float2 v) {
		return int2(floori(v.x), floori(v.y));
	}
	
	// componentwise ceil to int
	int2 ceili (float2 v) {
		return int2(ceili(v.x), ceili(v.y));
	}
	
	// componentwise round to int
	int2 roundi (float2 v) {
		return int2(roundi(v.x), roundi(v.y));
	}
	
	// componentwise pow
	float2 pow (float2 v, float2 e) {
		return float2(pow(v.x,e.x), pow(v.y,e.y));
	}
	
	// componentwise wrap
	float2 wrap (float2 v, float2 range) {
		return float2(wrap(v.x,range.x), wrap(v.y,range.y));
	}
	
	// componentwise wrap
	float2 wrap (float2 v, float2 a, float2 b) {
		return float2(wrap(v.x,a.x,b.x), wrap(v.y,a.y,b.y));
	}
	
	
	//// Angle conversion
	
	
	// converts degrees to radiants
	float2 radians (float2 deg) {
		return deg * DEG_TO_RAD;
	}
	
	// converts radiants to degrees
	float2 degrees (float2 deg) {
		return deg * DEG_TO_RAD;
	}
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	float2 lerp (float2 a, float2 b, float2 t) {
		return t * (b - a) + a;
	}
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	float2 map (float2 x, float2 in_a, float2 in_b) {
		return (x - in_a) / (in_b - in_a);
	}
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	float2 map (float2 x, float2 in_a, float2 in_b, float2 out_a, float2 out_b) {
		return lerp(out_a, out_b, map(x, in_a, in_b));
	}
	
	//// Various interpolation
	
	
	// standard smoothstep interpolation
	float2 smoothstep (float2 x) {
		float2 t = clamp(x);
		return t * t * (3.0f - 2.0f * t);
	}
	
	// 3 point bezier interpolation
	float2 bezier (float2 a, float2 b, float2 c, float t) {
		float2 d = lerp(a, b, t);
		float2 e = lerp(b, c, t);
		float2 f = lerp(d, e, t);
		return f;
	}
	
	// 4 point bezier interpolation
	float2 bezier (float2 a, float2 b, float2 c, float2 d, float t) {
		return bezier(
					  lerp(a, b, t),
					  lerp(b, c, t),
					  lerp(c, d, t),
					  t
			   );
	}
	
	// 5 point bezier interpolation
	float2 bezier (float2 a, float2 b, float2 c, float2 d, float2 e, float t) {
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
	float length (float2 v) {
		return sqrt((float)(v.x * v.x + v.y * v.y));
	}
	
	// squared magnitude of vector, cheaper than length() because it avoids the sqrt(), some algorithms only need the squared magnitude
	float length_sqr (float2 v) {
		return v.x * v.x + v.y * v.y;
	}
	
	// distance between points, equivalent to length(a - b)
	float distance (float2 a, float2 b) {
		return length(a - b);
	}
	
	// normalize vector so that it has length() = 1, undefined for zero vector
	float2 normalize (float2 v) {
		return float2(v) / length(v);
	}
	
	// normalize vector so that it has length() = 1, returns zero vector if vector was zero vector
	float2 normalizesafe (float2 v) {
		float len = length(v);
		if (len == float(0)) {
			return float(0);
		}
		return float2(v) / float2(len);
	}
	
	// dot product
	float dot (float2 l, float2 r) {
		return l.x * r.x + l.y * r.y;
	}
	
	// 2d cross product hack for convenient 2d stuff
	// same as cross({T.name[:-2]}3(l, 0), {T.name[:-2]}3(r, 0)).z,
	// ie. the cross product of the 2d vectors on the z=0 plane in 3d space and then return the z coord of that (signed mag of cross product)
	float cross (float2 l, float2 r) {
		return l.x * r.y - l.y * r.x;
	}
	
	// rotate 2d vector counterclockwise 90 deg, ie. float2(-y, x) which is fast
	float2 rotate90 (float2 v) {
		return float2(-v.y, v.x);
	}
}

