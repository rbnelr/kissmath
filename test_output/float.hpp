// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include <cmath>
#include <limits> // for std::numeric_limits<float>::quiet_NaN()
#include <cstdint>

namespace kissmath {
	
	// Use std math functions for these
	using std::abs;
	using std::floor;
	using std::ceil;
	using std::pow;
	using std::round;
	
	// wrap x into range [0,range)
	// negative x wrap back to +range unlike c++ % operator
	// negative range supported
	float wrap (float x, float range);
	
	// wrap x into [a,b) range
	float wrap (float x, float a, float b);
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	float clamp (float x, float a, float b);
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	float clamp (float x);
	
	
	//// Math constants
	
	constexpr float PI          = 3.1415926535897932384626433832795f;
	constexpr float TAU         = 6.283185307179586476925286766559f;
	constexpr float SQRT_2      = 1.4142135623730950488016887242097f;
	constexpr float EULER       = 2.7182818284590452353602874713527f;
	constexpr float DEG_TO_RAD  = 0.01745329251994329576923690768489f; // 180/PI
	constexpr float RAD_TO_DEG  = 57.295779513082320876798154814105f;  // PI/180
	
	#if _MSC_VER && !__INTELRZ_COMPILER && !__clan_
	constexpr float INF         = (float)(1e+300 * 1e+300);
	constexpr float QNAN        = std::numeric_limits<float>::quiet_NaN();
	#elif __GNUC__ || __clan_
	constexpr float INF         = __builtin_inff();
	constexpr float QNAN        = (float)__builtin_nan("0");
	#endif
	
	
	float wrap (float x, float a, float b, int* quotient);
	
	
	// floor and convert to int
	int floori (float x);
	
	// ceil and convert to int
	int ceili (float x);
	
	// round and convert to int
	int roundi (float x);
	
	
	// returns the greater value of a and b
	float min (float l, float r);
	
	// returns the smaller value of a and b
	float max (float l, float r);
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	float select (bool c, float l, float r);
	
	
	//// Angle conversion
	
	// converts degrees to radiants
	float radians (float deg);
	
	// converts radiants to degrees
	float degrees (float deg);
	
	//// Linear interpolation
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	float lerp (float a, float b, float t);
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	float map (float x, float in_a, float in_b);
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	float map (float x, float in_a, float in_b, float out_a, float out_b);
	
	
	//// Various interpolation
	
	// standard smoothstep interpolation
	float smoothstep (float x);
	
	// 3 point bezier interpolation
	float bezier (float a, float b, float c, float t);
	
	// 4 point bezier interpolation
	float bezier (float a, float b, float c, float d, float t);
	
	// 5 point bezier interpolation
	float bezier (float a, float b, float c, float d, float e, float t);
	
	
	// length(scalar) = abs(scalar)
	// for conformity with vectors
	float length (float x);
	
	// length_sqr(scalar) = abs(scalar)^2
	// for conformity with vectors (for vectors this func is preferred over length to avoid the sqrt)
	float length_sqr (float x);
	
	// scalar normalize for conformity with vectors
	// normalize(-6.2f) = -1f, normalize(7) = 1, normalize(0) = <div 0>
	// can be useful in some cases
	float normalize (float x);
	
	// normalize(x) for length(x) != 0 else 0
	float normalizesafe (float x);
	
	
}
