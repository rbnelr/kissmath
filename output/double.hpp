// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include <cmath>
#include <limits> // for std::numeric_limits<float>::quiet_NaN()
#include <cstdint>

namespace kissmath {
	typedef int64_t int64; // typedef this because the _t suffix is kinda unwieldy when using these types often
	
	// Use std math functions for these
	using std::abs;
	using std::floor;
	using std::ceil;
	using std::pow;
	using std::round;
	
	// wrap x into range [0,range)
	// negative x wrap back to +range unlike c++ % operator
	// negative range supported
	double wrap (double x, double range);
	
	// wrap x into [a,b) range
	double wrap (double x, double a, double b);
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	double clamp (double x, double a, double b);
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	double clamp (double x);
	
	
	//// Math constants
	
	constexpr double PId          = 3.1415926535897932384626433832795;
	constexpr double TAUd         = 6.283185307179586476925286766559;
	constexpr double SQRT_2d      = 1.4142135623730950488016887242097;
	constexpr double EULERd       = 2.7182818284590452353602874713527;
	constexpr double DEG_TO_RADd  = 0.01745329251994329576923690768489; // 180/PI
	constexpr double RAD_TO_DEGd  = 57.295779513082320876798154814105;  // PI/180
	
	#if _MSC_VER && !__INTELRZ_COMPILER && !__clan_
	constexpr double INFd         = 1e+300 * 1e+300;
	constexpr double QNANd        = std::numeric_limits<double>::quiet_NaN();
	#elif __GNUC__ || __clan_
	constexpr double INFd         = __builtin_inf();
	constexpr double QNANd        = __builtin_nan("0");
	#endif
	
	
	double wrap (double x, double a, double b, int64* quotient);
	
	
	// floor and convert to int
	int64 floori (double x);
	
	// ceil and convert to int
	int64 ceili (double x);
	
	// round and convert to int
	int64 roundi (double x);
	
	
	// returns the greater value of a and b
	double min (double l, double r);
	
	// returns the smaller value of a and b
	double max (double l, double r);
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	double select (bool c, double l, double r);
	
	
	//// Angle conversion
	
	// converts degrees to radiants
	double to_radians (double deg);
	
	// converts radiants to degrees
	double to_degrees (double rad);
	
	// converts degrees to radiants
	// shortform to make degree literals more readable
	double deg (double deg);
	
	//// Linear interpolation
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	double lerp (double a, double b, double t);
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	double map (double x, double in_a, double in_b);
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	double map (double x, double in_a, double in_b, double out_a, double out_b);
	
	
	//// Various interpolation
	
	// standard smoothstep interpolation
	double smoothstep (double x);
	
	// 3 point bezier interpolation
	double bezier (double a, double b, double c, double t);
	
	// 4 point bezier interpolation
	double bezier (double a, double b, double c, double d, double t);
	
	// 5 point bezier interpolation
	double bezier (double a, double b, double c, double d, double e, double t);
	
	
	// length(scalar) = abs(scalar)
	// for conformity with vectors
	double length (double x);
	
	// length_sqr(scalar) = abs(scalar)^2
	// for conformity with vectors (for vectors this func is preferred over length to avoid the sqrt)
	double length_sqr (double x);
	
	// scalar normalize for conformity with vectors
	// normalize(-6.2f) = -1f, normalize(7) = 1, normalize(0) = <div 0>
	// can be useful in some cases
	double normalize (double x);
	
	// normalize(x) for length(x) != 0 else 0
	double normalizesafe (double x);
	
	
}
