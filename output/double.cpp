// file was generated by kissmath.py at <TODO: add github link>
#include "double.hpp"

namespace kissmath {
	
	// wrap x into range [0,range)
	// negative x wrap back to +range unlike c++ % operator
	// negative range supported
	double wrap (double x, double range) {
		double modded = std::fmod(x, range);
		if (range > 0) {
			if (modded < 0) modded += range;
		} else {
			if (modded > 0) modded += range;
		}
		return modded;
	}
	
	// wrap x into [a,b) range
	double wrap (double x, double a, double b) {
		x -= a;
		double range = b -a;
		
		double modulo = wrap(x, range);
		
		return modulo + a;
	}
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	double clamp (double x, double a, double b) {
		return min(max(x, a), b);
	}
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	double clamp (double x) {
		return min(max(x, double(0)), double(1));
	}
	
	double wrap (double x, double a, double b, int64* quotient) {
		x -= a;
		double range = b -a;
		
		double modulo = wrap(x, range);
		*quotient = floori(x / range);
		
		return modulo + a;
	}
	
	
	// floor and convert to int
	int64 floori (double x) {
		return (int64)floor(x);
	}
	
	// ceil and convert to int
	int64 ceili (double x) {
		return (int64)ceil(x);
	}
	
	// round and convert to int
	int64 roundi (double x) {
		return std::llround(x);
	}
	
	
	// returns the greater value of a and b
	double min (double l, double r) {
		return l <= r ? l : r;
	}
	
	// returns the smaller value of a and b
	double max (double l, double r) {
		return l >= r ? l : r;
	}
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	double select (bool c, double l, double r) {
		return c ? l : r;
	}
	
	//// Angle conversion
	
	
	// converts degrees to radiants
	double to_radians (double deg) {
		return deg * DEG_TO_RADd;
	}
	
	// converts radiants to degrees
	double to_degrees (double rad) {
		return rad * RAD_TO_DEGd;
	}
	
	// converts degrees to radiants
	// shortform to make degree literals more readable
	double deg (double deg) {
		return deg * DEG_TO_RADd;
	}
	
	// linear interpolation
	// like getting the output of a linear function
	// ex. t=0 -> a ; t=1 -> b ; t=0.5 -> (a+b)/2
	double lerp (double a, double b, double t) {
		return t * (b - a) + a;
	}
	
	// linear mapping
	// sometimes called inverse linear interpolation
	// like getting the x for a y on a linear function
	// ex. map(70, 0,100) -> 0.7 ; map(0.5, -1,+1) -> 0.75
	double map (double x, double in_a, double in_b) {
		return (x - in_a) / (in_b - in_a);
	}
	
	// linear remapping
	// equivalent of lerp(out_a, out_b, map(x, in_a, in_b))
	double map (double x, double in_a, double in_b, double out_a, double out_b) {
		return lerp(out_a, out_b, map(x, in_a, in_b));
	}
	
	//// Various interpolation
	
	
	// standard smoothstep interpolation
	double smoothstep (double x) {
		double t = clamp(x);
		return t * t * (3.0 - 2.0 * t);
	}
	
	// 3 point bezier interpolation
	double bezier (double a, double b, double c, double t) {
		double d = lerp(a, b, t);
		double e = lerp(b, c, t);
		double f = lerp(d, e, t);
		return f;
	}
	
	// 4 point bezier interpolation
	double bezier (double a, double b, double c, double d, double t) {
		return bezier(
					  lerp(a, b, t),
					  lerp(b, c, t),
					  lerp(c, d, t),
					  t
			   );
	}
	
	// 5 point bezier interpolation
	double bezier (double a, double b, double c, double d, double e, double t) {
		return bezier(
					  lerp(a, b, t),
					  lerp(b, c, t),
					  lerp(c, d, t),
					  lerp(d, e, t),
					  t
			   );
	}
	
	
	// length(scalar) = abs(scalar)
	// for conformity with vectors
	double length (double x) {
		return std::fabs(x);
	}
	
	// length_sqr(scalar) = abs(scalar)^2
	// for conformity with vectors (for vectors this func is preferred over length to avoid the sqrt)
	double length_sqr (double x) {
		x = std::fabs(x);
		return x*x;
	}
	
	// scalar normalize for conformity with vectors
	// normalize(-6.2f) = -1f, normalize(7) = 1, normalize(0) = <div 0>
	// can be useful in some cases
	double normalize (double x) {
		return x / length(x);
	}
	
	// normalize(x) for length(x) != 0 else 0
	double normalizesafe (double x) {
		double len = length(x);
		if (len == double(0)) {
			return double(0);
		}
		return x /= len;
	}
	
}
