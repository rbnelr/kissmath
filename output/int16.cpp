// file was generated by kissmath.py at <TODO: add github link>
#include "int16.hpp"

namespace kissmath {
	
	// wrap x into range [0,range)
	// negative x wrap back to +range unlike c++ % operator
	// negative range supported
	int16 wrap (int16 x, int16 range) {
		int16 modded = x % range;
		if (range > 0) {
			if (modded < 0) modded += range;
		} else {
			if (modded > 0) modded += range;
		}
		return modded;
	}
	
	// wrap x into [a,b) range
	int16 wrap (int16 x, int16 a, int16 b) {
		x -= a;
		int16 range = b -a;
		
		int16 modulo = wrap(x, range);
		
		return modulo + a;
	}
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	int16 clamp (int16 x, int16 a, int16 b) {
		return min(max(x, a), b);
	}
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	int16 clamp (int16 x) {
		return min(max(x, int16(0)), int16(1));
	}
	
	// returns the greater value of a and b
	int16 min (int16 l, int16 r) {
		return l <= r ? l : r;
	}
	
	// returns the smaller value of a and b
	int16 max (int16 l, int16 r) {
		return l >= r ? l : r;
	}
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	int16 select (bool c, int16 l, int16 r) {
		return c ? l : r;
	}
	
	
	// length(scalar) = abs(scalar)
	// for conformity with vectors
	int16 length (int16 x) {
		return std::abs(x);
	}
	
	// length_sqr(scalar) = abs(scalar)^2
	// for conformity with vectors (for vectors this func is preferred over length to avoid the sqrt)
	int16 length_sqr (int16 x) {
		x = std::abs(x);
		return x*x;
	}
	
	// scalar normalize for conformity with vectors
	// normalize(-6.2f) = -1f, normalize(7) = 1, normalize(0) = <div 0>
	// can be useful in some cases
	int16 normalize (int16 x) {
		return x / length(x);
	}
	
	// normalize(x) for length(x) != 0 else 0
	int16 normalizesafe (int16 x) {
		int16 len = length(x);
		if (len == int16(0)) {
			return int16(0);
		}
		return x /= len;
	}
	
}
