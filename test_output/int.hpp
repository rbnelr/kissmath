// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include <cmath>
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
	int wrap (int x, int range);
	
	// wrap x into [a,b) range
	int wrap (int x, int a, int b);
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	int clamp (int x, int a, int b);
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	int clamp (int x);
	
	// returns the greater value of a and b
	int min (int l, int r);
	
	// returns the smaller value of a and b
	int max (int l, int r);
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	int select (bool c, int l, int r);
	
	
	// length(scalar) = abs(scalar)
	// for conformity with vectors
	int length (int x);
	
	// length_sqr(scalar) = abs(scalar)^2
	// for conformity with vectors (for vectors this func is preferred over length to avoid the sqrt)
	int length_sqr (int x);
	
	// scalar normalize for conformity with vectors
	// normalize(-6.2f) = -1f, normalize(7) = 1, normalize(0) = <div 0>
	// can be useful in some cases
	int normalize (int x);
	
	// normalize(x) for length(x) != 0 else 0
	int normalizesafe (int x);
	
	
}

