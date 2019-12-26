// file was generated by kissmath.py at <TODO: add github link>
#pragma once

#include <cmath>
#include <cstdint>

namespace kissmath {
	typedef uint64_t uint64; // typedef this because the _t suffix is kinda unwieldy when using these types often
	
	// Use std math functions for these
	using std::abs;
	using std::floor;
	using std::ceil;
	using std::pow;
	using std::round;
	
	// wrap x into range [0,range)
	// negative x wrap back to +range unlike c++ % operator
	// negative range supported
	uint64 wrap (uint64 x, uint64 range);
	
	// wrap x into [a,b) range
	uint64 wrap (uint64 x, uint64 a, uint64 b);
	
	// clamp x into range [a, b]
	// equivalent to min(max(x,a), b)
	uint64 clamp (uint64 x, uint64 a, uint64 b);
	
	// clamp x into range [0, 1]
	// also known as saturate in hlsl
	uint64 clamp (uint64 x);
	
	// returns the greater value of a and b
	uint64 min (uint64 l, uint64 r);
	
	// returns the smaller value of a and b
	uint64 max (uint64 l, uint64 r);
	
	// equivalent to ternary c ? l : r
	// for conformity with vectors
	uint64 select (bool c, uint64 l, uint64 r);
	
	
}

