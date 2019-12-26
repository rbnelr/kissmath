// file was generated by kissmath.py at <TODO: add github link>
#include "transform2d.hpp"

#include <cmath>

namespace kissmath {
	
	
	float2x2 rotate2 (float ang) {
		float s = std::sin(ang), c = std::cos(ang);
		return float2x2(
						 c, -s,
						 s,  c
			   );
	}
	
	float2x2 scale (float2 v) {
		return float2x2(
						v.x,   0,
						  0, v.y
			   );
	}
	
	float2x3 translate (float2 v) {
		return float2x3(
						1, 0, v.x,
						0, 1, v.y
			   );
	}
	
	
	double2x2 rotate2 (double ang) {
		double s = std::sin(ang), c = std::cos(ang);
		return double2x2(
						  c, -s,
						  s,  c
			   );
	}
	
	double2x2 scale (double2 v) {
		return double2x2(
						 v.x,   0,
						   0, v.y
			   );
	}
	
	double2x3 translate (double2 v) {
		return double2x3(
						 1, 0, v.x,
						 0, 1, v.y
			   );
	}
	
}

