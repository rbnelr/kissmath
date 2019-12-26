// file was generated by kissmath.py at <TODO: add github link>
#include "transform3d.hpp"

#include <cmath>

namespace kissmath {
	
	
	float3x3 rotate3_X (float ang) {
		float s = std::sin(ang), c = std::cos(ang);
		return float3x3(
						 1,  0,  0,
						 0,  c, -s,
						 0,  s,  c
			   );
	}
	
	float3x3 rotate3_Y (float ang) {
		float s = std::sin(ang), c = std::cos(ang);
		return float3x3(
						 c,  0,  s,
						 0,  1,  0,
						-s,  0,  c
			   );
	}
	
	float3x3 rotate3_Z (float ang) {
		float s = std::sin(ang), c = std::cos(ang);
		return float3x3(
						 c, -s,  0,
						 s,  c,  0,
						 0,  0,  1
			   );
	}
	
	float3x3 scale (float3 v) {
		return float3x3(
						v.x,   0,   0,
						  0, v.y,   0,
						  0,   0, v.z
			   );
	}
	
	float3x4 translate (float3 v) {
		return float3x4(
						1, 0, 0, v.x,
						0, 1, 0, v.y,
						0, 0, 1, v.z
			   );
	}
	
	
	double3x3 rotate3_X (double ang) {
		double s = std::sin(ang), c = std::cos(ang);
		return double3x3(
						  1,  0,  0,
						  0,  c, -s,
						  0,  s,  c
			   );
	}
	
	double3x3 rotate3_Y (double ang) {
		double s = std::sin(ang), c = std::cos(ang);
		return double3x3(
						  c,  0,  s,
						  0,  1,  0,
						 -s,  0,  c
			   );
	}
	
	double3x3 rotate3_Z (double ang) {
		double s = std::sin(ang), c = std::cos(ang);
		return double3x3(
						  c, -s,  0,
						  s,  c,  0,
						  0,  0,  1
			   );
	}
	
	double3x3 scale (double3 v) {
		return double3x3(
						 v.x,   0,   0,
						   0, v.y,   0,
						   0,   0, v.z
			   );
	}
	
	double3x4 translate (double3 v) {
		return double3x4(
						 1, 0, 0, v.x,
						 0, 1, 0, v.y,
						 0, 0, 1, v.z
			   );
	}
	
}

