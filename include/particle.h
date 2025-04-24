#pragma once
#include "vec3.h"

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 force;
    float mass;

    CUDA_CALLABLE Particle() : position(), velocity(), force(), mass(1.0f) {}
};