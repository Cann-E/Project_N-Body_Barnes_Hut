#pragma once
#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

struct Vec3 {
    float x, y, z;

    CUDA_CALLABLE Vec3() : x(0), y(0), z(0) {}  // Explicit implementation instead of =default
    CUDA_CALLABLE Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    CUDA_CALLABLE Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    CUDA_CALLABLE Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    CUDA_CALLABLE Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    CUDA_CALLABLE Vec3& operator+=(const Vec3& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }

    CUDA_CALLABLE float length_squared() const {
        return x*x + y*y + z*z;
    }
};