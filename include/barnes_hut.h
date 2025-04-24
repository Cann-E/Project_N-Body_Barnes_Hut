#pragma once
#include "particle.h"
#include <vector>
#include <memory>

#ifdef __CUDACC__
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

const float G = 6.67430e-11f;
const float softening = 0.1f;

struct Bounds {
    Vec3 center;
    Vec3 half_size;
    
    CUDA_CALLABLE Bounds() = default;
    CUDA_CALLABLE Bounds(Vec3 c, Vec3 hs) : center(c), half_size(hs) {}
};

// CPU Tree Node
struct TreeNode {
    Bounds bounds;
    Vec3 center_of_mass;
    float total_mass = 0.0f;
    std::unique_ptr<TreeNode> children[8];
    Particle* particle = nullptr;
};

// GPU Tree Node
struct GPUTreeNode {
    Bounds bounds;
    Vec3 center_of_mass;
    float total_mass;
    int children[8];
    int particle_index;
};

// CPU Functions
void cpu_direct_nbody(std::vector<Particle>& particles);
void cpu_barnes_hut(std::vector<Particle>& particles, float theta);

// GPU Functions
void gpu_barnes_hut(std::vector<Particle>& particles, float theta);