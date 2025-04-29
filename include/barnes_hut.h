#pragma once
#include "particle.h"
#include <vector>
#include <memory>
#include <fstream>
#include <functional>

#ifdef __CUDACC__
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#endif

const float G = 6.67430e-11f;
const float softening = 0.1f;
const float VISUALIZATION_SCALE = 0.001f;

struct Bounds {
    Vec3 center;
    Vec3 half_size;
    
    CUDA_CALLABLE Bounds() = default;
    CUDA_CALLABLE Bounds(Vec3 c, Vec3 hs) : center(c), half_size(hs) {}
};

struct TreeNode {
    Bounds bounds;
    Vec3 center_of_mass;
    float total_mass = 0.0f;
    std::unique_ptr<TreeNode> children[8];
    Particle* particle = nullptr;
    int depth = 0;

    void collect_boundaries(std::vector<std::pair<Bounds, int>>& boundaries, int max_depth = -1) const {
        boundaries.emplace_back(bounds, depth);
        if (max_depth == -1 || depth < max_depth) {
            for (int i = 0; i < 8; ++i) {
                if (children[i]) children[i]->collect_boundaries(boundaries, max_depth);
            }
        }
    }
};

struct GPUTreeNode {
    Bounds bounds;
    Vec3 center_of_mass;
    float total_mass;
    int children[8];
    int particle_index;
};

// Simulation Functions
void cpu_direct_nbody(std::vector<Particle>& particles);
void cpu_barnes_hut(std::vector<Particle>& particles, float theta);
void gpu_barnes_hut(std::vector<Particle>& particles, float theta);

// Visualization Functions (declarations only)
void save_particles_binary(const std::vector<Particle>& particles, const std::string& filename);
void save_tree_structure(const TreeNode& root, const std::string& filename);
