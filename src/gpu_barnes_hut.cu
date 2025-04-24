#include "barnes_hut.h"

__device__ int compute_morton_code(Vec3 pos, Bounds universe) {
    float x = (pos.x - universe.center.x + universe.half_size.x) / (2 * universe.half_size.x);
    float y = (pos.y - universe.center.y + universe.half_size.y) / (2 * universe.half_size.y);
    float z = (pos.z - universe.center.z + universe.half_size.z) / (2 * universe.half_size.z);

    int code = 0;
    for (int i = 0; i < 10; ++i) {
        code |= ((int)(x * (1 << 10)) & (1 << i)) << (2 * i);
        code |= ((int)(y * (1 << 10)) & (1 << i)) << (2 * i + 1);
        code |= ((int)(z * (1 << 10)) & (1 << i)) << (2 * i + 2);
    }
    return code;
}

__global__ void barnes_hut_kernel(const Particle* particles, const GPUTreeNode* tree, 
                                int n_nodes, float theta, Vec3* forces, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Particle p = particles[idx];
    Vec3 force = Vec3(0, 0, 0);
    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Root node

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        GPUTreeNode node = tree[node_idx];

        Vec3 r = node.center_of_mass - p.position;
        float dist_sq = r.length_squared() + softening * softening;
        float dist = sqrtf(dist_sq);

        if (node.particle_index >= 0) {
            if (node.particle_index != idx) {
                float inv_dist3 = 1.0f / (dist * dist * dist);
                force += r * (G * p.mass * node.total_mass * inv_dist3);
            }
        } else if (2.0f * node.bounds.half_size.x / dist < theta) {
            float inv_dist3 = 1.0f / (dist * dist * dist);
            force += r * (G * p.mass * node.total_mass * inv_dist3);
        } else {
            for (int i = 7; i >= 0; --i) {
                if (node.children[i] >= 0) {
                    stack[stack_ptr++] = node.children[i];
                }
            }
        }
    }

    forces[idx] = force;
}

void gpu_barnes_hut(std::vector<Particle>& particles, float theta) {
    const int n = particles.size();
    const int max_tree_nodes = n * 2;
    
    // Allocate device memory
    Particle* d_particles;
    GPUTreeNode* d_tree;
    Vec3* d_forces;
    cudaMalloc(&d_particles, n * sizeof(Particle));
    cudaMalloc(&d_tree, max_tree_nodes * sizeof(GPUTreeNode));
    cudaMalloc(&d_forces, n * sizeof(Vec3));

    // Copy particles to device
    cudaMemcpy(d_particles, particles.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);

    // Build tree (simplified)
    GPUTreeNode root;
    Vec3 min_pos = particles[0].position;
    Vec3 max_pos = particles[0].position;
    for (const auto& p : particles) {
        min_pos.x = fminf(min_pos.x, p.position.x);
        min_pos.y = fminf(min_pos.y, p.position.y);
        min_pos.z = fminf(min_pos.z, p.position.z);
        max_pos.x = fmaxf(max_pos.x, p.position.x);
        max_pos.y = fmaxf(max_pos.y, p.position.y);
        max_pos.z = fmaxf(max_pos.z, p.position.z);
    }
    Vec3 center = (min_pos + max_pos) * 0.5f;
    Vec3 half_size = (max_pos - min_pos) * 0.5f;
    root.bounds = Bounds(center, half_size);
    root.particle_index = -1;
    
    // Initialize tree on device
    cudaMemcpy(d_tree, &root, sizeof(GPUTreeNode), cudaMemcpyHostToDevice);

    // Compute forces
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    barnes_hut_kernel<<<grid_size, block_size>>>(d_particles, d_tree, 1, theta, d_forces, n);

    // Copy results back
    cudaMemcpy(particles.data(), d_forces, n * sizeof(Vec3), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_tree);
    cudaFree(d_forces);
}