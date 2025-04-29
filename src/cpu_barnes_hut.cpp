#include "barnes_hut.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>

int get_octant(const TreeNode& node, const Particle& p) {
    int octant = 0;
    if (p.position.x >= node.bounds.center.x) octant |= 1;
    if (p.position.y >= node.bounds.center.y) octant |= 2;
    if (p.position.z >= node.bounds.center.z) octant |= 4;
    return octant;
}

void insert_particle(TreeNode& node, Particle* p, int depth = 0) {
    if (node.particle == nullptr && node.children[0] == nullptr) {
        node.particle = p;
        node.depth = depth;
        return;
    }

    if (node.children[0] == nullptr) {
        for (int i = 0; i < 8; ++i) {
            Vec3 new_center = node.bounds.center;
            new_center.x += node.bounds.half_size.x * ((i & 1) ? 0.5f : -0.5f);
            new_center.y += node.bounds.half_size.y * ((i & 2) ? 0.5f : -0.5f);
            new_center.z += node.bounds.half_size.z * ((i & 4) ? 0.5f : -0.5f);
            
            node.children[i] = std::make_unique<TreeNode>();
            node.children[i]->bounds = Bounds(new_center, node.bounds.half_size * 0.5f);
            node.children[i]->depth = depth + 1;
        }
        insert_particle(*node.children[get_octant(node, *node.particle)], node.particle, depth + 1);
        node.particle = nullptr;
    }
    insert_particle(*node.children[get_octant(node, *p)], p, depth + 1);
}

void compute_center_of_mass(TreeNode& node) {
    if (node.particle != nullptr) {
        node.center_of_mass = node.particle->position;
        node.total_mass = node.particle->mass;
        return;
    }
    
    Vec3 com(0, 0, 0);
    float total_mass = 0.0f;
    
    for (const auto& child : node.children) {
        if (child) {
            compute_center_of_mass(*child);
            com += child->center_of_mass * child->total_mass;
            total_mass += child->total_mass;
        }
    }
    
    if (total_mass > 0) {
        node.center_of_mass = com * (1.0f / total_mass);
    }
    node.total_mass = total_mass;
}

void compute_force_barnes_hut(const Particle& p, const TreeNode& node, Vec3& force, float theta) {
    if (node.total_mass == 0.0f) return;
    
    Vec3 r = node.center_of_mass - p.position;
    float dist_sq = r.length_squared() + softening * softening;
    float dist = sqrtf(dist_sq);
    
    if ((2.0f * node.bounds.half_size.x / dist) < theta || node.particle != nullptr) {
        if (&p != node.particle) {
            float inv_dist3 = 1.0f / (dist * dist * dist);
            force += r * (G * p.mass * node.total_mass * inv_dist3);
        }
    } else {
        for (const auto& child : node.children) {
            if (child) compute_force_barnes_hut(p, *child, force, theta);
        }
    }
}

// Visualization function implementations
void save_particles_binary(const std::vector<Particle>& particles, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening " << filename << " for writing\n";
        return;
    }

    // Write header with particle count
    uint32_t count = particles.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    // Write scaled particle positions
    for (const auto& p : particles) {
        Vec3 scaled_pos = p.position * VISUALIZATION_SCALE;
        out.write(reinterpret_cast<const char*>(&scaled_pos), sizeof(Vec3));
    }
    std::cout << "Saved " << count << " particles to " << filename << "\n";
}

void save_tree_structure(const TreeNode& root, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening " << filename << " for writing\n";
        return;
    }

    std::vector<std::pair<Bounds, int>> boundaries;
    root.collect_boundaries(boundaries);
    
    out << "[";
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto& [bounds, depth] = boundaries[i];
        out << "{"
            << "\"x\":" << bounds.center.x * VISUALIZATION_SCALE << ","
            << "\"y\":" << bounds.center.y * VISUALIZATION_SCALE << ","
            << "\"width\":" << (bounds.half_size.x * 2 * VISUALIZATION_SCALE) << ","
            << "\"height\":" << (bounds.half_size.y * 2 * VISUALIZATION_SCALE) << ","
            << "\"depth\":" << depth
            << "}";
        if (i < boundaries.size() - 1) out << ",";
    }
    out << "]";
    std::cout << "Saved tree structure to " << filename << "\n";
}

void cpu_barnes_hut(std::vector<Particle>& particles, float theta) {
    TreeNode root;
    
    // Calculate bounds to contain all particles
    Vec3 min_pos = particles[0].position;
    Vec3 max_pos = particles[0].position;
    for (const auto& p : particles) {
        min_pos.x = std::min(min_pos.x, p.position.x);
        min_pos.y = std::min(min_pos.y, p.position.y);
        min_pos.z = std::min(min_pos.z, p.position.z);
        max_pos.x = std::max(max_pos.x, p.position.x);
        max_pos.y = std::max(max_pos.y, p.position.y);
        max_pos.z = std::max(max_pos.z, p.position.z);
    }
    Vec3 center = (min_pos + max_pos) * 0.5f;
    Vec3 half_size = (max_pos - min_pos) * 0.5f;
    root.bounds = Bounds(center, half_size);

    // Build tree
    for (auto& p : particles) {
        insert_particle(root, &p);
    }
    
    // Compute center of mass
    compute_center_of_mass(root);
    
    // Compute forces
    for (auto& p : particles) {
        Vec3 force(0, 0, 0);
        compute_force_barnes_hut(p, root, force, theta);
        p.force = force;
    }

    // Save visualization data
    save_particles_binary(particles, "web/particles.bin");
    save_tree_structure(root, "web/tree_structure.json");
}
