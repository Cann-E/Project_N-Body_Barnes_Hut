#include "barnes_hut.h"

void cpu_direct_nbody(std::vector<Particle>& particles) {
    const size_t n = particles.size();
    
    for (size_t i = 0; i < n; ++i) {
        particles[i].force = Vec3(0, 0, 0);
        
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            
            Vec3 r = particles[j].position - particles[i].position;
            float dist_sq = r.length_squared() + softening * softening;
            float inv_dist = 1.0f / sqrtf(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            
            particles[i].force += r * (G * particles[i].mass * particles[j].mass * inv_dist3);
        }
    }
}