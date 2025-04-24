#include "barnes_hut.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 1000000;  // Default: 1K particles
    const float theta = 0.5f;
    
    std::cout << "Running N-body simulation with N = " << N << " particles\n";
    std::cout << std::fixed << std::setprecision(3);

    // Initialize particles
    std::vector<Particle> particles(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-1000.0f, 1000.0f);
    std::uniform_real_distribution<float> mass_dist(1.0f, 100.0f);

    for (auto& p : particles) {
        p.position = Vec3(pos_dist(gen), pos_dist(gen), pos_dist(gen));
        p.mass = mass_dist(gen);
    }

    // Only run CPU Direct for N ≤ 10,000 (avoid O(N²) slowdown)
    if (N <= 100000) {
        auto start = std::chrono::high_resolution_clock::now();
        cpu_direct_nbody(particles);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "CPU Direct O(n^2): " << duration.count() << " sec\n";
    } else {
        std::cout << "CPU Direct: Skipped (N too large for O(N²) method)\n";
    }

    // Always run Barnes-Hut variants
    auto start = std::chrono::high_resolution_clock::now();
    cpu_barnes_hut(particles, theta);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end - start;
    std::cout << "CPU Barnes-Hut O(nlogn): " << cpu_time.count() << " sec\n";

    start = std::chrono::high_resolution_clock::now();
    gpu_barnes_hut(particles, theta);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;
    std::cout << "GPU Barnes-Hut O(nlogn): " << gpu_time.count() << " sec\n";

    // Speedup calculation (if GPU enabled)
    if (gpu_time.count() > 0 && N <= 10000) {
        double speedup = cpu_time.count() / gpu_time.count();
        std::cout << "GPU Speedup vs CPU Barnes-Hut: " << speedup << "x\n";
    }

    return 0;
}