#include "barnes_hut.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_particles> [theta=0.5] [output_file]\n";
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const float theta = (argc > 2) ? std::atof(argv[2]) : 0.5f;
    const std::string output_file = (argc > 3) ? argv[3] : "web/particles.bin";

    std::cout << "Running with " << N << " particles (θ=" << theta << ")\n";

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

    // Timing function
    auto time_simulation = [](auto&& func, const std::string& name) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << name << ": " << elapsed.count() << "s\n";
    };

    // Run simulations
    if (N <= 20000) {
        time_simulation([&](){ cpu_direct_nbody(particles); }, "CPU Direct");
    }

    time_simulation([&](){ cpu_barnes_hut(particles, theta); }, "CPU Barnes-Hut");
    time_simulation([&](){ gpu_barnes_hut(particles, theta); }, "GPU Barnes-Hut");

    save_particles_binary(particles, output_file);
    std::cout << "Particle data saved to " << output_file << "\n";

    return 0;
}
