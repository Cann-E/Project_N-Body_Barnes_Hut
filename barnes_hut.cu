/*
 * Barnes-Hut N-body Simulation with CUDA acceleration and OpenGL visualization
 * 
 * This program simulates gravitational interactions between N bodies using:
 * - Barnes-Hut algorithm (O(N log N) complexity)
 * - CUDA for parallel computation on GPU
 * - OpenGL for real-time 3D visualization
 * 
 * Key features:
 * - Octree data structure for efficient force calculations
 * - Parallel tree construction and traversal
 * - Interactive visualization with velocity-based coloring
 * - Benchmark mode for performance measurement
 */

 #include <cuda_runtime.h>
 #include <GL/glut.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h> // For strcmp
 
 /******************************
  * SIMULATION PARAMETERS
  ******************************/
 #define N 10000            // Number of bodies in simulation
 #define THETA 0.5f       // Barnes-Hut opening angle parameter
 #define DT 0.01f         // Time step for simulation
 #define G 6.67430e-11f   // Gravitational constant
 #define SOFTENING 1e-5f  // Softening parameter to prevent numerical instability
 #define MAX_DEPTH 8      // Maximum depth of octree
 #define MAX_NODES N*8    // Maximum number of octree nodes (N*8 is usually sufficient)
 
 /******************************
  * DATA STRUCTURES
  ******************************/
 
 // Structure representing a single body/particle
 struct Body {
     float x, y, z;      // 3D position coordinates
     float vx, vy, vz;   // Velocity components
     float mass;         // Mass of the body
 };
 
 // Structure representing a node in the octree
 struct OctreeNode {
     float mass;         // Total mass of all bodies in this node
     float cx, cy, cz;   // Center of mass coordinates
     float min_x, min_y, min_z; // Bounding box minimum coordinates
     float max_x, max_y, max_z; // Bounding box maximum coordinates
     int is_leaf;        // Flag indicating if this is a leaf node
     int body_idx;       // Index of body stored in this node (for leaf nodes)
     int children[8];    // Indices of child nodes (each octant)
 };
 
 /******************************
  * GLOBAL VARIABLES
  ******************************/
 Body *d_bodies = NULL;      // Device (GPU) memory for bodies
 Body *h_bodies = NULL;      // Host (CPU) memory for bodies
 OctreeNode *d_nodes = NULL; // Device memory for octree nodes
 int *d_node_count = NULL;   // Device variable tracking node count
 int h_node_count = 0;       // Host copy of node count
 
 /******************************
  * CUDA UTILITIES
  ******************************/
 
 // Macro for checking CUDA errors
 #define checkCudaErrors(call) { \
     const cudaError_t error = call; \
     if (error != cudaSuccess) { \
         printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
         printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
         exit(1); \
     } \
 }

// Device function to determine which octant a body belongs to
__device__ int getOctant(float x, float y, float z, float cx, float cy, float cz) {
    int oct = 0;
    if (x >= cx) oct |= 1;
    if (y >= cy) oct |= 2;
    if (z >= cz) oct |= 4;
    return oct;
}

// Device function to compute boundaries of a child octant
__device__ void computeChildBounds(
    int octant, 
    float min_x, float min_y, float min_z, 
    float max_x, float max_y, float max_z,
    float* new_min_x, float* new_min_y, float* new_min_z,
    float* new_max_x, float* new_max_y, float* new_max_z
) {
    float mid_x = (min_x + max_x) * 0.5f;
    float mid_y = (min_y + max_y) * 0.5f;
    float mid_z = (min_z + max_z) * 0.5f;
    
    *new_min_x = (octant & 1) ? mid_x : min_x;
    *new_max_x = (octant & 1) ? max_x : mid_x;
    
    *new_min_y = (octant & 2) ? mid_y : min_y;
    *new_max_y = (octant & 2) ? max_y : mid_y;
    
    *new_min_z = (octant & 4) ? mid_z : min_z;
    *new_max_z = (octant & 4) ? max_z : mid_z;
}

// Kernel to initialize the octree
__global__ void initOctreeKernel(OctreeNode* nodes, int* node_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize root node (index 0)
        nodes[0].mass = 0.0f;
        nodes[0].cx = nodes[0].cy = nodes[0].cz = 0.0f;
        nodes[0].min_x = nodes[0].min_y = nodes[0].min_z = -1.0f;
        nodes[0].max_x = nodes[0].max_y = nodes[0].max_z = 1.0f;
        nodes[0].is_leaf = 1;
        nodes[0].body_idx = -1;
        
        for (int i = 0; i < 8; i++) {
            nodes[0].children[i] = -1;
        }
        
        *node_count = 1; // Start with just the root node
    }
}

// Kernel to insert bodies into the octree
__global__ void insertBodiesKernel(Body* bodies, OctreeNode* nodes, int* node_count, int num_bodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) return;
    
    Body body = bodies[idx];
    
    // Skip bodies outside the simulation bounds
    if (body.x < -1.0f || body.x > 1.0f || 
        body.y < -1.0f || body.y > 1.0f || 
        body.z < -1.0f || body.z > 1.0f) {
        return;
    }
    
    // Start at the root node
    int current = 0;
    int depth = 0;
    
    while (depth < MAX_DEPTH) {
        // Atomically update the node's center of mass
        float node_mass = atomicAdd(&nodes[current].mass, body.mass);
        float total_mass = node_mass + body.mass;
        
        if (total_mass > 0) {
            atomicExch(&nodes[current].cx, (node_mass * nodes[current].cx + body.mass * body.x) / total_mass);
            atomicExch(&nodes[current].cy, (node_mass * nodes[current].cy + body.mass * body.y) / total_mass);
            atomicExch(&nodes[current].cz, (node_mass * nodes[current].cz + body.mass * body.z) / total_mass);
        }
        
        // If it's a leaf node with no body, assign this body
        if (nodes[current].is_leaf && nodes[current].body_idx == -1) {
            if (atomicCAS(&nodes[current].body_idx, -1, idx) == -1) {
                // Successfully assigned body to this node
                break;
            }
            // Someone else assigned a body, continue to subdivision
        }
        
        // If it's a leaf node with a body, we need to subdivide
        if (nodes[current].is_leaf && nodes[current].body_idx != -1) {
            // Mark as non-leaf
            atomicExch(&nodes[current].is_leaf, 0);
            
            // Get the body that was already in this node
            int old_idx = nodes[current].body_idx;
            Body old_body = bodies[old_idx];
            
            // Determine which octant the existing body belongs to
            float center_x = (nodes[current].min_x + nodes[current].max_x) * 0.5f;
            float center_y = (nodes[current].min_y + nodes[current].max_y) * 0.5f;
            float center_z = (nodes[current].min_z + nodes[current].max_z) * 0.5f;
            
            int old_octant = getOctant(old_body.x, old_body.y, old_body.z, center_x, center_y, center_z);
            
            // Create child node for the existing body
            if (nodes[current].children[old_octant] == -1) {
                int new_node_idx = atomicAdd(node_count, 1);
                if (new_node_idx < MAX_NODES) {
                    float new_min_x, new_min_y, new_min_z, new_max_x, new_max_y, new_max_z;
                    computeChildBounds(old_octant, 
                                     nodes[current].min_x, nodes[current].min_y, nodes[current].min_z,
                                     nodes[current].max_x, nodes[current].max_y, nodes[current].max_z,
                                     &new_min_x, &new_min_y, &new_min_z,
                                     &new_max_x, &new_max_y, &new_max_z);
                    
                    // Initialize the new node
                    nodes[new_node_idx].mass = old_body.mass;
                    nodes[new_node_idx].cx = old_body.x;
                    nodes[new_node_idx].cy = old_body.y;
                    nodes[new_node_idx].cz = old_body.z;
                    nodes[new_node_idx].min_x = new_min_x;
                    nodes[new_node_idx].min_y = new_min_y;
                    nodes[new_node_idx].min_z = new_min_z;
                    nodes[new_node_idx].max_x = new_max_x;
                    nodes[new_node_idx].max_y = new_max_y;
                    nodes[new_node_idx].max_z = new_max_z;
                    nodes[new_node_idx].is_leaf = 1;
                    nodes[new_node_idx].body_idx = old_idx;
                    
                    for (int i = 0; i < 8; i++) {
                        nodes[new_node_idx].children[i] = -1;
                    }
                    
                    // Link the parent to this child
                    atomicExch(&nodes[current].children[old_octant], new_node_idx);
                }
            }
        }
        
        // Determine which octant the current body belongs to
        float center_x = (nodes[current].min_x + nodes[current].max_x) * 0.5f;
        float center_y = (nodes[current].min_y + nodes[current].max_y) * 0.5f;
        float center_z = (nodes[current].min_z + nodes[current].max_z) * 0.5f;
        
        int octant = getOctant(body.x, body.y, body.z, center_x, center_y, center_z);
        
        // If the child doesn't exist, create it
        if (nodes[current].children[octant] == -1) {
            int new_node_idx = atomicAdd(node_count, 1);
            if (new_node_idx < MAX_NODES) {
                float new_min_x, new_min_y, new_min_z, new_max_x, new_max_y, new_max_z;
                computeChildBounds(octant, 
                                 nodes[current].min_x, nodes[current].min_y, nodes[current].min_z,
                                 nodes[current].max_x, nodes[current].max_y, nodes[current].max_z,
                                 &new_min_x, &new_min_y, &new_min_z,
                                 &new_max_x, &new_max_y, &new_max_z);
                
                // Initialize the new node
                nodes[new_node_idx].mass = 0.0f;
                nodes[new_node_idx].cx = 0.0f;
                nodes[new_node_idx].cy = 0.0f;
                nodes[new_node_idx].cz = 0.0f;
                nodes[new_node_idx].min_x = new_min_x;
                nodes[new_node_idx].min_y = new_min_y;
                nodes[new_node_idx].min_z = new_min_z;
                nodes[new_node_idx].max_x = new_max_x;
                nodes[new_node_idx].max_y = new_max_y;
                nodes[new_node_idx].max_z = new_max_z;
                nodes[new_node_idx].is_leaf = 1;
                nodes[new_node_idx].body_idx = -1;
                
                for (int i = 0; i < 8; i++) {
                    nodes[new_node_idx].children[i] = -1;
                }
                
                // Link the parent to this child
                atomicExch(&nodes[current].children[octant], new_node_idx);
            }
        }
        
        // Move to the appropriate child node
        int next_node = nodes[current].children[octant];
        if (next_node == -1) break; // Something went wrong
        
        current = next_node;
        depth++;
    }
}

/**
 * brief Computes forces on all bodies using Barnes-Hut approximation
 * 
 * param bodies Array of bodies
 * param nodes Octree nodes
 * param num_bodies Total number of bodies
 * param theta Barnes-Hut opening angle parameter
 * param dt Time step
 */
__global__ void barnes_hut_kernel(Body* bodies, OctreeNode* nodes, int num_bodies, float theta, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) return;
    
    Body body = bodies[idx];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f; // Force components
    
    // Stack-based tree traversal (avoid recursion)
    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = 0; // Start with root node
    
    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        OctreeNode node = nodes[node_idx];
        
        // Calculate distance between body and node's center of mass
        float dx = node.cx - body.x;
        float dy = node.cy - body.y;
        float dz = node.cz - body.z;
        float dist_squared = dx*dx + dy*dy + dz*dz + SOFTENING;
        
        // Calculate the width of the node's bounding box
        float node_width = node.max_x - node.min_x;
        
        // Barnes-Hut criterion: if (s/d < θ) or leaf node, use as-is
        if (node.is_leaf || (node_width * node_width) / dist_squared < theta * theta) {
            // Skip self-interaction
            if (node.is_leaf && node.body_idx == idx) continue;
            
            // Calculate gravitational force
            float dist = sqrtf(dist_squared);
            float force = G * body.mass * node.mass / (dist_squared * dist);
            
            // Accumulate force components
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        } else {
            // Node is too close, need to visit children
            for (int i = 0; i < 8; i++) {
                int child = node.children[i];
                if (child != -1) {
                    stack[stack_top++] = child;
                }
            }
        }
    }
    
    // Update velocity using computed force (F = ma => a = F/m)
    bodies[idx].vx += fx * dt / body.mass;
    bodies[idx].vy += fy * dt / body.mass;
    bodies[idx].vz += fz * dt / body.mass;
    
    // Update position using new velocity
    bodies[idx].x += bodies[idx].vx * dt;
    bodies[idx].y += bodies[idx].vy * dt;
    bodies[idx].z += bodies[idx].vz * dt;
}

/******************************
 * SIMULATION FUNCTIONS
 ******************************/

/**
 * brief Initializes simulation data
 * Creates initial galaxy-like distribution of bodies
 */
void initSimulation() {
    // Allocate host memory for bodies
    h_bodies = (Body*)malloc(N * sizeof(Body));
    
    // Initialize bodies in a spherical distribution with orbital velocities
    for (int i = 0; i < N; i++) {
        // Use spherical coordinates for galaxy-like distribution
        // Radius concentrated toward center (powf with exponent < 1)
        float radius = 0.5f * powf(rand() / (float)RAND_MAX, 0.5f);
        float theta = 2.0f * M_PI * (rand() / (float)RAND_MAX); // Azimuthal angle
        float phi = acosf(2.0f * (rand() / (float)RAND_MAX) - 1.0f); // Polar angle
        
        // Convert to Cartesian coordinates
        h_bodies[i].x = radius * sinf(phi) * cosf(theta);
        h_bodies[i].y = radius * sinf(phi) * sinf(theta);
        h_bodies[i].z = radius * cosf(phi) * 0.2f; // Flatten in z-direction
        
        // Add orbital velocity for galaxy-like rotation
        // v = sqrt(G*M/r), where M is central mass (arbitrarily set to 100)
        float orbit_speed = sqrtf(G * 10000.0f / radius) * 0.5f;
        h_bodies[i].vx = -orbit_speed * sinf(theta); // Tangential velocity
        h_bodies[i].vy = orbit_speed * cosf(theta);
        h_bodies[i].vz = 0.0f;
        
        // Assign random mass
        h_bodies[i].mass = 0.1f + 0.9f * (rand() / (float)RAND_MAX);
    }
    
    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_bodies, N * sizeof(Body)));
    checkCudaErrors(cudaMalloc(&d_nodes, MAX_NODES * sizeof(OctreeNode)));
    checkCudaErrors(cudaMalloc(&d_node_count, sizeof(int)));
    
    // Copy initial data to device
    checkCudaErrors(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body), cudaMemcpyHostToDevice));
}

/**
 * brief Updates simulation state for one time step
 * 1. Rebuilds octree
 * 2. Computes forces
 * 3. Updates positions/velocities
 */
void updateSimulation() {
    // Configure CUDA kernel launch parameters
    dim3 blocks((N + 255) / 256); // Enough blocks to cover all bodies
    dim3 threads(256);            // 256 threads per block
    
    // Reset octree node count
    h_node_count = 0;
    checkCudaErrors(cudaMemcpy(d_node_count, &h_node_count, sizeof(int), cudaMemcpyHostToDevice));
    
    // 1. Initialize octree (single thread)
    initOctreeKernel<<<1, 1>>>(d_nodes, d_node_count);
    
    // 2. Insert all bodies into octree (parallel across bodies)
    insertBodiesKernel<<<blocks, threads>>>(d_bodies, d_nodes, d_node_count, N);
    
    // 3. Compute forces and update positions (parallel across bodies)
    barnes_hut_kernel<<<blocks, threads>>>(d_bodies, d_nodes, N, THETA, DT);
    
    // Check for errors
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy data back to host for rendering
    checkCudaErrors(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost));
    
    // For debugging: check how many nodes were created
    checkCudaErrors(cudaMemcpy(&h_node_count, d_node_count, sizeof(int), cudaMemcpyDeviceToHost));
    // printf("Octree node count: %d\n", h_node_count); // Uncomment to see
}

/******************************
 * OPENGL VISUALIZATION FUNCTIONS
 ******************************/
void render() {
    // Clear screen and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Set up view transformation
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0); // Camera at (0,0,3) looking at origin
    
    // Rotate view slowly for better visualization
    static float angle = 0.0f;
    angle += 0.1f;
    glRotatef(angle, 0.0f, 1.0f, 0.0f); // Rotate around Y-axis
    
    // Draw all bodies as points
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < N; i++) {
        // Color based on velocity (blue = slow, red = fast)
        float speed = sqrtf(h_bodies[i].vx * h_bodies[i].vx + 
                          h_bodies[i].vy * h_bodies[i].vy + 
                          h_bodies[i].vz * h_bodies[i].vz);
        float r = fminf(1.0f, speed * 20.0f);     // Red increases with speed
        float g = fminf(1.0f, h_bodies[i].mass);  // Green based on mass
        float b = fminf(1.0f, 1.0f - speed * 10.0f); // Blue decreases with speed
        
        glColor3f(r, g, b);
        glVertex3f(h_bodies[i].x, h_bodies[i].y, h_bodies[i].z);
    }
    glEnd();
    
    // Swap buffers to display
    glutSwapBuffers();
}

// Idle function for animation
void idle() {
    updateSimulation();
    glutPostRedisplay(); // Trigger render()
}

// Keyboard controls
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESC key
            // Cleanup
            free(h_bodies);
            cudaFree(d_bodies);
            cudaFree(d_nodes);
            cudaFree(d_node_count);
            exit(0);
            break;
    }
}

// Mouse controls
void mouse(int button, int state, int x, int y) {
    // Handle mouse interactions
}

// Mouse motion controls
void motion(int x, int y) {
    // Handle mouse motion
}

// Initialize OpenGL
void initGL(int width, int height) {
    // Set background color (dark blue)
    glClearColor(0.0f, 0.0f, 0.1f, 1.0f);
    
    // Enable depth testing and point smoothing
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Set up viewport and projection matrix
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width/(float)height, 0.1f, 100.0f);
}


// Window resize handler
void reshape(int width, int height) {
    initGL(width, height);
}

/******************************
 * BENCHMARKING FUNCTION
 ******************************/

/**
 * @brief Runs simulation without visualization to measure performance
 * @param iterations Number of iterations to run
 */
void runBenchmark(int iterations) {
    printf("Starting benchmark for N=%d bodies...\n", N);
    
    // Warm-up run to initialize everything
    updateSimulation();
    
    // CUDA events for precise timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0;
    
    // Run simulation iterations
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        updateSimulation();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        
        // Print progress
        if ((i+1) % 10 == 0) {
            printf("Completed %d/%d iterations...\n", i+1, iterations);
        }
    }
    
    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Print benchmark results
    printf("\nBarnes-Hut Benchmark Results:\n");
    printf("============================\n");
    printf("Number of bodies (N):    %d\n", N);
    printf("Iterations:              %d\n", iterations);
    printf("Total computation time:  %.3f ms\n", total_time);
    printf("Average per iteration:   %.3f ms\n", total_time/iterations);
    printf("Throughput:              %.1f bodies/ms\n", (N*iterations)/total_time);
    printf("============================\n");
}

/******************************
* MAIN FUNCTION
******************************/

 int main(int argc, char **argv) {
    // Check for benchmark mode flag
    int benchmark_mode = 0;
    if (argc > 1 && strcmp(argv[1], "--benchmark") == 0) {
        benchmark_mode = 1;
    }

    if (!benchmark_mode) {
        // Initialize GLUT for visualization
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(800, 600);
        glutCreateWindow("CUDA Barnes-Hut N-Body Simulation");
        initGL(800, 600);
    }

    // Initialize simulation data
    initSimulation();

    if (benchmark_mode) {
        // Run in benchmark mode (no visualization)
        runBenchmark(100);  // Run 100 iterations
        return 0;
    } else {
        // Set up GLUT callback functions
        glutDisplayFunc(render);    // Display callback
        glutIdleFunc(idle);         // Animation callback
        glutKeyboardFunc(keyboard); // Keyboard handler
        glutReshapeFunc(reshape);   // Window resize handler
        glutMouseFunc(mouse);       // Mouse click handler
        glutMotionFunc(motion);     // Mouse motion handler
        
        // Print instructions
        printf("Starting Barnes-Hut simulation with %d bodies\n", N);
        printf("Press ESC to exit\n");
        
        // Start main GLUT loop
        glutMainLoop();
    }

    return 0;
}
