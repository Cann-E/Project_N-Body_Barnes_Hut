CC = g++
NVCC = nvcc
CFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra
NVCCFLAGS = -O3 -arch=sm_70 -std=c++17 --expt-relaxed-constexpr
LDFLAGS = -lcudart -L$(CUDA_PATH)/lib64

SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
WEB_DIR = web

# Output files
PARTICLE_DATA = $(WEB_DIR)/particles.bin
TREE_DATA = $(WEB_DIR)/tree_structure.json
VISUALIZER = $(WEB_DIR)/visualizer.html

CPU_SOURCES = $(SRC_DIR)/cpu_direct.cpp $(SRC_DIR)/cpu_barnes_hut.cpp
GPU_SOURCES = $(SRC_DIR)/gpu_barnes_hut.cu
MAIN_SOURCE = $(SRC_DIR)/main.cpp

CPU_OBJECTS = $(CPU_SOURCES:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)
GPU_OBJECTS = $(GPU_SOURCES:$(SRC_DIR)/%.cu=$(BIN_DIR)/%.o)
MAIN_OBJECT = $(BIN_DIR)/main.o

EXEC = nbody_sim

all: $(BIN_DIR) $(WEB_DIR) $(EXEC)

$(BIN_DIR) $(WEB_DIR):
	mkdir -p $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) -dc $< -o $@

$(EXEC): $(CPU_OBJECTS) $(GPU_OBJECTS) $(MAIN_OBJECT)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR) $(EXEC) $(WEB_DIR)/*.bin $(WEB_DIR)/*.json

run: $(EXEC) $(WEB_DIR)
	@read -p "Enter number of particles: " N && \
	read -p "Enter theta (default 0.5): " THETA && \
	./$(EXEC) $$N $${THETA:-0.5} && \
	echo -e "\nData saved to:\n- $(PARTICLE_DATA)\n- $(TREE_DATA)" && \
	echo -e "\nStarting web server at http://localhost:8080" && \
	cd $(WEB_DIR) && python3 -m http.server 8080

visualize: $(WEB_DIR)
	cd $(WEB_DIR) && python3 -m http.server 8080

.PHONY: all clean run visualize
