CC = g++
NVCC = nvcc
CFLAGS = -O3 -march=native -std=c++17
NVCCFLAGS = -O3 -arch=sm_70 -diag-suppress=20012  # Suppress default constructor warnings
LDFLAGS = -lcudart

SRC_DIR = src
INC_DIR = include
BIN = bin

CPU_SOURCES = $(SRC_DIR)/cpu_direct.cpp $(SRC_DIR)/cpu_barnes_hut.cpp $(SRC_DIR)/main.cpp
CPU_OBJECTS = $(CPU_SOURCES:$(SRC_DIR)/%.cpp=$(BIN)/%.o)

GPU_SOURCES = $(SRC_DIR)/gpu_barnes_hut.cu
GPU_OBJECTS = $(GPU_SOURCES:$(SRC_DIR)/%.cu=$(BIN)/%.o)

EXEC = nbody_sim

all: $(BIN) $(EXEC)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

$(BIN)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) -c $< -o $@

$(EXEC): $(CPU_OBJECTS) $(GPU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN) $(EXEC)

.PHONY: all clean