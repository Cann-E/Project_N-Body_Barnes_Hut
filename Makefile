# Barnes-Hut CUDA/OpenGL Makefile
# For MobaXterm with working X11 forwarding

# Compiler and flags
NVCC        = nvcc
NVCC_FLAGS  = -O3 -arch=sm_70
LIBS        = -lglut -lGL -lGLU -lGLEW -lm
TARGET      = nbody_sim
SRC         = barnes_hut.cu

# Build and run
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

visualize: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	@echo "Running Barnes Hut for N bodies..."
	@./$(TARGET) --benchmark

clean:
	rm -f $(TARGET) *.o

.PHONY: all run benchmark clean
