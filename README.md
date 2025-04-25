# Barnes-Hut N-Body Simulation

A hybrid CPU/GPU implementation of the Barnes-Hut tree algorithm for N-body simulations, with performance benchmarks.

## Features
- **CPU Direct** (O(N²)) - Baseline brute-force method
- **CPU Barnes-Hut** (O(N log N)) - Octree-based approximation
- **GPU Barnes-Hut** (O(N log N)) - CUDA-accelerated version
- **Auto-benchmarking** with time and speedup reporting

## To Compile and Run on GPU Server

```bash
mkdir barnes_hut
```
```bash
cd barnes_hut
```

### 1. For cleaning tmp files
```bash
make clean
```
### 2. After making changes to code
 ```bash
 make
  ```
### 3. Checking the code time
```bash
./nbody_sim
```

### Note:

Since, there are bit restriction on GPU Server, the code will compile but on .h .cpp files, it'll say 
"cannot open source file" on includes line If working on VS code studio.
there is a way to fix but Github will hide this file.

Create folder " .vscode " folder under barnes_hut, Then create file under .vscode name it " c_cpp_properties.json "

and add this in the file and save it.

{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++14",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}

#### This will allow to use open source file on the server.
