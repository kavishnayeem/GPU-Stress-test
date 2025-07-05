# GPU Stress Test Project

A comprehensive GPU stress testing harness designed to push NVIDIA T4 (Turing TU104) GPUs to their thermal and memory bandwidth limits. This project combines CUDA kernels, Vulkan compute shaders, and multi-threaded C++ to create an end-to-end stress testing solution.

## 🎯 Project Overview

This project demonstrates a sophisticated GPU stress testing framework that:
- **Memory Stress**: Streams ≈14 GB of VRAM using custom CUDA kernels
- **Compute Stress**: Emulates transformer feed-forward networks with Vulkan compute shaders
- **Thermal Stress**: Pushes the GPU to its thermal limits (ΔT ≈ +35°C in 4 minutes)
- **Bandwidth Stress**: Saturates the 192 GB/s GDDR6 memory bus

## 🖥️ Target Hardware

**NVIDIA T4 (TU104) Specifications:**
- 2,560 FP32 CUDA cores (64 ALUs per SM)
- 320 Tensor cores (3rd generation)
- 320 INT cores for mixed-precision operations
- 15 GB GDDR6 @ 192 GB/s throughput
- 40 Streaming Multiprocessor clusters

## 🏗️ Architecture

The stress test operates through three main components:

### 1. CUDA Memory Flood Kernel
- Allocates ≈80% of available VRAM (>11 GB)
- Launches 67 million threads (65535 blocks × 1024 threads)
- Forces uncached reads/writes to drive GDDR6 bandwidth >190 GB/s

### 2. Vulkan Transformer Shader
- Implements transformer block with 8,192-dim hidden size
- 4× feed-forward expansion with >3 GB weight tensors
- 16.8 million shader invocations per pass
- Saturates tensor cores and L2 cache

### 3. Multi-threaded Host System
- **CPU Stress**: OpenMP-SIMD transcendental math on 800M-float arrays
- **PCIe Stress**: 16 parallel CUDA streams copying 2 GB buffers
- **Monitoring**: NVML logging every 500ms for GPU%, memory%, VRAM usage, and temperature

## 📊 Results

### Thermal Performance
- **Idle Temperature**: 46°C
- **Peak Temperature**: 81°C (after 4 minutes)
- **Temperature Rise**: +35°C under sustained load
- **GPU Utilization**: 98% sustained
- **Memory Controller**: 93% utilization

### Memory Performance
- **VRAM Usage**: 95% of 14.9 GB capacity
- **Memory Bandwidth**: >190 GB/s sustained
- **Memory Controller**: 93% utilization

## 🛠️ Technologies Used

- **CUDA 12.4**: Custom kernels and unified memory telemetry
- **Vulkan 1.3**: SPIR-V compute pipelines
- **NVML**: Low-overhead GPU monitoring (500ms intervals)
- **OpenMP 4.5**: Host-side SIMD CPU load generation
- **C++**: Multi-threaded orchestration
- **GLSL**: Compute shader implementation

## 📁 Project Structure

```
GPU - Stress Test/
├── index.html          # Interactive project showcase
├── Stress_test.ipynb   # Jupyter notebook with implementation
├── Graph.png          # GPU usage visualization
├── T4.jpg             # NVIDIA T4 GPU image
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.4+
- Vulkan SDK 1.3+
- NVML (NVIDIA Management Library)
- OpenMP-compatible compiler
- Linux environment (tested on Ubuntu 22.04)

### Installation

1. **Install Dependencies**:
```bash
sudo apt-get update
sudo apt-get install -y libvulkan-dev glslang-tools libxxf86vm-dev \
    libxrandr-dev libwayland-dev libnvidia-ml-dev g++ nvidia-cuda-toolkit
```

2. **Install Vulkan Components**:
```bash
sudo apt-get install -y --no-install-recommends \
    vulkan-tools \
    libvulkan-dev \
    nvidia-driver-525 \
    nvidia-utils-525
```

3. **Configure Vulkan Loader**:
```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libvulkan.so.1.3.224 /usr/lib/x86_64-linux-gnu/libvulkan.so.1
sudo ln -s /usr/lib/x86_64-linux-gnu/libvulkan.so.1 /usr/lib/x86_64-linux-gnu/libvulkan.so
```

4. **Build GLSL Compiler**:
```bash
git clone https://github.com/KhronosGroup/glslang.git
cd glslang
python3 update_glslang_sources.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GLSLANG_BINARIES=ON
cmake --build build -j4
sudo cp build/StandAlone/glslangValidator /usr/local/bin/
```

### Running the Stress Test

1. Open the Jupyter notebook:
```bash
jupyter notebook Stress_test.ipynb
```

2. Execute the cells in sequence to:
   - Set up the environment
   - Compile the compute shader
   - Launch the stress test
   - Monitor results

## 📈 Monitoring

The stress test provides real-time monitoring through NVML:
- GPU utilization percentage
- Memory controller utilization
- VRAM usage (current/total)
- GPU temperature
- Power consumption

## ⚠️ Safety Considerations

- **Thermal Limits**: The test pushes the GPU to 81°C - ensure adequate cooling
- **Power Draw**: Approaches the 70W TDP limit of the T4
- **Memory Usage**: Uses 95% of available VRAM
- **Duration**: Designed for short-term stress testing (5-10 minutes)

## 🎯 Use Cases

- **GPU Validation**: Verify GPU stability under extreme load
- **Thermal Testing**: Test cooling system effectiveness
- **Performance Benchmarking**: Measure sustained compute and memory performance
- **Research**: Study GPU behavior under stress conditions
- **Development**: Test GPU-intensive applications

## 📝 License

This project is for educational and research purposes. Please ensure you have proper cooling and monitoring when running GPU stress tests.

## 🤝 Contributing

Contributions are welcome! Please ensure any modifications maintain the safety and monitoring capabilities of the original stress test.

---

**Note**: This stress test is designed for the NVIDIA T4 GPU. Results may vary on other GPU models, and appropriate adjustments may be needed for different hardware configurations. 