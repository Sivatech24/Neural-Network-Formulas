#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <windows.h>  // For enabling ANSI colors on Windows

#define N 100

// Enable ANSI color in Windows terminal
void enableAnsiColors() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hOut, dwMode);
}

// Load input data from file
void loadInput(const char* filename, float* input) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Error: Cannot open file " << filename << "\n";
        exit(1);
    }

    int i = 0;
    while (file >> input[i] && i < N) {
        ++i;
    }

    if (i < N) {
        std::cerr << "⚠️ Warning: Only loaded " << i << " values from " << filename << ", filling rest with 0\n";
        for (; i < N; ++i) input[i] = 0.0f;
    }
}

// Forward pass kernel with ReLU activation
__global__ void forwardPassKernel(const float* input, const float* weights, float* output, float bias) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float z = input[i] * weights[i] + bias;
        output[i] = fmaxf(z, 0.0f);  // ReLU
    }
}

// Visualize output with colored bar
void visualizeOutput(const float* output) {
    std::cout << "\n🔎 Neural Output Visualization:\n";
    for (int i = 0; i < N; ++i) {
        int color = static_cast<int>(output[i] * 255.0f);
        color = std::max(0, std::min(255, color));
        std::cout << "\033[48;2;" << color << ";" << 255 - color << ";100m  \033[0m";
    }
    std::cout << "\n";
}

// CUDA error checking
void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "❌ CUDA Error: " << message << " — " << cudaGetErrorString(result) << "\n";
        exit(1);
    }
}

int main() {
    enableAnsiColors();  // for Windows terminal support

    float *h_input = new float[N];
    float *h_weights = new float[N];
    float *h_output = new float[N];
    float bias = 0.1f;

    loadInput("data.txt", h_input);

    for (int i = 0; i < N; ++i)
        h_weights[i] = 0.5f;  // dummy weight value

    float *d_input, *d_weights, *d_output;
    checkCuda(cudaMalloc(&d_input, N * sizeof(float)), "allocating d_input");
    checkCuda(cudaMalloc(&d_weights, N * sizeof(float)), "allocating d_weights");
    checkCuda(cudaMalloc(&d_output, N * sizeof(float)), "allocating d_output");

    checkCuda(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice), "copying input");
    checkCuda(cudaMemcpy(d_weights, h_weights, N * sizeof(float), cudaMemcpyHostToDevice), "copying weights");

    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    forwardPassKernel<<<gridSize, blockSize>>>(d_input, d_weights, d_output, bias);
    checkCuda(cudaDeviceSynchronize(), "kernel execution");

    checkCuda(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost), "copying output");

    visualizeOutput(h_output);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;

    return 0;
}
