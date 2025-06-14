#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define N 100

// Load from .txt
void loadInput(const char* filename, float* input) {
    std::ifstream file(filename);
    for (int i = 0; i < N && file >> input[i]; ++i);
}

// Kernel for forward pass: Z = X * W + b; A = ReLU(Z)
__global__ void forwardPassKernel(const float* input, const float* weights, float* output, float bias) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float z = input[i] * weights[i] + bias;
        output[i] = z > 0 ? z : 0;  // ReLU
    }
}

// Terminal color print (host side)
void visualizeOutput(const float* output) {
    for (int i = 0; i < N; ++i) {
        int color = static_cast<int>(output[i] * 255.0f);
        if (color > 255) color = 255;
        if (color < 0) color = 0;
        std::cout << "\033[48;2;" << color << ";" << 255 - color << ";100m  \033[0m";
    }
    std::cout << std::endl;
}

int main() {
    // Allocate host memory
    float *h_input = new float[N];
    float *h_weights = new float[N];
    float *h_output = new float[N];
    float bias = 0.1f;

    loadInput("data.txt", h_input);
    for (int i = 0; i < N; ++i)
        h_weights[i] = 0.5f;  // dummy weights

    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_weights, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input and weights to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    forwardPassKernel<<<gridSize, blockSize>>>(d_input, d_weights, d_output, bias);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Visualize result
    visualizeOutput(h_output);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;

    return 0;
}
