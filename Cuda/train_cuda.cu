#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.01f
#define EPOCHS 500

// Activation functions
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

// Forward propagation
__global__ void forward_kernel(float* X, float* W1, float* B1, float* A1,
                               float* W2, float* B2, float* OUT, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float z = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            z += X[idx * INPUT_SIZE + i] * W1[i * HIDDEN_SIZE + j];
        }
        z += B1[j];
        A1[idx * HIDDEN_SIZE + j] = sigmoid(z);
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        float z = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            z += A1[idx * HIDDEN_SIZE + j] * W2[j * OUTPUT_SIZE + k];
        }
        z += B2[k];
        OUT[idx * OUTPUT_SIZE + k] = sigmoid(z);
    }
}

// Backpropagation
__global__ void backward_kernel(float* X, float* Y, float* W1, float* B1, float* A1,
                                float* W2, float* B2, float* OUT, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float d_out = OUT[idx] - Y[idx];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float a1 = A1[idx * HIDDEN_SIZE + j];
        float d_w2 = d_out * sigmoid_derivative(OUT[idx]) * a1;
        W2[j] -= LEARNING_RATE * d_w2;
    }
    B2[0] -= LEARNING_RATE * d_out * sigmoid_derivative(OUT[idx]);

    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float x = X[idx * INPUT_SIZE + i];
            float a1 = A1[idx * HIDDEN_SIZE + j];
            float d_hidden = d_out * W2[j] * sigmoid_derivative(a1);
            float d_w1 = d_hidden * x;
            W1[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_w1;
            if (i == 0) B1[j] -= LEARNING_RATE * d_hidden;
        }
    }
}

// Binary file loaders
bool load_bin_file(const char* filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(data), size);
    return true;
}

bool save_bin_file(const char* filename, float* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<char*>(data), size);
    return true;
}

// Training function with dynamic sample size
void train(float* h_X, float* h_Y, float* output_host, int num_samples) {
    float* d_X, * d_Y, * d_W1, * d_B1, * d_A1, * d_W2, * d_B2, * d_OUT;

    cudaMalloc(&d_X, INPUT_SIZE * num_samples * sizeof(float));
    cudaMalloc(&d_Y, num_samples * sizeof(float));
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_B1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_A1, num_samples * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_B2, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_OUT, num_samples * sizeof(float));

    float h_W1[INPUT_SIZE * HIDDEN_SIZE], h_B1[HIDDEN_SIZE];
    float h_W2[HIDDEN_SIZE * OUTPUT_SIZE], h_B2[OUTPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_W1[i] = (rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < HIDDEN_SIZE; i++) h_B1[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_W2[i] = (rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < OUTPUT_SIZE; i++) h_B2[i] = 0.0f;

    cudaMemcpy(d_X, h_X, INPUT_SIZE * num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, sizeof(h_W1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, sizeof(h_B1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, sizeof(h_W2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, sizeof(h_B2), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((num_samples + blockSize.x - 1) / blockSize.x);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        forward_kernel<<<gridSize, blockSize>>>(d_X, d_W1, d_B1, d_A1, d_W2, d_B2, d_OUT, num_samples);
        cudaDeviceSynchronize();

        backward_kernel<<<gridSize, blockSize>>>(d_X, d_Y, d_W1, d_B1, d_A1, d_W2, d_B2, d_OUT, num_samples);
        cudaDeviceSynchronize();

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " complete.\n";
    }

    cudaMemcpy(output_host, d_OUT, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W1); cudaFree(d_B1);
    cudaFree(d_A1); cudaFree(d_W2); cudaFree(d_B2); cudaFree(d_OUT);
}

int main() {
    std::ifstream fin_input("input.bin", std::ios::binary | std::ios::ate);
    std::ifstream fin_target("target.bin", std::ios::binary | std::ios::ate);

    if (!fin_input || !fin_target) {
        std::cerr << "Failed to open input.bin or target.bin\n";
        return 1;
    }

    size_t input_bytes = fin_input.tellg();
    size_t target_bytes = fin_target.tellg();
    fin_input.close();
    fin_target.close();

    if (input_bytes % (INPUT_SIZE * sizeof(float)) != 0 || target_bytes % sizeof(float) != 0) {
        std::cerr << "Binary file sizes not aligned\n";
        return 1;
    }

    int num_samples = static_cast<int>(input_bytes / (INPUT_SIZE * sizeof(float)));
    if (target_bytes / sizeof(float) != num_samples) {
        std::cerr << "Mismatch in sample count\n";
        return 1;
    }

    std::cout << "Training with " << num_samples << " samples\n";

    float* input_data = new float[INPUT_SIZE * num_samples];
    float* target_data = new float[num_samples];
    float* predicted_output = new float[num_samples];

    if (!load_bin_file("input.bin", input_data, input_bytes)) {
        std::cerr << "Failed to load input.bin\n";
        return 1;
    }

    if (!load_bin_file("target.bin", target_data, target_bytes)) {
        std::cerr << "Failed to load target.bin\n";
        return 1;
    }

    train(input_data, target_data, predicted_output, num_samples);

    if (!save_bin_file("output.bin", predicted_output, num_samples * sizeof(float))) {
        std::cerr << "Failed to save output.bin\n";
        return 1;
    }

    std::cout << "Training complete. Output saved to output.bin\n";

    delete[] input_data;
    delete[] target_data;
    delete[] predicted_output;

    return 0;
}
