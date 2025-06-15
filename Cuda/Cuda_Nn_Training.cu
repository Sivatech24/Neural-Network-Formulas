// CUDA-based Neural Network for BTC Prediction with Better Learning
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1
#define NUM_SAMPLES 10000
#define LEARNING_RATE 0.1f
#define EPOCHS 100

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

__global__ void forward(float* X, float* W1, float* B1, float* A1,
                        float* W2, float* B2, float* OUT, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= samples) return;

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            sum += X[idx * INPUT_SIZE + i] * W1[i * HIDDEN_SIZE + j];
        }
        sum += B1[j];
        A1[idx * HIDDEN_SIZE + j] = sigmoid(sum);
    }

    for (int k = 0; k < OUTPUT_SIZE; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            sum += A1[idx * HIDDEN_SIZE + j] * W2[j * OUTPUT_SIZE + k];
        }
        sum += B2[k];
        OUT[idx * OUTPUT_SIZE + k] = sigmoid(sum);
    }
}

__global__ void backward(float* X, float* Y, float* W1, float* B1, float* A1,
                         float* W2, float* B2, float* OUT, int samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= samples) return;

    float y_true = Y[idx];
    float y_pred = OUT[idx];
    float d_out = (y_pred - y_true) * sigmoid_derivative(y_pred);

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float a = A1[idx * HIDDEN_SIZE + j];
        float grad_w2 = d_out * a;
        atomicAdd(&W2[j], -LEARNING_RATE * grad_w2);
    }
    atomicAdd(&B2[0], -LEARNING_RATE * d_out);

    for (int j = 0; j < HIDDEN_SIZE; ++j) {
        float da = d_out * W2[j] * sigmoid_derivative(A1[idx * HIDDEN_SIZE + j]);
        for (int i = 0; i < INPUT_SIZE; ++i) {
            float grad_w1 = da * X[idx * INPUT_SIZE + i];
            atomicAdd(&W1[i * HIDDEN_SIZE + j], -LEARNING_RATE * grad_w1);
        }
        atomicAdd(&B1[j], -LEARNING_RATE * da);
    }
}

void load_data(const char* input_file, const char* target_file, float*& X, float*& Y) {
    std::ifstream fin(input_file, std::ios::binary);
    std::ifstream ftgt(target_file, std::ios::binary);
    X = new float[NUM_SAMPLES * INPUT_SIZE];
    Y = new float[NUM_SAMPLES];
    fin.read(reinterpret_cast<char*>(X), sizeof(float) * NUM_SAMPLES * INPUT_SIZE);
    ftgt.read(reinterpret_cast<char*>(Y), sizeof(float) * NUM_SAMPLES);
    fin.close();
    ftgt.close();
}

int main() {
    float *h_X, *h_Y, *h_OUT;
    load_data("input.bin", "target.bin", h_X, h_Y);
    h_OUT = new float[NUM_SAMPLES];

    float *d_X, *d_Y, *d_W1, *d_B1, *d_A1, *d_W2, *d_B2, *d_OUT;
    cudaMalloc(&d_X, sizeof(float) * INPUT_SIZE * NUM_SAMPLES);
    cudaMalloc(&d_Y, sizeof(float) * NUM_SAMPLES);
    cudaMalloc(&d_W1, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE);
    cudaMalloc(&d_B1, sizeof(float) * HIDDEN_SIZE);
    cudaMalloc(&d_A1, sizeof(float) * NUM_SAMPLES * HIDDEN_SIZE);
    cudaMalloc(&d_W2, sizeof(float) * HIDDEN_SIZE);
    cudaMalloc(&d_B2, sizeof(float));
    cudaMalloc(&d_OUT, sizeof(float) * NUM_SAMPLES);

    float* W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float* B1 = new float[HIDDEN_SIZE];
    float* W2 = new float[HIDDEN_SIZE];
    float B2 = 0;

    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) W1[i] = (rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        B1[i] = 0.0f;
        W2[i] = (rand() / (float)RAND_MAX) - 0.5f;
    }

    cudaMemcpy(d_X, h_X, sizeof(float) * INPUT_SIZE * NUM_SAMPLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, sizeof(float) * NUM_SAMPLES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, sizeof(float) * INPUT_SIZE * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, B1, sizeof(float) * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, sizeof(float) * HIDDEN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, &B2, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (NUM_SAMPLES + blockSize - 1) / blockSize;

    for (int e = 0; e < EPOCHS; e++) {
        forward<<<gridSize, blockSize>>>(d_X, d_W1, d_B1, d_A1, d_W2, d_B2, d_OUT, NUM_SAMPLES);
        cudaDeviceSynchronize();

        backward<<<gridSize, blockSize>>>(d_X, d_Y, d_W1, d_B1, d_A1, d_W2, d_B2, d_OUT, NUM_SAMPLES);
        cudaDeviceSynchronize();
        std::cout << "Epoch " << e << " done\n";
    }

    cudaMemcpy(h_OUT, d_OUT, sizeof(float) * NUM_SAMPLES, cudaMemcpyDeviceToHost);
    std::ofstream fout("output.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(h_OUT), sizeof(float) * NUM_SAMPLES);
    fout.close();

    std::cout << "Saved output.bin with predictions.\n";

    delete[] h_X; delete[] h_Y; delete[] h_OUT;
    delete[] W1; delete[] B1; delete[] W2;
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W1); cudaFree(d_B1);
    cudaFree(d_A1); cudaFree(d_W2); cudaFree(d_B2); cudaFree(d_OUT);
    return 0;
}