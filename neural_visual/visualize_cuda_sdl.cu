#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <windows.h>

#define N 100
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 100

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

__global__ void forwardPassKernel(const float* input, const float* weights, float* output, float bias) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        float z = input[i] * weights[i] + bias;
        output[i] = fmaxf(z, 0.0f);
    }
}

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << "❌ CUDA Error: " << message << " — " << cudaGetErrorString(result) << "\n";
        exit(1);
    }
}

// 🖼️ Visualize with SDL2
void visualizeWithSDL(const float* output) {
    SDL_Window* window = SDL_CreateWindow("CUDA Output Visualization",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    bool running = true;
    SDL_Event event;

    int barWidth = WINDOW_WIDTH / N;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (int i = 0; i < N; ++i) {
            float val = output[i];
            val = fminf(fmaxf(val, 0.0f), 1.0f); // Clamp 0.0–1.0
            Uint8 r = static_cast<Uint8>(val * 255);
            Uint8 g = static_cast<Uint8>(255 - r);
            Uint8 b = 100;

            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            SDL_Rect rect = { i * barWidth, 0, barWidth - 1, WINDOW_HEIGHT };
            SDL_RenderFillRect(renderer, &rect);
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);  // Initialize SDL2

    float *h_input = new float[N];
    float *h_weights = new float[N];
    float *h_output = new float[N];
    float bias = 0.1f;

    loadInput("data.txt", h_input);
    for (int i = 0; i < N; ++i)
        h_weights[i] = 0.5f;

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

    // ✅ Visualize with SDL
    visualizeWithSDL(h_output);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;

    SDL_Quit();  // Clean up SDL2
    return 0;
}
