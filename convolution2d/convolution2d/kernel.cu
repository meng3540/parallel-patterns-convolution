#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5 // Size of input array
#define M 3  // Size of convolution kernel
#define BLOCK_SIZE 16

// Kernel function using shared memory (tiling)
#define TILE_SIZE 16


__global__
void convolution2DKernelshared(const float* input, const float* kernel, float* output,
    int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight) {
    __shared__ float inputTile[TILE_SIZE + 2][TILE_SIZE + 2]; // Input tile in shared memory
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;


    // Indexing for shared memory
    int sharedRow = localRow + 1;
    int sharedCol = localCol + 1;


    // Load data into shared memory
    if (row < inputHeight && col < inputWidth) {
        inputTile[sharedRow][sharedCol] = input[row * inputWidth + col];


        // Load halo elements
        if (localRow == 0) {
            if (row > 0)
                inputTile[sharedRow - 1][sharedCol] = input[(row - 1) * inputWidth + col];
            else
                inputTile[sharedRow - 1][sharedCol] = 0.0f;
        }


        if (localRow == blockDim.y - 1) {
            if (row < inputHeight - 1)
                inputTile[sharedRow + 1][sharedCol] = input[(row + 1) * inputWidth + col];
            else
                inputTile[sharedRow + 1][sharedCol] = 0.0f;
        }


        if (localCol == 0) {
            if (col > 0)
                inputTile[sharedRow][sharedCol - 1] = input[row * inputWidth + col - 1];
            else
                inputTile[sharedRow][sharedCol - 1] = 0.0f;
        }


        if (localCol == blockDim.x - 1) {
            if (col < inputWidth - 1)
                inputTile[sharedRow][sharedCol + 1] = input[row * inputWidth + col + 1];
            else
                inputTile[sharedRow][sharedCol + 1] = 0.0f;
        }


        __syncthreads();


        float result = 0.0f;


        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                result += inputTile[sharedRow - kernelHeight / 2 + i][sharedCol - kernelWidth / 2 + j] * kernel[i * kernelWidth + j];
            }
        }


        output[row * inputWidth + col] = result;
    }
}


// kernel function
__global__
void convolution2DKernel(const float* input, const float* kernel, float* output,
    int inputWidth, int inputHeight,
    int kernelWidth, int kernelHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < inputWidth && row < inputHeight) {
        int halfKernelWidth = kernelWidth / 2;
        int halfKernelHeight = kernelHeight / 2;

        float result = 0.0f;

        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row - halfKernelHeight + i;
                int inputCol = col - halfKernelWidth + j;

                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                    result += input[inputRow * inputWidth + inputCol] * kernel[i * kernelWidth + j];
                }
            }
        }

        output[row * inputWidth + col] = result;
    }
}

int main() {
    const int width = N;
    const int height = N;
    const int kernelSize = M;

    // Input matrix
    float input[N][N] = {
        {3, 3, 2, 1, 0},
        {0, 0, 1, 3, 1},
        {3, 1, 2, 2, 3},
        {2, 0, 0, 2, 2},
        {2, 0, 0, 0, 1}
    };

    // Kernel
    float kernel[M][M] = {
        {0, 1, 2},
        {2, 2, 0},
        {0, 1, 2}
    };

    float* h_input, * h_output, * h_kernel;
    float* d_input, * d_output, * d_kernel;

    // Allocate memory on host
    h_input = (float*)malloc(width * height * sizeof(float));
    h_output = (float*)malloc(width * height * sizeof(float));
    h_kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));

    // Initialize input array
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            h_input[i * width + j] = input[i][j];
        }
    }

    // Initialize kernel
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            h_kernel[i * kernelSize + j] = kernel[i][j];
        }
    }

    // Allocate memory on device
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    // Copy input array and kernel to device
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch kernel
    convolution2DKernel << <grid, block >> > (d_input, d_kernel, d_output, width, height, kernelSize, kernelSize);
    //convolution2DKernelshared << <grid, block >> > (d_input, d_kernel, d_output, width, height, kernelSize, kernelSize);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    printf("Input:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f ", h_input[i * width + j]);
        }
        printf("\n");
    }

    printf("\nKernel:\n");
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            printf("%.2f ", h_kernel[i * kernelSize + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f ", h_output[i * width + j]);
        }
        printf("\n");
    }

    // Free memory
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
