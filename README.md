# CUDA Convolution

## Summary

Convolution is a mathematical operation frequently used in signal and image processing. In parallel programming, convolution can be parallelized to enhance performance, especially when dealing with large datasets. Here's a simplified example:

Let's consider a 1D convolution operation on an array of data “A” with a kernel “K”. The convolution operation can be represented as:

    C[i] = ∑(A[i+k]*K[k])

where “C” is the resulting convolution array, “A[i+k]” represents the element of the input array “A” at index “i+k”, and “K[k]” represents the element of the kernel “K” at index “k”.

To parallelize this operation, you can distribute the computation across multiple threads or processors. For example, you can divide the input array “A” into smaller segments and assign each segment to a separate thread. Each thread computes a portion of the convolution operation on its assigned segment concurrently. Finally, the results from all threads are combined to produce the final output array “C”.

In essence, parallel convolution involves dividing the input data and distributing the workload across multiple processing units to exploit parallelism and improve performance.

When employing padding same, the padding is applied in such a way that the output size matches the input size. This ensures that the convolution operation does not reduce the dimensions of the input data.

### Padding same
![Padding same gif](res/padding_same.gif)

## Definition

**Parallel convolution** is a technique used in parallel programming to compute convolution operations concurrently across multiple processing units, such as CPU cores or GPUs. By distributing the computation workload, parallel convolution aims to exploit parallelism and accelerate the processing of large datasets, enhancing overall performance.

## Basic Algorithm (Padding Same)
 
The function iterates over each element of the output array, computing the convolution result for each element based on the input data and the kernel.

Boundary conditions are checked to ensure that the convolution operation stays within the bounds of the input data.

The computed convolution result is stored in the corresponding element of the output array.

```cuda
__global__ void convolution2DKernel(const float* input, const float* kernel, float* output,
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
```
## Visual Studio Project
The Visual Studio project for this CUDA convolution implementation can be found [here](convolution2d/convolution2d.sln). The CUDA C code is located in the [kernel.cu](convolution2d/convolution2d/kernel.cu).
