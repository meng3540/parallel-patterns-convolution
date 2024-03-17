Our basic algorithm is a program that performs 2D convolution from global memory. It takes an input matrix representing the data to be convolved, while the kernel defines the convolution operation's weights. 
The actual convolution is executed by launching the CUDA kernel, convolution2DKernel. 
It iterates over each element of the output matrix and computes the convolution result by summing the element-wise multiplication of the input matrix with the corresponding kernel weights.
