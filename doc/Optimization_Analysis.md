If you take a look at globalMem_Nsight_1 and sharedMem_Nsight_2 you will see the memory throughput.
This is the main metric we used to see how effcient our optimization was. We used shared memory(tiling) as out enhancement.
What is memory throuput ? Memory thoroughput is the speed in which data is transferred between the GPU and the CPU.
This is good for convolution because at higher levels for example image processing or convolution neural networks. Sometimes large matrices are being convoleved a higher memory throughput allows for this process to be completed faster.
For global memory the memory throughput for the 2D convolution was 200 mb/s compared to the enhancement made with tiling which had a memoty thoughput of 800 mb/s.  
