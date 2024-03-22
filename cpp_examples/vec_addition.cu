#include <iostream>
#include <cassert>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1 << 16;

    // Host vector pointers
    int *h_a, *h_b, *h_c;
    size_t bytes = sizeof(int) * N;

    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    // Initizalize vectors A & B
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Alocate memory on the device
    int *d_a, *d_b, *d_c;

    // Alocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy data (CPU -> GPU)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 1<<10 => 1024
    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS; // for example: (65536 + 1024 - 1) / 1024 => 64.99 ??

    // Run kernel
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    
    // Copy output to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < N; i++) {
        assert(h_c[i] == h_a[i] + h_b[i]);
    }

    // Free pinned memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Everything is ok" << std::endl;
}