#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void vector_add_kernel(float* vec_a, float* vec_b, float* vec_c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vec_c[i] = vec_a[i] + vec_b[i];
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}



torch::Tensor vector_addition(torch::Tensor vec_a, torch::Tensor vec_b) {
    assert(vec_a.device().type() == torch::kCUDA);
    assert(vec_b.device().type() == torch::kCUDA);

    const auto vec_size = vec_a.size(0);
    torch::Tensor output = torch::empty({vec_size}, vec_a.options());
    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(vec_size, numThreads);

    vector_add_kernel<<<numBlocks, numThreads>>>(
        vec_a.data_ptr<float>(), 
        vec_b.data_ptr<float>(),
        output.data_ptr<float>(),
        vec_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}