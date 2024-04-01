#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {return (a + b - 1) / b;}


__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] =  0.21f * x[idx] + 0.72f * x[idx + n] + 0.07f * x[idx + 2*n];
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    int threads = 256;
    
    auto output = torch::empty({h, w}, input.options());
    rgb_to_grayscale_kernel<<<cdiv(w * h, threads), threads>>> (input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), h*w);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}




