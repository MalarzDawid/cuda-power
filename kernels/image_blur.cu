#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

__global__ void image_blur_kernel(unsigned char* in, unsigned char* out, int step, int w, int h) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int ch = threadIdx.z;

    int baseOffset = ch * w * h;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-step; blurRow < step + 1; ++blurRow) {
            for (int blurCol=-step; blurCol < step + 1; ++blurCol) {
                int currRow = row + blurRow;
                int currCol = col + blurCol;

                if (currRow >= 0 && currRow < h && currCol >= 0 && currCol < w) {
                    pixVal += in[baseOffset + currRow * w + currCol];
                    ++pixels;
                }
            }
        }

        out[baseOffset + row * w + col] = (unsigned char)(pixVal/pixels);
    }
    
}

torch::Tensor blur(torch::Tensor input, int step) {
    CHECK_INPUT(input);
    int ch = input.size(0);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty_like(input);

    dim3 dimGrid(cdiv(w, 16), cdiv(h, 16), 1);
    dim3 dimBlock(16, 16, ch);

    image_blur_kernel<<<dimGrid, dimBlock>>> (input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), step, w, h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;    
}