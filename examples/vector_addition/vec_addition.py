from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

SIZE = 1000


def compile_ext(cuda_source: str, cpp_headers: str, ext_name: str, func: list):
    cuda_source = Path(cuda_source).read_text()

    ext = load_inline(
        name=ext_name,
        cpp_sources=cpp_headers,
        cuda_sources=cuda_source,
        functions=func,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )
    return ext


def tensor_details(tensor: torch.Tensor, name: str, head: int = 10):
    print("*" * 50)
    print(f"Tensor {name}")
    print(f"\t Shape: {tensor.shape}")
    print(f"\t Dtype: {tensor.dtype}")
    print(f"\t Device: {tensor.device}")
    print(f"Sample:\n {tensor[:head]}\n")


def main():
    # Create input data
    v_a = torch.arange(0, SIZE, dtype=torch.float32, device="cuda")
    v_b = torch.arange(0, SIZE, dtype=torch.float32, device="cuda")

    # Print details
    tensor_details(v_a, "A")
    tensor_details(v_b, "B")

    # Set up cuda & cpp source
    cuda_source = "vec_addition_kernel.cu"
    cpp_source = (
        "torch::Tensor vector_addition(torch::Tensor vec_a, torch::Tensor vec_b);"
    )

    # Compile extension
    ext = compile_ext(cuda_source, cpp_source, "vector_ext", ["vector_addition"])

    # Use extension
    output = ext.vector_addition(v_a, v_b)
    tensor_details(output, "Output")


if __name__ == "__main__":
    main()
