
import torch


def main() -> None:
    print(f"Is Cuda Available: {torch.cuda.is_available()}")
    print(f"Number of CUDA-capable devices: {torch.cuda.device_count()}")
    current_cuda_device : int = torch.cuda.current_device()
    print(f"Current CUDA device: {current_cuda_device}")
    print(f"Is Cuda Available: {torch.cuda.get_device_name(current_cuda_device)}")


if __name__ == "__main__":
    main()
