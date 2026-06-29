"""Print CUDA availability, GPU VRAM, and PyTorch version."""

import torch


def format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def main() -> None:
    print("=" * 50)
    print("scheduling_research — environment check")
    print("=" * 50)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"GPU count       : {torch.cuda.device_count()}")

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"\nGPU {idx}: {props.name}")
            print(f"  Total VRAM    : {format_bytes(props.total_memory)}")
            print(f"  Compute cap.  : {props.major}.{props.minor}")

        free, total = torch.cuda.mem_get_info(0)
        print(f"\nGPU 0 current usage:")
        print(f"  Free VRAM     : {format_bytes(free)}")
        print(f"  Allocated     : {format_bytes(torch.cuda.memory_allocated(0))}")
        print(f"  Reserved      : {format_bytes(torch.cuda.memory_reserved(0))}")
    else:
        print("\nNo CUDA GPU detected.")
        print("Check NVIDIA driver and reinstall PyTorch with CUDA support.")

    print("=" * 50)


if __name__ == "__main__":
    main()
