import os
import glob
import torch

# -----------------------------------------
# Paths to scan – add/remove patterns here
# -----------------------------------------
PATTERNS = [
    "batched/train/*_GPU.pt",   # training batches
    "batched/valid/*_GPU.pt",     # validation batches
    # If there are *_CPU.pt files, add another pattern:
    # "batched/**/**/*_CPU.pt",
]

def main():
    file_list = []
    for pat in PATTERNS:
        file_list.extend(glob.glob(pat))

    if not file_list:
        raise FileNotFoundError(
            "No .pt files matched the patterns: {}".format(PATTERNS)
        )

    max_comps = 0
    examined_batches = 0
    for pt_path in sorted(file_list):
        print(f"Scanning {pt_path} ...")
        batches = torch.load(pt_path, map_location="cpu")

        for sample, _ in batches:
            comps_tensor_first_part = sample[1]  # index 1 holds the tensor
            num_comps = comps_tensor_first_part.shape[1]
            max_comps = max(max_comps, num_comps)
            examined_batches += 1

    print("\nFinished.")
    print(f"Examined {examined_batches} batches across {len(file_list)} files.")
    print(f"Maximum number of computations found: {max_comps}")

if __name__ == "__main__":
    main()