import argparse
import glob
import os
import sys
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan .pt dataset files and report the maximum number of comps."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--patterns",
        nargs="+",
        metavar="GLOB",
        help="One or more shell glob patterns (quote them!)",
    )
    group.add_argument(
        "--file",
        metavar="PATH",
        help="Single .pt file to analyse (skip globbing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build the file list
    if args.file:
        if not os.path.isfile(args.file):
            sys.exit(f"ERROR: {args.file} does not exist")
        file_list = [args.file]
    else:  # patterns provided
        file_list = []
        for pat in args.patterns:
            file_list.extend(glob.glob(pat))
        if not file_list:
            sys.exit(f"ERROR: no files matched patterns: {args.patterns}")

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
