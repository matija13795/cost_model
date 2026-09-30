"""
Randomly sample whole batches from a list-of-batches .pt file
until >= N examples are collected, then save the slice.

Usage:
    python slice_pt_dataset_wholebatches.py big_val_5M.pt val_subset_250k.pt
"""

import argparse, random, torch, os, sys

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("in_pt")
    p.add_argument("out_pt")
    p.add_argument("--n", type=int, default=250_000,
                   help="minimum #examples to collect (default 250,000)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for repeatability")
    return p.parse_args()

def batch_len(batch):
    # assumes each batch is (inputs, labels)
    return batch[1].shape[0]

def main():
    args = parse()
    random.seed(args.seed)

    print(f"Loading {args.in_pt} …")
    data = torch.load(args.in_pt, map_location="cpu")
    if not isinstance(data, list):
        sys.exit("✗ Expected a list of (inputs, labels) batches.")

    random.shuffle(data)
    collected, out = 0, []
    for b in data:
        out.append(b)
        collected += batch_len(b)
        if collected >= args.n:
            break

    print(f"Collected {collected:,} examples in {len(out)} batches.")
    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    torch.save(out, args.out_pt)
    print(f"✓ Saved to {args.out_pt}")

if __name__ == "__main__":
    main()