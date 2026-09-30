import argparse
import os
import sys
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect dataset structure and preview contents."
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--num", type=int, default=10)
    return parser.parse_args()


def preview_tensor(tensor, max_elements=10):
    if tensor.numel() <= max_elements:
        return tensor
    return tensor.flatten()[:max_elements]


def describe_item(name, item, indent=2):
    prefix = " " * indent

    if isinstance(item, torch.Tensor):
        print(f"{prefix}{name}: shape={tuple(item.shape)}, dtype={item.dtype}")
        print(f"{prefix}  preview: {preview_tensor(item)}")

    elif isinstance(item, dict):
        print(f"{prefix}{name}: dict with keys={list(item.keys())}")
        for k, v in item.items():
            describe_item(f"{name}['{k}']", v, indent + 2)

    elif isinstance(item, (list, tuple)):
        print(f"{prefix}{name}: {type(item)} (len={len(item)})")
        for i, v in enumerate(item):
            describe_item(f"{name}[{i}]", v, indent + 2)

    else:
        print(f"{prefix}{name}: type={type(item)}, value={item}")

def calculate_stats(lengths):
    print(f"Max Sequence Length: {max(lengths)}")
    print(f"Min Sequence Length: {min(lengths)}")
    print(f"Mean Sequence Length: {np.mean(lengths):.2f}")
    print(f"95th Percentile: {np.percentile(lengths, 95)}")

def get_seq_lengths(data):
    all_lengths = []
    
    for i in range(len(data)):
        sample, _ = data[i]
        
        # Extract metadata from shapes
        # sample[1] is (num_schedules, num_comps, 33)
        num_comps = sample[1].shape[1]
        
        # sample[4] is (num_schedules, num_loops, 8)
        num_loops = sample[4].shape[1]
        
        # The sum we discussed:
        # 2 tokens per loop (Start/End)
        # 72 tokens per computation (1 domain + 4 transforms + 1 access + 66 ops)
        length = (num_loops * 2) + (num_comps * 72)
        all_lengths.append(length)

        expr_seq_len = sample[5].shape[2]
        if expr_seq_len != 66:
            print("Houston, we have a problem.")
        
    return all_lengths

def check_loop_indexing(sample_dict, loop_tensor):
    indices = []
    
    def walk_tree(node):
        # Collect the index for the current loop
        indices.append(node['loop_index'].item())
        # Recurse through children
        for child in node['child_list']:
            walk_tree(child)
            
    # Start walking from the roots
    for root in sample_dict['roots']:
        walk_tree(root)
        
    max_idx = max(indices)
    num_loops = loop_tensor.shape[1]
    
    print(f"Max loop index found in tree: {max_idx}")
    print(f"Loop tensor second dimension: {num_loops}")
    
    if max_idx == num_loops - 1:
        print("Indexing check: PASSED (Indices match tensor rows).")
    else:
        print("Indexing check: FAILED (Tree indices do not align with tensor).")


def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        sys.exit(f"ERROR: {args.file} does not exist")

    print(f"Loading {args.file} ...")
    data = torch.load(args.file, map_location="cpu")

    print(f"\nTotal samples: {len(data)}")

    # --- First sample: deep inspection ---
    print("\n===== FULL INSPECTION OF SAMPLE 0 =====")
    sample, label = data[0]

    describe_item("sample", sample)
    describe_item("label", label)

    # --- Remaining samples: shapes only ---
    print(f"\n===== NEXT {min(args.num - 1, len(data) - 1)} SAMPLES (SHAPES ONLY) =====")

    for i, (sample, label) in enumerate(data[1:args.num], start=1):
        print(f"\n--- Sample {i} ---")
        check_loop_indexing(sample[0], sample[4])
        if isinstance(sample, (list, tuple)):
            for idx, item in enumerate(sample):
                if isinstance(item, torch.Tensor):
                    print(f"  sample[{idx}]: shape={tuple(item.shape)}")
                else:
                    print(f"  sample[{idx}]: type={type(item)}")
        else:
            print(f"  sample: type={type(sample)}")

        if isinstance(label, torch.Tensor):
            print(f"  label: shape={tuple(label.shape)}")
        else:
            print(f"  label: type={type(label)}")

    print()
    print()

    lengths = get_seq_lengths(data)
    calculate_stats(lengths)

if __name__ == "__main__":
    main()