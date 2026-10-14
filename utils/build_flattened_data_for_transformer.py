import argparse
import torch
from tqdm import tqdm
from pathlib import Path

def build_seq_map(tree):
    sequence = [] # List of (type, index, depth, segment)
    comp_count = 1

    def dfs(node, depth):
        nonlocal comp_count
        # Type 0: L_START
        sequence.append((0, node['loop_index'].item(), depth, 0))
        
        if node['has_comps']:
            for c_idx in node['computations_indices']:
                c_val = c_idx.item()
                # Type 1: Domain
                sequence.append((1, c_val, depth + 1, comp_count))
                # Type 2: Transf
                for t_sub in range(4):
                    sequence.append((2, c_val * 4 + t_sub, depth + 1, comp_count))
                # Type 3: Access
                sequence.append((3, c_val, depth + 1, comp_count))
                # Type 4: Ops
                for op_sub in range(66):
                    sequence.append((4, c_val * 100 + op_sub, depth + 1, comp_count))
                comp_count += 1
        
        for child in node['child_list']:
            dfs(child, depth + 1)
        
        # Type 5: L_END
        sequence.append((5, 0, depth, 0))

    for root in tree['roots']:
        dfs(root, 0)
    return sequence

def convert(dataset, max_seq_len):
    converted = []
    dropped = 0
    for data_point in tqdm(dataset, desc="Flattening"):
        sample, label = data_point
        tree, dom, trans, acc, loops, ops = sample

        # 1. Build Map
        raw_map = build_seq_map(tree)
        
        # 2. Filter programs that are too long
        if len(raw_map) + 1 > max_seq_len:
            dropped += 1
            continue

        # 3. Prepare Tensors
        types = torch.zeros(max_seq_len, dtype=torch.long)
        indices = torch.zeros(max_seq_len, dtype=torch.long)
        depths = torch.zeros(max_seq_len, dtype=torch.long)
        segments = torch.zeros(max_seq_len, dtype=torch.long)
        mask = torch.ones(max_seq_len, dtype=torch.bool) # True = Masked

        # [CLS] Token at pos 0
        types[0], indices[0], depths[0], segments[0], mask[0] = 6, 0, 0, 0, False

        for i, (t, idx, d, s) in enumerate(raw_map):
            pos = i + 1
            types[pos] = t
            indices[pos] = idx
            depths[pos] = d
            segments[pos] = s
            mask[pos] = False

        new_sample = (dom, trans, acc, loops, ops, types, indices, depths, segments, mask)
        converted.append((new_sample, label))
        
    print(f"Dropped {dropped} samples (> {max_seq_len} tokens)")
    return converted

def main(in_path: Path, out_path: Path, max_seq_len: int):
    print(f"Loading: {in_path}")
    data = torch.load(in_path, map_location="cpu")
    new_data = convert(data, max_seq_len)
    torch.save(new_data, out_path)
    print(f"Saved {len(new_data)} samples -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", type=Path, required=True, dest="in_path")
    ap.add_argument("--out", type=Path, required=True, dest="out_path")
    ap.add_argument("--max_len", type=int, default=669, help="Maximum sequence length")
    
    args = ap.parse_args()
    main(args.in_path, args.out_path, args.max_len)