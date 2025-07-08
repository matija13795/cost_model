"""
Convert the original dataset so that each sample contains pre-computed
post-order computation indices instead of the raw tree structure.

Usage
-----
python convert_to_postorder_dataset.py \
       --in  /path/to/original_dataset.pt \
       --out /path/to/converted_dataset.pt
"""

import argparse
import torch
from tqdm import tqdm
from pathlib import Path

# ----------------------------------------------------------------------
# utility: compute post-order indices for one subtree
# ----------------------------------------------------------------------
def get_post_order_indices(node):
    indices = []
    for child in node.get("child_list", []):
        indices.extend(get_post_order_indices(child))
    indices.extend(node.get("computations_indices", []))
    return indices

# ----------------------------------------------------------------------
def convert(dataset):
    """Return a new list of samples in the desired 5-tuple format."""
    converted = []
    for data_point in tqdm(dataset, desc="converting"):
        sample, label = data_point
        (tree,
         comps_first,
         comps_vectors,
         comps_third,
         _loops_tensor,  # unused
         expr_tree) = sample

        post_order = []
        for root in tree["roots"]:
            post_order.extend(get_post_order_indices(root))
        post_order_tensor = torch.tensor(post_order, dtype=torch.long)

        new_sample = (
            post_order_tensor,  # <-- replaces tree
            comps_first,
            comps_vectors,
            comps_third,
            expr_tree,
        )
        converted.append((new_sample, label))
    return converted

def main(in_path: Path, out_path: Path):
    print(f"Loading dataset: {in_path}")
    data = torch.load(in_path)
    new_data = convert(data)
    torch.save(new_data, out_path)
    print(f"Saved {len(new_data)} samples → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  type=Path, required=True, dest="in_path", help="original dataset (.pt)")
    ap.add_argument("--out", type=Path, required=True, dest="out_path", help="output dataset (.pt)")
    args = ap.parse_args()
    main(args.in_path, args.out_path)