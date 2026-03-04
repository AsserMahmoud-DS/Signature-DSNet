"""
Generate train/test pair files for CEDAR and GDPS datasets.
Creates gray_train.txt and gray_test.txt with format: refer_path test_path label
"""

import os
import json
from pathlib import Path
from itertools import combinations, product
from tqdm import tqdm


def generate_cedar_pairs(data_root, train_ratio=0.73, random_state=42):
    """
    Generate train/test pair files for CEDAR dataset.
    
    CEDAR structure:
    - full_org/: original_1_1.png, original_1_2.png, ..., original_55_24.png
    - full_forg/: forgeries_1_1.png, ..., forgeries_55_24.png
    
    Each writer has 24 genuine and 24 forged signatures.
    Split: ~40 writers (552 pairs) for train, ~15 writers (207 pairs) for test.
    
    Args:
        data_root: Path to CEDAR directory
        train_ratio: Proportion of writers for training (40/55 ≈ 0.73)
        random_state: Random seed for reproducibility
    """
    
    full_org_path = os.path.join(data_root, "full_org")
    full_forg_path = os.path.join(data_root, "full_forg")
    
    # Parse genuine signatures
    genuine_sigs = {}  # {writer_id: [list of (filename, rel_path)]}
    for img_file in os.listdir(full_org_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            parts = img_file.split('_')
            writer_id = int(parts[1])
            
            if writer_id not in genuine_sigs:
                genuine_sigs[writer_id] = []
            
            rel_path = os.path.join("full_org", img_file)
            genuine_sigs[writer_id].append((img_file, rel_path))
    
    # Parse forged signatures
    forged_sigs = {}  # {writer_id: [list of (filename, rel_path)]}
    for img_file in os.listdir(full_forg_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            parts = img_file.split('_')
            writer_id = int(parts[1])
            
            if writer_id not in forged_sigs:
                forged_sigs[writer_id] = []
            
            rel_path = os.path.join("full_forg", img_file)
            forged_sigs[writer_id].append((img_file, rel_path))
    
    assert len(genuine_sigs) == 55, f"Expected 55 writers, got {len(genuine_sigs)}"
    assert len(forged_sigs) == 55, f"Expected 55 writers, got {len(forged_sigs)}"
    
    all_writers = sorted(genuine_sigs.keys())
    num_train = int(len(all_writers) * train_ratio)
    
    # Fixed split for reproducibility
    import random
    random.seed(random_state)
    train_writers = sorted(random.sample(all_writers, num_train))
    test_writers = sorted(set(all_writers) - set(train_writers))
    
    print(f"CEDAR: {len(train_writers)} train writers, {len(test_writers)} test writers")
    
    # Generate positive and negative pairs
    def generate_pairs_for_split(writers, is_train=True):
        pairs = []
        
        for writer_id in tqdm(writers, desc=f"generating {'train' if is_train else 'test'} pairs"):
            gen_list = genuine_sigs[writer_id]
            forg_list = forged_sigs[writer_id]
            
            # Positive pairs: genuine-to-genuine from same writer (24 * 23 / 2 = 276 unique pairs)
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    refer_path = gen_list[i][1]
                    test_path = gen_list[j][1]
                    pairs.append((refer_path, test_path, 1)) # 1 for genuine
            
            # Negative pairs: genuine-to-forged from same writer (24 * 24 = 576 pairs)
            for i in range(len(gen_list)):
                for j in range(len(forg_list)):
                    refer_path = gen_list[i][1]
                    test_path = forg_list[j][1]
                    pairs.append((refer_path, test_path, 0)) # 0 for forged
        
        return pairs
    
    train_pairs = generate_pairs_for_split(train_writers, is_train=True)
    test_pairs = generate_pairs_for_split(test_writers, is_train=False)
    
    # Write to files (tab-separated to handle filenames with spaces)
    train_txt = os.path.join(data_root, "gray_train.txt")
    test_txt = os.path.join(data_root, "gray_test.txt")
    
    with open(train_txt, 'w') as f:
        for refer, test, label in train_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")
    
    with open(test_txt, 'w') as f:
        for refer, test, label in test_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")
    
    print(f"✅ Generated {train_txt}: {len(train_pairs)} pairs")
    print(f"✅ Generated {test_txt}: {len(test_pairs)} pairs")
    
    return {
        'train_file': train_txt,
        'test_file': test_txt,
        'num_train_pairs': len(train_pairs),
        'num_test_pairs': len(test_pairs),
        'train_writers': train_writers,
        'test_writers': test_writers
    }


def generate_gdps_pairs(data_root, train_ratio=0.7, random_state=42):
    """
    Generate train/test pair files for GDPS dataset.
    
    GDPS structure:
    - train/: writer folders (1/ to N/)
      - Each writer folder: genuine/ and forge/
      - Images are .jpg files
    - test/: writer folders (1/ to M/)
      - Each writer folder: genuine/ and forge/
      - Images are .jpg files
    
    Pair file paths are relative to GDPS_ROOT and include "train/" or "test/" prefix:
      e.g. train/1/genuine/c001001.jpg train/1/genuine/c001002.jpg 1
      or   test/1/genuine/c001001.jpg test/1/genuine/c001002.jpg 1
    
    Args:
        data_root: Path to GDPS root directory
        train_ratio: Proportion of writers for training (only used if both train/ and test/ exist)
        random_state: Random seed for reproducibility
    """
    
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    
    # Helper function to load signatures from a dataset split (train or test)
    def load_signatures_from_split(split_path, split_name):
        """Load signatures from train/ or test/ folder"""
        genuine_sigs = {}  # {writer_id: [rel_paths]}
        forged_sigs = {}   # {writer_id: [rel_paths]}
        
        if not os.path.isdir(split_path):
            return genuine_sigs, forged_sigs
        
        for writer_dir in sorted(os.listdir(split_path)):
            writer_path = os.path.join(split_path, writer_dir)
            if not os.path.isdir(writer_path):
                continue
            
            try:
                writer_id = int(writer_dir)
            except ValueError:
                continue
            
            genuine_sigs[writer_id] = []
            forged_sigs[writer_id] = []
            
            # Scan genuine and forge folders
            genuine_folder = os.path.join(writer_path, "genuine")
            forge_folder = os.path.join(writer_path, "forge")
            
            if os.path.isdir(genuine_folder):
                for img_file in os.listdir(genuine_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.join(split_name, writer_dir, "genuine", img_file)
                        genuine_sigs[writer_id].append(rel_path)
            
            if os.path.isdir(forge_folder):
                for img_file in os.listdir(forge_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.join(split_name, writer_dir, "forge", img_file)
                        forged_sigs[writer_id].append(rel_path)
        
        return genuine_sigs, forged_sigs
    
    # Load signatures from both train and test splits
    train_genuine, train_forged = load_signatures_from_split(train_path, "train")
    test_genuine, test_forged = load_signatures_from_split(test_path, "test")
    
    # If we have both splits, use them as-is; otherwise use all writers from whichever split exists
    if train_genuine and test_genuine:
        # Both splits exist - use them directly
        train_writers = sorted(train_genuine.keys())
        test_writers = sorted(test_genuine.keys())
        
        genuine_sigs_train = train_genuine
        forged_sigs_train = train_forged
        genuine_sigs_test = test_genuine
        forged_sigs_test = test_forged
    else:
        # Only one split exists (e.g., only test/) - split writers
        genuine_sigs = train_genuine if train_genuine else test_genuine
        forged_sigs = train_forged if train_forged else test_forged
        
        all_writers = sorted(genuine_sigs.keys())
        num_train = int(len(all_writers) * train_ratio)
        
        import random
        random.seed(random_state)
        train_writers = sorted(random.sample(all_writers, num_train))
        test_writers = sorted(set(all_writers) - set(train_writers))
        
        # For this case, data comes from the same source
        genuine_sigs_train = genuine_sigs
        forged_sigs_train = forged_sigs
        genuine_sigs_test = genuine_sigs
        forged_sigs_test = forged_sigs
    
    print(f"GDPS: {len(train_writers)} train writers, {len(test_writers)} test writers")
    
    # Generate positive and negative pairs
    def generate_pairs_for_split(writers, genuine_sigs, forged_sigs, is_train=True):
        pairs = []
        
        for writer_id in tqdm(writers, desc=f"generating {'train' if is_train else 'test'} pairs"):
            gen_list = genuine_sigs.get(writer_id, [])
            forg_list = forged_sigs.get(writer_id, [])
            
            if len(gen_list) < 2:
                continue  # Skip writers with < 2 genuine signatures
            
            # Positive pairs: genuine-to-genuine from same writer
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    pairs.append((gen_list[i], gen_list[j], 1))
            
            # Negative pairs: genuine-to-forged from same writer
            for i in range(len(gen_list)):
                for j in range(len(forg_list)):
                    pairs.append((gen_list[i], forg_list[j], 0))
        
        return pairs
    
    train_pairs = generate_pairs_for_split(train_writers, genuine_sigs_train, forged_sigs_train, is_train=True)
    test_pairs = generate_pairs_for_split(test_writers, genuine_sigs_test, forged_sigs_test, is_train=False)
    
    # Write pair files to GDPS_ROOT (tab-separated to handle filenames with spaces)
    train_txt = os.path.join(data_root, "gray_train.txt")
    test_txt = os.path.join(data_root, "gray_test.txt")
    
    with open(train_txt, 'w') as f:
        for refer, test, label in train_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")
    
    with open(test_txt, 'w') as f:
        for refer, test, label in test_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")
    
    print(f"✅ Generated {train_txt}: {len(train_pairs)} pairs")
    print(f"✅ Generated {test_txt}: {len(test_pairs)} pairs")
    
    return {
        'train_file': train_txt,
        'test_file': test_txt,
        'num_train_pairs': len(train_pairs),
        'num_test_pairs': len(test_pairs),
        'train_writers': train_writers,
        'test_writers': test_writers
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate train/test pair files for signature datasets")
    parser.add_argument('--cedar', type=str, help="Path to CEDAR dataset root")
    parser.add_argument('--gdps', type=str, help="Path to GDPS dataset root")
    
    args = parser.parse_args()
    
    if args.cedar:
        generate_cedar_pairs(args.cedar)
    
    if args.gdps:
        generate_gdps_pairs(args.gdps)
