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
        if img_file.endswith('.png'):
            parts = img_file.split('_')
            writer_id = int(parts[1])
            
            if writer_id not in genuine_sigs:
                genuine_sigs[writer_id] = []
            
            rel_path = os.path.join("full_org", img_file)
            genuine_sigs[writer_id].append((img_file, rel_path))
    
    # Parse forged signatures
    forged_sigs = {}  # {writer_id: [list of (filename, rel_path)]}
    for img_file in os.listdir(full_forg_path):
        if img_file.endswith('.png'):
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
    
    # Write to files
    train_txt = os.path.join(data_root, "gray_train.txt")
    test_txt = os.path.join(data_root, "gray_test.txt")
    
    with open(train_txt, 'w') as f:
        for refer, test, label in train_pairs:
            f.write(f"{refer} {test} {label}\n")
    
    with open(test_txt, 'w') as f:
        for refer, test, label in test_pairs:
            f.write(f"{refer} {test} {label}\n")
    
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
    - test/: writer folders (1/ to 150/)
      - Each writer folder: forge/ and genuine/
      - Files: c{userid}{instance}.png (genuine), cf{userid}{instance}.png (forged)
    
    Args:
        data_root: Path to GDPS root directory
        train_ratio: Proportion of writers for training
        random_state: Random seed for reproducibility
    """
    
    # Check if test folder exists
    test_path = os.path.join(data_root, "test")
    if not os.path.exists(test_path):
        test_path = data_root  # Fallback: assume data_root IS the test folder
    
    # Parse writers from directories
    genuine_sigs = {}  # {writer_id: [rel_paths]}
    forged_sigs = {}   # {writer_id: [rel_paths]}
    
    for writer_dir in sorted(os.listdir(test_path)):
        writer_path = os.path.join(test_path, writer_dir)
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
                if img_file.endswith('.png'):
                    if test_path == data_root:
                        rel_path = os.path.join(writer_dir, "genuine", img_file)
                    else:
                        rel_path = os.path.join("test", writer_dir, "genuine", img_file)
                    genuine_sigs[writer_id].append(rel_path)
        
        if os.path.isdir(forge_folder):
            for img_file in os.listdir(forge_folder):
                if img_file.endswith('.png'):
                    if test_path == data_root:
                        rel_path = os.path.join(writer_dir, "forge", img_file)
                    else:
                        rel_path = os.path.join("test", writer_dir, "forge", img_file)
                    forged_sigs[writer_id].append(rel_path)
    
    print(f"GDPS: Found {len(genuine_sigs)} writers")
    print(f"DEBUG: Sample genuine_sigs[2]: {genuine_sigs.get(2, [])} ({len(genuine_sigs.get(2, []))} items)")
    print(f"DEBUG: Sample forged_sigs[2]: {forged_sigs.get(2, [])} ({len(forged_sigs.get(2, []))} items)")
    
    # Filter writers with at least some signatures
    valid_writers = [w for w in genuine_sigs.keys() if len(genuine_sigs[w]) > 0 and len(forged_sigs[w]) > 0]
    print(f"GDPS: {len(valid_writers)} writers with both genuine and forged signatures")
    
    all_writers = sorted(valid_writers)
    num_train = int(len(all_writers) * train_ratio)
    
    import random
    random.seed(random_state)
    train_writers = sorted(random.sample(all_writers, num_train))
    test_writers = sorted(set(all_writers) - set(train_writers))
    
    print(f"GDPS: {len(train_writers)} train writers, {len(test_writers)} test writers")
    
    # Generate positive and negative pairs
    def generate_pairs_for_split(writers, is_train=True):
        pairs = []
        
        for writer_id in tqdm(writers, desc=f"generating {'train' if is_train else 'test'} pairs"):
            gen_list = genuine_sigs[writer_id]
            forg_list = forged_sigs[writer_id]
            
            if len(gen_list) < 2 and len(gen_list) < 1:
                continue  # Skip writers with no genuine signatures
            
            # Positive pairs: genuine-to-genuine from same writer
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    pairs.append((gen_list[i], gen_list[j], 1))
            
            # Negative pairs: genuine-to-forged from same writer
            for i in range(len(gen_list)):
                for j in range(len(forg_list)):
                    pairs.append((gen_list[i], forg_list[j], 0))
        
        return pairs
    
    train_pairs = generate_pairs_for_split(train_writers, is_train=True)
    test_pairs = generate_pairs_for_split(test_writers, is_train=False)
    
    # Write to files
    train_txt = os.path.join(data_root, "gray_train.txt")
    test_txt = os.path.join(data_root, "gray_test.txt")
    
    with open(train_txt, 'w') as f:
        for refer, test, label in train_pairs:
            f.write(f"{refer} {test} {label}\n")
    
    with open(test_txt, 'w') as f:
        for refer, test, label in test_pairs:
            f.write(f"{refer} {test} {label}\n")
    
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
