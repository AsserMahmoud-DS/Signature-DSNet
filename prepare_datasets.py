"""
Generate train/test pair files for CEDAR and GDPS datasets.
Creates gray_train.txt and gray_test.txt with format: refer_path test_path label
"""

import os
import json
import re
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
    - train/: FLAT structure
      - genuine/    (all genuine images: c*.jpg)
      - forge/      (all forged images: cf*.jpg)
    - test/: HIERARCHICAL structure
      - 1/, 2/, 3/, ... (subject folders)
        - Each subject folder:
          - genuine/  (c*.jpg)
          - forge/    (cf*.jpg)
    
    Pair file paths are relative to GDPS_ROOT:
      Train: train/genuine/c001001.jpg train/genuine/c001002.jpg 1
      Test:  test/1/genuine/c001001.jpg test/1/genuine/c001002.jpg 1
    
    Args:
        data_root: Path to GDPS root directory
        train_ratio: Not used (for compatibility)
        random_state: Random seed for reproducibility
    """
    
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    
    # Load training signatures (flat structure: train/genuine/ and train/forge/)
    def load_train_signatures():
        """Load train signatures from flat train/{genuine,forge}/ structure"""
        genuine_sigs = []
        forged_sigs = []
        
        genuine_folder = os.path.join(train_path, "genuine")
        forge_folder = os.path.join(train_path, "forge")
        
        if os.path.isdir(genuine_folder):
            for img_file in sorted(os.listdir(genuine_folder)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.join("train", "genuine", img_file)
                    genuine_sigs.append(rel_path)
        
        if os.path.isdir(forge_folder):
            for img_file in sorted(os.listdir(forge_folder)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rel_path = os.path.join("train", "forge", img_file)
                    forged_sigs.append(rel_path)
        
        return genuine_sigs, forged_sigs
    
    # Load test signatures (hierarchical structure: test/subject_id/{genuine,forge}/)
    def load_test_signatures():
        """Load test signatures from hierarchical test/subject_id/{genuine,forge}/ structure"""
        test_sigs = {}  # {subject_id: {'genuine': [...], 'forge': [...]}}
        
        if not os.path.isdir(test_path):
            return test_sigs
        
        for subject_dir in sorted(os.listdir(test_path)):
            subject_path = os.path.join(test_path, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            
            try:
                subject_id = int(subject_dir)
            except ValueError:
                continue
            
            test_sigs[subject_id] = {'genuine': [], 'forge': []}
            
            # Scan genuine folder
            genuine_folder = os.path.join(subject_path, "genuine")
            if os.path.isdir(genuine_folder):
                for img_file in sorted(os.listdir(genuine_folder)):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.join("test", subject_dir, "genuine", img_file)
                        test_sigs[subject_id]['genuine'].append(rel_path)
            
            # Scan forge folder
            forge_folder = os.path.join(subject_path, "forge")
            if os.path.isdir(forge_folder):
                for img_file in sorted(os.listdir(forge_folder)):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = os.path.join("test", subject_dir, "forge", img_file)
                        test_sigs[subject_id]['forge'].append(rel_path)
        
        return test_sigs
    
    # Load signatures
    train_genuine, train_forged = load_train_signatures()
    test_sigs = load_test_signatures()
    
    print(f"GDPS: {len(train_genuine)} train genuine, {len(train_forged)} train forged")
    print(f"GDPS: {len(test_sigs)} test subjects")
    
    # Generate training pairs (per-writer to avoid cross-writer pairs)
    def generate_train_pairs():
        """Generate train pairs by organizing signatures by writer ID from filename."""
        pairs = []
        train_by_writer = {}  # {writer_id: {'genuine': [...], 'forge': [...]}}
        
        # Extract writer ID from filename and organize by writer
        for sig_path in train_genuine:
            filename = os.path.basename(sig_path)
            # Format: c-032-11 (Copy).jpg → writer ID is 032
            match = re.match(r'c-(\d+)-', filename)
            if match:
                writer_id = match.group(1)
                if writer_id not in train_by_writer:
                    train_by_writer[writer_id] = {'genuine': [], 'forge': []}
                train_by_writer[writer_id]['genuine'].append(sig_path)
        
        for sig_path in train_forged:
            filename = os.path.basename(sig_path)
            # Format: cf-012-13 (Copy).jpg → writer ID is 012
            match = re.match(r'cf-(\d+)-', filename)
            if match:
                writer_id = match.group(1)
                if writer_id not in train_by_writer:
                    train_by_writer[writer_id] = {'genuine': [], 'forge': []}
                train_by_writer[writer_id]['forge'].append(sig_path)
        
        # Generate pairs per writer
        for writer_id in tqdm(sorted(train_by_writer.keys()), desc="generating train pairs (per-writer)"):
            gen_list = train_by_writer[writer_id]['genuine']
            forg_list = train_by_writer[writer_id]['forge']
            
            if len(gen_list) < 2:
                continue
            
            # Positive pairs: genuine-to-genuine from same writer
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    pairs.append((gen_list[i], gen_list[j], 1))
            
            # Negative pairs: genuine-to-forged from same writer
            for i in range(len(gen_list)):
                for j in range(len(forg_list)):
                    pairs.append((gen_list[i], forg_list[j], 0))
        
        return pairs
    
    # Generate test pairs (from hierarchical test structure)
    def generate_test_pairs():
        pairs = []
        
        for subject_id in tqdm(sorted(test_sigs.keys()), desc="generating test pairs"):
            gen_list = test_sigs[subject_id]['genuine']
            forg_list = test_sigs[subject_id]['forge']
            
            if len(gen_list) < 2:
                continue  # Skip subjects with < 2 genuine signatures
            
            # Positive pairs: genuine-to-genuine from same subject
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    pairs.append((gen_list[i], gen_list[j], 1))
            
            # Negative pairs: genuine-to-forged from same subject
            for i in range(len(gen_list)):
                for j in range(len(forg_list)):
                    pairs.append((gen_list[i], forg_list[j], 0))
        
        return pairs
    
    train_pairs = generate_train_pairs()
    test_pairs = generate_test_pairs()
    
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
        'num_train_genuine': len(train_genuine),
        'num_train_forged': len(train_forged),
        'num_test_subjects': len(test_sigs)
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
