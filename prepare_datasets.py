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


def generate_gdps_pairs(data_root, train_ratio=0.7, random_state=42, positive_ratio=0.5):
    """
    Generate train/test pair files for GDPS dataset.
    
    GDPS structure:
    - train/: writer folders OR flat (handled here)
    - test/: subject_id/genuine/ and subject_id/forge/
    
    Args:
        positive_ratio: Target ratio of positive (genuine) pairs (0.33 for 67% forged, 0.5 for balanced)
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
    
    # Generate training pairs (from flat train structure)
    def generate_train_pairs():
        """
        Load train signatures organized by writer.
        Expected structure: train/writer_id/genuine/ and train/writer_id/forge/
        """
        pairs = []
        import random
        random.seed(random_state)
        
        # If train is flat (train/genuine/ and train/forge/), convert to per-writer
        # Otherwise, iterate through train/writer_id/ folders
        
        writer_dirs = sorted([d for d in os.listdir(train_path) 
                              if os.path.isdir(os.path.join(train_path, d)) and d.isdigit()])
        
        if writer_dirs:
            # Per-writer training structure
            for writer_id in tqdm(writer_dirs, desc="generating train pairs (per-writer)"):
                writer_path = os.path.join(train_path, writer_id)
                
                gen_list = []
                forg_list = []
                
                # Load genuine
                gen_folder = os.path.join(writer_path, "genuine")
                if os.path.isdir(gen_folder):
                    for img in sorted(os.listdir(gen_folder)):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            rel_path = os.path.join("train", writer_id, "genuine", img)
                            gen_list.append(rel_path)
                
                # Load forged
                forge_folder = os.path.join(writer_path, "forge")
                if os.path.isdir(forge_folder):
                    for img in sorted(os.listdir(forge_folder)):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            rel_path = os.path.join("train", writer_id, "forge", img)
                            forg_list.append(rel_path)
                
                if len(gen_list) < 2:
                    continue
                
                # Positive pairs: genuine-to-genuine
                positive_pairs = []
                for i in range(len(gen_list)):
                    for j in range(i + 1, len(gen_list)):
                        positive_pairs.append((gen_list[i], gen_list[j], 1))
                
                # Negative pairs: genuine-to-forged
                negative_pairs = []
                for i in range(len(gen_list)):
                    for j in range(len(forg_list)):
                        negative_pairs.append((gen_list[i], forg_list[j], 0))
                
                # Balance: subsample negatives to match ratio
                if positive_ratio < 1.0 and len(negative_pairs) > 0:
                    target_negatives = int(len(positive_pairs) / positive_ratio) - len(positive_pairs)
                    if target_negatives > 0 and len(negative_pairs) > target_negatives:
                        negative_pairs = random.sample(negative_pairs, target_negatives)
                
                pairs.extend(positive_pairs)
                pairs.extend(negative_pairs)
        
        else:
            # Flat training structure (fallback - NOT recommended for writer-specific model)
            print("⚠️  WARNING: Using flat train structure. This creates artificial cross-writer pairs!")
            gen_list, forg_list = load_train_signatures()
            
            # Same logic as before
            for i in range(len(gen_list)):
                for j in range(i + 1, len(gen_list)):
                    pairs.append((gen_list[i], gen_list[j], 1))
            
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
