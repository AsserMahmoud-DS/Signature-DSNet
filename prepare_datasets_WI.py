"""
Writer-independent pair generation for signature datasets.

- Keeps the existing folder layouts unchanged.
- Creates pair files with TAB-separated columns:
  refer_path <TAB> test_path <TAB> label
- GPDS writer-independent split is done by writer IDs across the full 150-writer pool.
"""

import os
import re
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from prepare_datasets import generate_cedar_pairs


def _parse_writer_id_from_filename(filename: str, is_forgery: bool) -> Optional[str]:
    """Extract writer id from GDPS filename.

    Expected common patterns:
    - Genuine: c-032-11 (Copy).jpg
    - Forgery: cf-012-13 (Copy).jpg
    """
    if is_forgery:
        match = re.match(r"^cf-(\d+)-", filename, flags=re.IGNORECASE)
    else:
        match = re.match(r"^c-(\d+)-", filename, flags=re.IGNORECASE)
    if not match:
        return None
    # Normalize for stable sorting/comparison.
    return str(int(match.group(1)))


def _add_sig(
    writer_map: Dict[str, Dict[str, List[str]]],
    writer_id: str,
    rel_path: str,
    is_forgery: bool,
) -> None:
    if writer_id not in writer_map:
        writer_map[writer_id] = {"genuine": [], "forge": []}
    key = "forge" if is_forgery else "genuine"
    writer_map[writer_id][key].append(rel_path)


def _collect_gdps_signatures_by_writer(data_root: str) -> Dict[str, Dict[str, List[str]]]:
    """Collect all GPDS signatures by writer across train/ and test/ trees.

    This does not move files. It only builds a mapping:
    {writer_id: {'genuine': [...], 'forge': [...]}}
    """
    writer_map: Dict[str, Dict[str, List[str]]] = {}

    # 1) Flat train structure: train/genuine and train/forge
    train_genuine = os.path.join(data_root, "train", "genuine")
    train_forge = os.path.join(data_root, "train", "forge")

    if os.path.isdir(train_genuine):
        for img_file in sorted(os.listdir(train_genuine)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            writer_id = _parse_writer_id_from_filename(img_file, is_forgery=False)
            if writer_id is None:
                continue
            rel_path = os.path.join("train", "genuine", img_file)
            _add_sig(writer_map, writer_id, rel_path, is_forgery=False)

    if os.path.isdir(train_forge):
        for img_file in sorted(os.listdir(train_forge)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            writer_id = _parse_writer_id_from_filename(img_file, is_forgery=True)
            if writer_id is None:
                continue
            rel_path = os.path.join("train", "forge", img_file)
            _add_sig(writer_map, writer_id, rel_path, is_forgery=True)

    # 2) Hierarchical test structure: test/<subject_id>/{genuine,forge}
    test_root = os.path.join(data_root, "test")
    if os.path.isdir(test_root):
        for subject_dir in sorted(os.listdir(test_root)):
            subject_path = os.path.join(test_root, subject_dir)
            if not os.path.isdir(subject_path):
                continue

            subject_hint = None
            try:
                subject_hint = str(int(subject_dir))
            except ValueError:
                pass

            genuine_dir = os.path.join(subject_path, "genuine")
            forge_dir = os.path.join(subject_path, "forge")

            if os.path.isdir(genuine_dir):
                for img_file in sorted(os.listdir(genuine_dir)):
                    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    writer_id = _parse_writer_id_from_filename(img_file, is_forgery=False)
                    if writer_id is None:
                        writer_id = subject_hint
                    if writer_id is None:
                        continue
                    rel_path = os.path.join("test", subject_dir, "genuine", img_file)
                    _add_sig(writer_map, writer_id, rel_path, is_forgery=False)

            if os.path.isdir(forge_dir):
                for img_file in sorted(os.listdir(forge_dir)):
                    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    writer_id = _parse_writer_id_from_filename(img_file, is_forgery=True)
                    if writer_id is None:
                        writer_id = subject_hint
                    if writer_id is None:
                        continue
                    rel_path = os.path.join("test", subject_dir, "forge", img_file)
                    _add_sig(writer_map, writer_id, rel_path, is_forgery=True)

    return writer_map


def _generate_pairs_for_writers(
    writer_map: Dict[str, Dict[str, List[str]]],
    writer_ids: List[str],
    split_name: str,
) -> List[Tuple[str, str, int]]:
    pairs: List[Tuple[str, str, int]] = []

    for writer_id in tqdm(writer_ids, desc=f"generating {split_name} pairs"):
        gen_list = sorted(writer_map[writer_id]["genuine"])
        forg_list = sorted(writer_map[writer_id]["forge"])

        if len(gen_list) < 2:
            continue

        # Positive: genuine-genuine within same writer.
        for i in range(len(gen_list)):
            for j in range(i + 1, len(gen_list)):
                pairs.append((gen_list[i], gen_list[j], 1))

        # Negative: genuine-forgery within same writer.
        for i in range(len(gen_list)):
            for j in range(len(forg_list)):
                pairs.append((gen_list[i], forg_list[j], 0))

    return pairs


def _sample_pairs_with_label_ratio(
    pairs: List[Tuple[str, str, int]],
    target_count: Optional[int],
    random_state: int,
    split_name: str,
) -> List[Tuple[str, str, int]]:
    """Downsample pairs while approximately preserving original class ratio."""
    if target_count is None or target_count <= 0 or target_count >= len(pairs):
        return pairs

    rng = random.Random(random_state)
    pos_pairs = [p for p in pairs if p[2] == 1]
    neg_pairs = [p for p in pairs if p[2] == 0]

    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        sampled = pairs[:]
        rng.shuffle(sampled)
        sampled = sampled[:target_count]
        print(
            f"{split_name}: capped to {len(sampled)} pairs without stratification "
            f"(single-class source)."
        )
        return sampled

    total = len(pairs)
    target_pos = int(round(target_count * (len(pos_pairs) / total)))
    target_pos = max(1, min(target_pos, len(pos_pairs), target_count - 1))
    target_neg = target_count - target_pos

    if target_neg > len(neg_pairs):
        target_neg = len(neg_pairs)
        target_pos = min(target_count - target_neg, len(pos_pairs))
    if target_pos > len(pos_pairs):
        target_pos = len(pos_pairs)
        target_neg = min(target_count - target_pos, len(neg_pairs))

    sampled = rng.sample(pos_pairs, target_pos) + rng.sample(neg_pairs, target_neg)
    rng.shuffle(sampled)

    label_counts = Counter([p[2] for p in sampled])
    print(
        f"{split_name}: capped to {len(sampled)} pairs "
        f"(pos={label_counts.get(1, 0)}, neg={label_counts.get(0, 0)})"
    )
    return sampled


def _write_pairs(pair_file: str, pairs: List[Tuple[str, str, int]]) -> None:
    with open(pair_file, "w") as f:
        for refer, test, label in pairs:
            f.write(f"{refer}\t{test}\t{label}\n")


def sample_pair_file(
    src_pair_file: str,
    dst_pair_file: str,
    target_count: int,
    random_state: int = 42,
) -> Dict[str, object]:
    """Create a smaller pair file by sampling while preserving class ratio."""
    if target_count <= 0:
        raise ValueError("target_count must be > 0")
    if not os.path.exists(src_pair_file):
        raise FileNotFoundError(f"Source pair file not found: {src_pair_file}")

    pairs: List[Tuple[str, str, int]] = []
    with open(src_pair_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                if len(parts) > 0:
                    print(f"⚠️  Skipping malformed line {line_num}: {line.strip()[:80]}")
                continue
            refer, test, label = parts[0], parts[1], int(parts[2])
            pairs.append((refer, test, label))

    sampled = _sample_pairs_with_label_ratio(
        pairs,
        target_count=target_count,
        random_state=random_state,
        split_name="pair-file sample",
    )
    _write_pairs(dst_pair_file, sampled)

    counts = Counter([p[2] for p in sampled])
    return {
        "src_pair_file": src_pair_file,
        "dst_pair_file": dst_pair_file,
        "target_count": target_count,
        "num_source_pairs": len(pairs),
        "num_written_pairs": len(sampled),
        "num_pos_pairs": counts.get(1, 0),
        "num_neg_pairs": counts.get(0, 0),
    }


def generate_gdps_pairs(
    data_root: str,
    train_ratio: float = 0.7,
    val_ratio_within_heldout: float = 0.5,
    random_state: int = 42,
    target_train_pairs: Optional[int] = None,
    target_val_pairs: Optional[int] = None,
    target_test_pairs: Optional[int] = None,
    train_filename: str = "gray_train.txt",
    val_filename: str = "gray_val.txt",
    test_filename: str = "gray_test.txt",
):
    """Generate train/val/test pair files for GDPS with writer-independent split.

    Strategy:
    1) Collect all signatures from both current train/ and test/ folders by writer ID.
    2) Split writer IDs into train vs held-out by `train_ratio`.
    3) Split held-out writers into val vs test by `val_ratio_within_heldout`.
    4) Generate within-writer positive/negative pairs for each split.
    5) Optionally cap each split by target pair counts.
    """

    writer_map = _collect_gdps_signatures_by_writer(data_root)
    all_writers = sorted(writer_map.keys(), key=lambda x: int(x))

    if len(all_writers) == 0:
        raise RuntimeError(f"No GDPS writers found under: {data_root}")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 < val_ratio_within_heldout < 1.0):
        raise ValueError("val_ratio_within_heldout must be in (0, 1)")

    rng = random.Random(random_state)
    all_writers_shuffled = all_writers[:]
    rng.shuffle(all_writers_shuffled)

    n_total = len(all_writers_shuffled)
    n_train = int(round(n_total * train_ratio))
    n_train = max(1, min(n_train, n_total - 2))

    train_writers = sorted(all_writers_shuffled[:n_train], key=lambda x: int(x))
    heldout_writers = sorted(all_writers_shuffled[n_train:], key=lambda x: int(x))

    n_heldout = len(heldout_writers)
    n_val = int(round(n_heldout * val_ratio_within_heldout))
    n_val = max(1, min(n_val, n_heldout - 1))

    heldout_shuffled = heldout_writers[:]
    rng.shuffle(heldout_shuffled)
    val_writers = sorted(heldout_shuffled[:n_val], key=lambda x: int(x))
    test_writers = sorted(heldout_shuffled[n_val:], key=lambda x: int(x))

    print(f"GDPS-WI: total writers={len(all_writers)}")
    print(
        f"GDPS-WI: train writers={len(train_writers)}, "
        f"val writers={len(val_writers)}, test writers={len(test_writers)}"
    )

    # Lightweight data sanity summary.
    train_sig_count = sum(len(writer_map[w]["genuine"]) + len(writer_map[w]["forge"]) for w in train_writers)
    val_sig_count = sum(len(writer_map[w]["genuine"]) + len(writer_map[w]["forge"]) for w in val_writers)
    test_sig_count = sum(len(writer_map[w]["genuine"]) + len(writer_map[w]["forge"]) for w in test_writers)
    print(
        f"GDPS-WI: signatures in train writers={train_sig_count}, "
        f"val writers={val_sig_count}, test writers={test_sig_count}"
    )

    train_pairs_full = _generate_pairs_for_writers(writer_map, train_writers, "train")
    val_pairs_full = _generate_pairs_for_writers(writer_map, val_writers, "val")
    test_pairs_full = _generate_pairs_for_writers(writer_map, test_writers, "test")

    train_pairs = _sample_pairs_with_label_ratio(
        train_pairs_full,
        target_count=target_train_pairs,
        random_state=random_state + 11,
        split_name="train",
    )
    val_pairs = _sample_pairs_with_label_ratio(
        val_pairs_full,
        target_count=target_val_pairs,
        random_state=random_state + 17,
        split_name="val",
    )
    test_pairs = _sample_pairs_with_label_ratio(
        test_pairs_full,
        target_count=target_test_pairs,
        random_state=random_state + 23,
        split_name="test",
    )

    train_txt = os.path.join(data_root, train_filename)
    val_txt = os.path.join(data_root, val_filename)
    test_txt = os.path.join(data_root, test_filename)

    _write_pairs(train_txt, train_pairs)
    _write_pairs(val_txt, val_pairs)
    _write_pairs(test_txt, test_pairs)

    print(f"✅ Generated {train_txt}: {len(train_pairs)} pairs")
    print(f"✅ Generated {val_txt}: {len(val_pairs)} pairs")
    print(f"✅ Generated {test_txt}: {len(test_pairs)} pairs")

    return {
        "train_file": train_txt,
        "val_file": val_txt,
        "test_file": test_txt,
        "num_total_writers": len(all_writers),
        "num_train_writers": len(train_writers),
        "num_val_writers": len(val_writers),
        "num_test_writers": len(test_writers),
        "train_writers": train_writers,
        "val_writers": val_writers,
        "test_writers": test_writers,
        "num_train_pairs_full": len(train_pairs_full),
        "num_val_pairs_full": len(val_pairs_full),
        "num_test_pairs_full": len(test_pairs_full),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate writer-independent pair files for signature datasets"
    )
    parser.add_argument("--cedar", type=str, help="Path to CEDAR dataset root")
    parser.add_argument("--gdps", type=str, help="Path to GDPS dataset root")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Writer-level train ratio for GDPS split (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio-heldout",
        type=float,
        default=0.5,
        help="Val-writer ratio inside held-out writers (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for writer split",
    )
    parser.add_argument(
        "--target-train-pairs",
        type=int,
        default=None,
        help="Optional cap for generated train pairs",
    )
    parser.add_argument(
        "--target-val-pairs",
        type=int,
        default=None,
        help="Optional cap for generated val pairs",
    )
    parser.add_argument(
        "--target-test-pairs",
        type=int,
        default=None,
        help="Optional cap for generated test pairs",
    )

    args = parser.parse_args()

    if args.cedar:
        # CEDAR path remains unchanged; writer-level split is already correct there.
        generate_cedar_pairs(args.cedar)

    if args.gdps:
        generate_gdps_pairs(
            data_root=args.gdps,
            train_ratio=args.train_ratio,
            val_ratio_within_heldout=args.val_ratio_heldout,
            random_state=args.seed,
            target_train_pairs=args.target_train_pairs,
            target_val_pairs=args.target_val_pairs,
            target_test_pairs=args.target_test_pairs,
        )
