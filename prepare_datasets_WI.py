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


def generate_gdps_pairs(
    data_root: str,
    train_ratio: float = 0.7,
    random_state: int = 42,
):
    """Generate train/test pair files for GDPS with writer-independent split.

    Strategy:
    1) Collect all signatures from both current train/ and test/ folders by writer ID.
    2) Split writer IDs into train/test by `train_ratio`.
    3) Generate within-writer positive/negative pairs for each split.

    Note:
    - Your validation split can be derived from this new test set (e.g., split test pairs in half).
    """

    writer_map = _collect_gdps_signatures_by_writer(data_root)
    all_writers = sorted(writer_map.keys(), key=lambda x: int(x))

    if len(all_writers) == 0:
        raise RuntimeError(f"No GDPS writers found under: {data_root}")

    random.seed(random_state)
    n_train = int(len(all_writers) * train_ratio)
    train_writers = sorted(random.sample(all_writers, n_train), key=lambda x: int(x))
    test_writers = sorted(set(all_writers) - set(train_writers), key=lambda x: int(x))

    print(f"GDPS-WI: total writers={len(all_writers)}")
    print(f"GDPS-WI: train writers={len(train_writers)}, test writers={len(test_writers)}")

    # Lightweight data sanity summary.
    train_sig_count = sum(len(writer_map[w]["genuine"]) + len(writer_map[w]["forge"]) for w in train_writers)
    test_sig_count = sum(len(writer_map[w]["genuine"]) + len(writer_map[w]["forge"]) for w in test_writers)
    print(f"GDPS-WI: signatures in train writers={train_sig_count}, test writers={test_sig_count}")

    train_pairs = _generate_pairs_for_writers(writer_map, train_writers, "train")
    test_pairs = _generate_pairs_for_writers(writer_map, test_writers, "test")

    train_txt = os.path.join(data_root, "gray_train.txt")
    test_txt = os.path.join(data_root, "gray_test.txt")

    with open(train_txt, "w") as f:
        for refer, test, label in train_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")

    with open(test_txt, "w") as f:
        for refer, test, label in test_pairs:
            f.write(f"{refer}\t{test}\t{label}\n")

    print(f"✅ Generated {train_txt}: {len(train_pairs)} pairs")
    print(f"✅ Generated {test_txt}: {len(test_pairs)} pairs")

    return {
        "train_file": train_txt,
        "test_file": test_txt,
        "num_total_writers": len(all_writers),
        "num_train_writers": len(train_writers),
        "num_test_writers": len(test_writers),
        "train_writers": train_writers,
        "test_writers": test_writers,
        "num_train_pairs": len(train_pairs),
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for writer split",
    )

    args = parser.parse_args()

    if args.cedar:
        # CEDAR path remains unchanged; writer-level split is already correct there.
        generate_cedar_pairs(args.cedar)

    if args.gdps:
        generate_gdps_pairs(
            data_root=args.gdps,
            train_ratio=args.train_ratio,
            random_state=args.seed,
        )
