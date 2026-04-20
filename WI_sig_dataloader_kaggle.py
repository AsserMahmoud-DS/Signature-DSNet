import os
import re
import random
import hashlib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

from module.preprocess import normalize_image, get_clean_signature_crop


def _pair_key(refer: str, test: str, label: int):
    if label == 1 and refer > test:
        refer, test = test, refer
    return refer, test, label


def _extract_writer_id_from_relpath(path_str):
    filename = os.path.basename(path_str)
    match = re.match(r"^cf?-(\d+)-", filename, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))

    cedar_match = re.match(r"^(?:original|forgeries)_(\d+)_\d+\.", filename, flags=re.IGNORECASE)
    if cedar_match:
        return str(int(cedar_match.group(1)))

    parts = path_str.replace('\\', '/').split('/')
    if 'test' in parts:
        idx = parts.index('test')
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return str(int(parts[idx + 1]))
    return None


def _get_multi_ref_config(opt, train):
    enabled = bool(getattr(opt, 'multi_ref_test', False)) if opt is not None else False
    enabled = enabled and (not train)
    num_refs = int(getattr(opt, 'num_refs', 1)) if opt is not None else 1
    ref_seed = int(getattr(opt, 'ref_seed', 42)) if opt is not None else 42

    if enabled and num_refs < 1:
        raise ValueError(f"num_refs must be >= 1 when multi_ref_test is enabled, got {num_refs}")
    return enabled, max(1, num_refs), ref_seed


def _writer_sort_key(writer_id):
    try:
        return (0, int(writer_id))
    except (TypeError, ValueError):
        return (1, str(writer_id))


def _stable_writer_seed(base_seed, writer_id):
    seed_blob = f"{int(base_seed)}::{writer_id}".encode('utf-8')
    digest = hashlib.sha256(seed_blob).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def _expand_pairs_for_multi_ref_eval(raw_pairs, num_refs, ref_seed, dataset_name='dataset'):
    writer_pool = defaultdict(lambda: {'genuine': set(), 'forgery': set()})
    unknown_examples = []

    for refer, test, label in raw_pairs:
        writer_id = _extract_writer_id_from_relpath(refer)
        if writer_id is None:
            writer_id = _extract_writer_id_from_relpath(test)
        if writer_id is None:
            if len(unknown_examples) < 5:
                unknown_examples.append((refer, test, int(label)))
            continue

        writer_pool[writer_id]['genuine'].add(refer)
        if int(label) == 1:
            writer_pool[writer_id]['genuine'].add(test)
        else:
            writer_pool[writer_id]['forgery'].add(test)

    if unknown_examples:
        raise ValueError(
            f"{dataset_name}: failed to extract writer id for {len(unknown_examples)}+ pairs in multi-ref mode. "
            f"Examples: {unknown_examples}"
        )

    expanded_pairs = []
    writer_count = 0
    total_queries = 0

    for writer_id in sorted(writer_pool.keys(), key=_writer_sort_key):
        genuine_candidates = sorted(writer_pool[writer_id]['genuine'])
        forgery_queries = sorted(writer_pool[writer_id]['forgery'])

        if len(genuine_candidates) <= num_refs:
            raise ValueError(
                f"{dataset_name}: writer {writer_id} has only {len(genuine_candidates)} genuine images in this split, "
                f"but num_refs={num_refs} requires at least {num_refs + 1}."
            )

        writer_seed = _stable_writer_seed(ref_seed, writer_id)
        rng = random.Random(writer_seed)
        references = sorted(rng.sample(genuine_candidates, num_refs))
        reference_set = set(references)

        genuine_queries = [p for p in genuine_candidates if p not in reference_set]
        queries = [(q, 1) for q in genuine_queries] + [(q, 0) for q in forgery_queries]

        writer_count += 1
        total_queries += len(queries)

        for query_path, label in queries:
            for ref_path in references:
                expanded_pairs.append((ref_path, query_path, int(label)))

    summary = {
        'writers': writer_count,
        'num_refs': int(num_refs),
        'num_queries': int(total_queries),
        'num_pairs': int(len(expanded_pairs)),
    }
    return expanded_pairs, summary


def _get_crop_margin_ratio(opt, default=0.0):
    # Backward compatible: if not provided by old notebooks/configs, use 0.0.
    if opt is None:
        return float(default)
    return max(0.0, float(getattr(opt, 'crop_margin_ratio', default)))


def _get_crop_v2_min_blob_size(opt, default=50):
    if opt is None:
        return int(default)
    return max(1, int(getattr(opt, 'crop_v2_min_blob_size', default)))


def _get_crop_v2_disk_radius(opt, default=1):
    if opt is None:
        return int(default)
    return max(1, int(getattr(opt, 'crop_v2_disk_radius', default)))


def _is_cropped_mode(mode):
    return str(mode).strip().lower() in {'cropped', 'cropped_v2_median'}


def _resolve_image_mode(mode, default='cropped'):
    if mode is None:
        return default
    mode_str = str(mode).strip().lower()
    if mode_str in {'normalized', 'cropped', 'cropped_v2_median'}:
        return mode_str
    print(f"⚠️  Unsupported mode='{mode}'. Falling back to '{default}'.")
    return default


def _apply_crop_margin(cropped_image: np.ndarray, margin_ratio: float) -> np.ndarray:
    if margin_ratio <= 0.0:
        return cropped_image

    h, w = cropped_image.shape
    if h == 0 or w == 0:
        return cropped_image

    pad_h = int(h * margin_ratio)
    pad_w = int(w * margin_ratio)
    if pad_h <= 0 and pad_w <= 0:
        return cropped_image

    padded = np.ones((h + 2 * pad_h, w + 2 * pad_w), dtype=np.uint8) * 255
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = cropped_image
    return padded


def imread_tool(img_path, crop_margin_ratio=0.0):
    image = np.asarray(Image.open(img_path).convert('L'))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize)
    cropped_image = _apply_crop_margin(cropped_image, float(crop_margin_ratio))
    return Image.fromarray(normalized_image), Image.fromarray(cropped_image)


def _load_sig_image_for_mode(
    img_path,
    mode,
    crop_margin_ratio=0.0,
    crop_v2_min_blob_size=50,
    crop_v2_disk_radius=1,
):
    mode_resolved = _resolve_image_mode(mode)
    if mode_resolved == 'normalized':
        normalized_img, _ = imread_tool(img_path, crop_margin_ratio=0.0)
        return normalized_img

    if mode_resolved == 'cropped':
        _, cropped_img = imread_tool(img_path, crop_margin_ratio=float(crop_margin_ratio))
        return cropped_img

    with Image.open(img_path) as _img:
        image = np.asarray(_img.convert('L'))
    cropped_v2 = get_clean_signature_crop(
        image.astype(np.uint8),
        min_blob_size=int(crop_v2_min_blob_size),
        disk_radius=int(crop_v2_disk_radius),
    )
    cropped_v2 = _apply_crop_margin(cropped_v2, float(crop_margin_ratio))
    return Image.fromarray(cropped_v2)


def _get_rotation_degrees(opt, default=10.0):
    # Keep legacy behavior when the notebook/config does not provide this value.
    if opt is None:
        return float(default)
    return float(getattr(opt, 'rotation_degrees', default))


class LetterboxResize:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        img = img.resize((new_w, new_h), Image.LANCZOS)

        new_img = Image.new("L", (self.size, self.size), color=255)
        new_img.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))

        return new_img


class SigDataset_CEDAR_Kaggle(Dataset):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        self.image_root = image_root
        self.pair_root = pair_root
        self.image_size = image_size
        self.mode = _resolve_image_mode(mode)
        self.crop_margin_ratio = _get_crop_margin_ratio(opt, default=0.0)
        self.crop_v2_min_blob_size = _get_crop_v2_min_blob_size(opt, default=50)
        self.crop_v2_disk_radius = _get_crop_v2_disk_radius(opt, default=1)
        self.multi_ref_enabled, self.multi_ref_num_refs, self.multi_ref_seed = _get_multi_ref_config(opt, train)
        
        trans_list = [
            LetterboxResize(image_size),
            transforms.ToTensor()]
        self.basic_transforms = transforms.Compose(trans_list)

        rotation_degrees = _get_rotation_degrees(opt, default=10.0)
        pre_tensor_aug_list = [
                        transforms.RandomApply([
                            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
                        ], p=0.3),
            transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees), fill=255)]
        self.pre_tensor_augment = transforms.Compose(pre_tensor_aug_list)
        self.post_tensor_augment = transforms.RandomErasing(
            p=0.05,
            scale=(0.005, 0.02),
            ratio=(0.3, 3.3),
            value=1.0,
        )
        
        data_root = image_root

        if pair_filename is not None:
            pair_path = os.path.join(pair_root, pair_filename)
        elif train:
            pair_path = os.path.join(pair_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(pair_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_test_part.txt")
        
        # First collect all image paths
        img_paths_to_load = []
        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    if img[-4:] == '.png':
                        img_path = os.path.join(dir_path, img)
                        img_paths_to_load.append(img_path)
        
        # Then load all images with a single progress bar
        self.img_dict = {}
        for img_path in tqdm(img_paths_to_load, desc="Loading CEDAR images", unit="img"):
            sig_image = _load_sig_image_for_mode(
                img_path,
                mode=self.mode,
                crop_margin_ratio=(self.crop_margin_ratio if _is_cropped_mode(self.mode) else 0.0),
                crop_v2_min_blob_size=self.crop_v2_min_blob_size,
                crop_v2_disk_radius=self.crop_v2_disk_radius,
            )
            self.img_dict[img_path] = sig_image
        
        with open(pair_path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.pairs = []
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split('\t')  # Use tab as delimiter to handle filenames with spaces
            if len(parts) < 3:
                if len(parts) > 0:  # Skip empty lines
                    print(f"⚠️  Line {line_num} has {len(parts)} columns (expected 3): {line.strip()[:80]}")
                continue
            
            try:
                refer, test, label = parts[0], parts[1], parts[2]
                
                refer_path = os.path.join(data_root, refer)
                test_path = os.path.join(data_root, test)

                if refer_path not in self.img_dict or test_path not in self.img_dict:
                    raise KeyError(f"Missing cached image for pair: {refer_path} | {test_path}")

                self.pairs.append((refer_path, test_path))
                self.labels.append(int(label))
            except (KeyError, ValueError) as e:
                print(f"❌ Error processing line {line_num}: {line.strip()}")
                print(f"   columns: {parts}")
                print(f"   Error: {e}")
                raise

        if self.multi_ref_enabled:
            raw_pairs = [(r, t, l) for (r, t), l in zip(self.pairs, self.labels)]
            expanded_pairs, summary = _expand_pairs_for_multi_ref_eval(
                raw_pairs,
                num_refs=self.multi_ref_num_refs,
                ref_seed=self.multi_ref_seed,
                dataset_name='CEDAR',
            )
            self.pairs = [(r, t) for r, t, _ in expanded_pairs]
            self.labels = [int(l) for _, _, l in expanded_pairs]
            print(
                "🔁 CEDAR multi-ref test enabled: "
                f"writers={summary['writers']} refs={summary['num_refs']} "
                f"queries={summary['num_queries']} pairs={summary['num_pairs']}"
            )
        
        self.train = train
        print(f"Kaggle: Loaded {len(self.labels)} pairs (pre-cached in RAM)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        refer_path, test_path = self.pairs[index]
        refer_img = self.img_dict[refer_path].copy()
        test_img = self.img_dict[test_path].copy()

        if self.train:
            refer_img = self.pre_tensor_augment(refer_img)
            test_img = self.pre_tensor_augment(test_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)

        return image_pair, torch.tensor([[self.labels[index]]])


class SigDataset_CEDAR_Kaggle_Lite(Dataset):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        """RAM-cached CEDAR loader that only caches images referenced by the pair file.

        Differences vs `SigDataset_CEDAR_Kaggle`:
        - Reads pair file first and caches only needed images (subset-friendly)
        - Does NOT pre-build `self.datas` (avoids duplicating pair tensors in RAM)
        """

        self.image_root = image_root
        self.pair_root = pair_root
        self.image_size = image_size
        self.mode = _resolve_image_mode(mode)
        self.train = train
        self.crop_margin_ratio = _get_crop_margin_ratio(opt, default=0.0)
        self.crop_v2_min_blob_size = _get_crop_v2_min_blob_size(opt, default=50)
        self.crop_v2_disk_radius = _get_crop_v2_disk_radius(opt, default=1)
        self.multi_ref_enabled, self.multi_ref_num_refs, self.multi_ref_seed = _get_multi_ref_config(opt, train)

        trans_list = [
            LetterboxResize(image_size),
            transforms.ToTensor(),
        ]
        self.basic_transforms = transforms.Compose(trans_list)

        rotation_degrees = _get_rotation_degrees(opt, default=10.0)
        pre_tensor_aug_list = [
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
            ], p=0.3),
            transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees), fill=255),
        ]
        self.pre_tensor_augment = transforms.Compose(pre_tensor_aug_list)
        self.post_tensor_augment = transforms.RandomErasing(
            p=0.05,
            scale=(0.005, 0.02),
            ratio=(0.3, 3.3),
            value=1.0,
        )

        data_root = image_root

        if pair_filename is not None:
            pair_path = os.path.join(pair_root, pair_filename)
        elif train:
            pair_path = os.path.join(pair_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(pair_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_test_part.txt")

        if not os.path.exists(pair_path):
            raise FileNotFoundError(
                f"Pair file not found at {pair_path}. "
                "Make sure to generate CEDAR gray_train.txt/gray_test.txt first."
            )

        # Read pair file first
        self.pairs = []  # (refer_rel, test_rel, label_int)
        needed_rel_paths = set()
        with open(pair_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    if len(parts) > 0:
                        print(f"⚠️  Line {line_num} has {len(parts)} columns (expected 3): {line.strip()[:80]}")
                    continue

                refer_rel, test_rel, label = parts[0], parts[1], parts[2]
                try:
                    label_int = int(label)
                except ValueError as e:
                    print(f"❌ Error parsing label on line {line_num}: {line.strip()}")
                    raise e

                self.pairs.append((refer_rel, test_rel, label_int))
                needed_rel_paths.add(refer_rel)
                needed_rel_paths.add(test_rel)

        # Cache only needed images
        self.img_dict = {}
        img_paths_to_load = []
        for rel_path in sorted(needed_rel_paths):
            abs_path = os.path.join(data_root, rel_path)
            img_paths_to_load.append((rel_path, abs_path))

        for rel_path, abs_path in tqdm(img_paths_to_load, desc="Loading CEDAR images (lite)", unit="img"):
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Missing image referenced by pair file: {abs_path}")
            sig_image = _load_sig_image_for_mode(
                abs_path,
                mode=self.mode,
                crop_margin_ratio=(self.crop_margin_ratio if _is_cropped_mode(self.mode) else 0.0),
                crop_v2_min_blob_size=self.crop_v2_min_blob_size,
                crop_v2_disk_radius=self.crop_v2_disk_radius,
            )
            self.img_dict[rel_path] = sig_image

        if self.multi_ref_enabled:
            expanded_pairs, summary = _expand_pairs_for_multi_ref_eval(
                self.pairs,
                num_refs=self.multi_ref_num_refs,
                ref_seed=self.multi_ref_seed,
                dataset_name='CEDAR-lite',
            )
            self.pairs = expanded_pairs
            print(
                "🔁 CEDAR-lite multi-ref test enabled: "
                f"writers={summary['writers']} refs={summary['num_refs']} "
                f"queries={summary['num_queries']} pairs={summary['num_pairs']}"
            )

        print(
            f"Kaggle CEDAR (lite): Loaded {len(self.pairs)} pairs; "
            f"cached {len(self.img_dict)} images in RAM"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        refer_rel, test_rel, label_int = self.pairs[index]
        refer_img = self.img_dict[refer_rel].copy()
        test_img = self.img_dict[test_rel].copy()

        if self.train:
            refer_img = self.pre_tensor_augment(refer_img)
            test_img = self.pre_tensor_augment(test_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)

        return image_pair, torch.tensor([[label_int]])


class SigDataset_GDPS_Kaggle(Dataset):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        self.image_root = image_root
        self.pair_root = pair_root
        self.image_size = image_size
        self.mode = _resolve_image_mode(mode)
        self.crop_margin_ratio = _get_crop_margin_ratio(opt, default=0.0)
        self.crop_v2_min_blob_size = _get_crop_v2_min_blob_size(opt, default=50)
        self.crop_v2_disk_radius = _get_crop_v2_disk_radius(opt, default=1)
        self.multi_ref_enabled, self.multi_ref_num_refs, self.multi_ref_seed = _get_multi_ref_config(opt, train)
        
        trans_list = [
            LetterboxResize(image_size),
            transforms.ToTensor()]
        self.basic_transforms = transforms.Compose(trans_list)

        rotation_degrees = _get_rotation_degrees(opt, default=10.0)
        pre_tensor_aug_list = [
                        transforms.RandomApply([
                            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
                        ], p=0.3),
            transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees), fill=255)]
        self.pre_tensor_augment = transforms.Compose(pre_tensor_aug_list)
        self.post_tensor_augment = transforms.RandomErasing(
            p=0.05,
            scale=(0.005, 0.02),
            ratio=(0.3, 3.3),
            value=1.0,
        )
        
        # For GDPS: pair files are at pair_root (same as image_root usually)
        # Paths in pair files already include "test/" or "train/" prefix
        data_root = image_root
        pair_path = None
        
        if pair_filename is not None:
            pair_path = os.path.join(pair_root, pair_filename)
        elif train:
            pair_path = os.path.join(pair_root, "gray_train.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_train_part.txt")
        else:
            pair_path = os.path.join(pair_root, "gray_test.txt")
            if opt and hasattr(opt, 'part') and opt.part:
                pair_path = os.path.join(pair_root, "gray_test_part.txt")
        
        # Pre-cache images from all subdirectories
        self.img_dict = {}
        if not os.path.exists(pair_path):
            print(f"⚠️  Pair file not found at {pair_path}")
            print(f"    Make sure to run generate_gdps_pairs() first to create pair files")
            self.img_dict = {}
        else:
            # Collect all unique image paths referenced in the pair file (both columns)
            rel_paths_needed = set()
            with open(pair_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # Normalize separators to os.sep for consistent lookup
                        rel_paths_needed.add(parts[0].replace('\\', '/').replace('/', os.sep))
                        rel_paths_needed.add(parts[1].replace('\\', '/').replace('/', os.sep))

            # Pre-cache only the images we actually need
            img_paths_to_load = []
            for rel_path in rel_paths_needed:
                full_path = os.path.join(data_root, rel_path)
                if os.path.exists(full_path):
                    img_paths_to_load.append((rel_path, full_path))

            # Load images with a single progress bar
            for rel_path, full_path in tqdm(img_paths_to_load, desc="Loading GDPS images", unit="img"):
                sig_image = _load_sig_image_for_mode(
                    full_path,
                    mode=self.mode,
                    crop_margin_ratio=(self.crop_margin_ratio if _is_cropped_mode(self.mode) else 0.0),
                    crop_v2_min_blob_size=self.crop_v2_min_blob_size,
                    crop_v2_disk_radius=self.crop_v2_disk_radius,
                )
                self.img_dict[rel_path] = sig_image
        
        if pair_path is None or not os.path.exists(pair_path):
            print(f"⚠️  Pair file not found at {pair_path}")
            print(f"    Make sure to run generate_gdps_pairs() first to create pair files")
            self.labels = []
            self.pairs = []
        else:
            with open(pair_path, 'r') as f:
                lines = f.readlines()

            self.labels = []
            self.pairs = []
            seen_pairs = set()
            duplicate_pairs = 0
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split('\t')  # Use tab as delimiter to handle filenames with spaces
                if len(parts) < 3:
                    if len(parts) > 0:  # Skip empty lines
                        print(f"⚠️  Line {line_num} has {len(parts)} columns (expected 3): {line.strip()[:80]}")
                    continue
                
                try:
                    refer = parts[0].replace('\\', '/').replace('/', os.sep)
                    test  = parts[1].replace('\\', '/').replace('/', os.sep)
                    label = int(parts[2])

                    pair_key = _pair_key(refer, test, label)
                    if pair_key in seen_pairs:
                        duplicate_pairs += 1
                        continue
                    seen_pairs.add(pair_key)
                    
                    # Paths from pair file are already relative to image_root
                    if refer in self.img_dict and test in self.img_dict:
                        self.pairs.append((refer, test))
                        self.labels.append(label)
                except ValueError as e:
                    print(f"❌ Error parsing line {line_num}: {line.strip()}")
                    print(f"   columns: {parts}")
                    print(f"   Error: {e}")
                    raise

            writer_counts = Counter()
            unknown_writer_pairs = 0
            for refer, test in self.pairs:
                writer_id = _extract_writer_id_from_relpath(refer)
                if writer_id is None:
                    writer_id = _extract_writer_id_from_relpath(test)
                if writer_id is None:
                    unknown_writer_pairs += 1
                else:
                    writer_counts[writer_id] += 1

            writer_values = list(writer_counts.values())
            writer_min = min(writer_values) if writer_values else 0
            writer_max = max(writer_values) if writer_values else 0
            writer_mean = float(sum(writer_values) / len(writer_values)) if writer_values else 0.0

            if duplicate_pairs > 0:
                print(f"⚠️  GPDS pair file duplicates removed in loader: {duplicate_pairs}")
            print(
                "📊 GPDS writer coverage in loaded pairs: "
                f"writers={len(writer_counts)} unknown_writer_pairs={unknown_writer_pairs} "
                f"writer_pairs[min/mean/max]={writer_min}/{writer_mean:.1f}/{writer_max}"
            )

        if self.multi_ref_enabled:
            raw_pairs = [(r, t, l) for (r, t), l in zip(self.pairs, self.labels)]
            expanded_pairs, summary = _expand_pairs_for_multi_ref_eval(
                raw_pairs,
                num_refs=self.multi_ref_num_refs,
                ref_seed=self.multi_ref_seed,
                dataset_name='GPDS',
            )
            self.pairs = [(r, t) for r, t, _ in expanded_pairs]
            self.labels = [int(l) for _, _, l in expanded_pairs]
            print(
                "🔁 GPDS multi-ref test enabled: "
                f"writers={summary['writers']} refs={summary['num_refs']} "
                f"queries={summary['num_queries']} pairs={summary['num_pairs']}"
            )
        
        self.train = train
        print(f"Kaggle GDPS: Loaded {len(self.labels)} pairs (pre-cached in RAM)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        refer_path, test_path = self.pairs[index]
        refer_img = self.img_dict[refer_path].copy()
        test_img = self.img_dict[test_path].copy()

        if self.train:
            refer_img = self.pre_tensor_augment(refer_img)
            test_img = self.pre_tensor_augment(test_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)

        return image_pair, torch.tensor([[self.labels[index]]])
