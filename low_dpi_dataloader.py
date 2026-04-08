from dataclasses import dataclass
from typing import Any, Dict
import os

import torch
import numpy as np
from PIL import Image, ImageFilter
import pywt
import torchvision.transforms as transforms

from module.preprocess import normalize_image
from WI_sig_dataloader_kaggle import (
    SigDataset_CEDAR_Kaggle as _BaseCEDARKaggle,
    SigDataset_CEDAR_Kaggle_Lite as _BaseCEDARKaggleLite,
    SigDataset_GDPS_Kaggle as _BaseGDPSKaggle,
)


@dataclass
class LowDPIConfig:
    pair_profile: str = "hq_hq"
    target_max_side: int = 200
    sharpen_enabled: bool = True
    sharpen_mode: str = "unsharp"
    sharpen_alpha: float = 0.6
    sharpen_sigma: float = 1.0
    wavelet_name: str = "db2"
    wavelet_detail_gain: float = 1.3
    wavelet_hh_gain: float = 1.0
    rotation_degrees: float = 10.0


def build_lowdpi_config(opt: Any = None) -> LowDPIConfig:
    if opt is None:
        return LowDPIConfig()

    return LowDPIConfig(
        pair_profile=getattr(opt, "pair_profile", "hq_hq"),
        target_max_side=int(getattr(opt, "lowdpi_target_max_side", 200)),
        sharpen_enabled=bool(getattr(opt, "lowdpi_enable_sharpen", True)),
        sharpen_mode=str(getattr(opt, "lowdpi_sharpen_mode", "unsharp")),
        sharpen_alpha=float(getattr(opt, "lowdpi_sharpen_alpha", 0.6)),
        sharpen_sigma=float(getattr(opt, "lowdpi_sharpen_sigma", 1.0)),
        wavelet_name=str(getattr(opt, "lowdpi_wavelet_name", "db2")),
        wavelet_detail_gain=float(getattr(opt, "lowdpi_wavelet_detail_gain", 1.3)),
        wavelet_hh_gain=float(getattr(opt, "lowdpi_wavelet_hh_gain", 1.0)),
        rotation_degrees=float(getattr(opt, "lowdpi_rotation_degrees", 10.0)),
    )


def is_hq_lowdpi_profile(cfg: LowDPIConfig) -> bool:
    return str(cfg.pair_profile).strip().lower() == "hq_lowdpi70"


def summarize_lowdpi_config(cfg_or_opt: Any) -> Dict[str, Any]:
    cfg = cfg_or_opt if isinstance(cfg_or_opt, LowDPIConfig) else build_lowdpi_config(cfg_or_opt)
    return {
        "pair_profile": cfg.pair_profile,
        "target_max_side": cfg.target_max_side,
        "sharpen_enabled": cfg.sharpen_enabled,
        "sharpen_mode": cfg.sharpen_mode,
        "sharpen_alpha": cfg.sharpen_alpha,
        "sharpen_sigma": cfg.sharpen_sigma,
        "wavelet_name": cfg.wavelet_name,
        "wavelet_detail_gain": cfg.wavelet_detail_gain,
        "wavelet_hh_gain": cfg.wavelet_hh_gain,
        "rotation_degrees": cfg.rotation_degrees,
    }


def _box_downsample_preserve_aspect(img: Image.Image, target_max_side: int) -> Image.Image:
    if target_max_side <= 0:
        raise ValueError("target_max_side must be > 0")

    w, h = img.size
    max_side = max(w, h)
    if max_side <= target_max_side:
        return img.copy()

    scale = float(target_max_side) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BOX)


def _apply_detail_transfer_sharpen(img: Image.Image, alpha: float, sigma: float) -> Image.Image:
    base = np.asarray(img, dtype=np.float32)
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=float(sigma))), dtype=np.float32)
    detail = base - blurred
    sharpened = np.clip(base + float(alpha) * detail, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(sharpened)


def _apply_wavelet_sharpen(
    img: Image.Image,
    wavelet_name: str,
    detail_gain: float,
    hh_gain: float,
) -> Image.Image:
    base = np.asarray(img, dtype=np.float32)
    ll, (lh, hl, hh) = pywt.dwt2(base, wavelet_name)

    lh = lh * float(detail_gain)
    hl = hl * float(detail_gain)
    hh = hh * float(hh_gain)

    reconstructed = pywt.idwt2((ll, (lh, hl, hh)), wavelet_name)
    # idwt2 can return +1 on odd dimensions depending on wavelet/filter length.
    reconstructed = reconstructed[: base.shape[0], : base.shape[1]]

    sharpened = np.clip(reconstructed, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(sharpened)


def build_lowdpi_test_image_from_path(img_path: str, cfg: LowDPIConfig) -> Image.Image:
    with Image.open(img_path) as _img:
        raw_img = _img.convert("L")

    # Realistic ordering: downsample first, then run noise removal/cropping on the low-DPI signal.
    downsampled = _box_downsample_preserve_aspect(raw_img, cfg.target_max_side)
    downsampled_np = np.asarray(downsampled, dtype=np.uint8)
    _, cropped = normalize_image(downsampled_np, downsampled_np.shape)

    out = Image.fromarray(cropped)
    if cfg.sharpen_enabled:
        sharpen_mode = str(cfg.sharpen_mode).strip().lower()
        if sharpen_mode == "unsharp":
            out = _apply_detail_transfer_sharpen(out, alpha=cfg.sharpen_alpha, sigma=cfg.sharpen_sigma)
        elif sharpen_mode == "wavelet":
            out = _apply_wavelet_sharpen(
                out,
                wavelet_name=cfg.wavelet_name,
                detail_gain=cfg.wavelet_detail_gain,
                hh_gain=cfg.wavelet_hh_gain,
            )
        else:
            raise ValueError(
                "Unsupported lowdpi_sharpen_mode='{}' (expected 'unsharp' or 'wavelet')".format(cfg.sharpen_mode)
            )

    return out


def _build_lowdpi_profile_augs(train: bool, cfg: LowDPIConfig):
    if not train:
        return None, None
    rotation_degrees = max(0.0, float(cfg.rotation_degrees))
    if rotation_degrees == 0.0:
        return None, None
    # LowDPI profile policy: rotation only, no blur, no random erasing.
    # Rotation is applied to reference branch only in low-DPI mode.
    pre_tensor_augment = transforms.Compose([
        transforms.RandomRotation(
            degrees=(-rotation_degrees, rotation_degrees),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=255,
        )
    ])
    return pre_tensor_augment, None


class SigDataset_CEDAR_Kaggle(_BaseCEDARKaggle):
    """LowDPI-aware wrapper over WI CEDAR Kaggle loader.

    - HQ/HQ behavior is inherited unchanged from WI loader.
    - HQ/LowDPI behavior is enabled when pair_profile='hq_lowdpi70'.
    """

    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        super().__init__(opt, image_root, pair_root, train=train, image_size=image_size, mode=mode, pair_filename=pair_filename)
        self.lowdpi_cfg = build_lowdpi_config(opt)
        self.use_lowdpi_profile = is_hq_lowdpi_profile(self.lowdpi_cfg)
        if self.use_lowdpi_profile and self.train:
            self.pre_tensor_augment, self.post_tensor_augment = _build_lowdpi_profile_augs(self.train, self.lowdpi_cfg)

    def __getitem__(self, index):
        if not self.use_lowdpi_profile:
            return super().__getitem__(index)

        refer_path, test_path = self.pairs[index]
        refer_img = self.img_dict[refer_path].copy()
        test_img = build_lowdpi_test_image_from_path(test_path, self.lowdpi_cfg)

        if self.train and self.pre_tensor_augment is not None:
            refer_img = self.pre_tensor_augment(refer_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train and self.post_tensor_augment is not None:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)
        return image_pair, torch.tensor([[self.labels[index]]])


class SigDataset_CEDAR_Kaggle_Lite(_BaseCEDARKaggleLite):
    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        super().__init__(opt, image_root, pair_root, train=train, image_size=image_size, mode=mode, pair_filename=pair_filename)
        self.lowdpi_cfg = build_lowdpi_config(opt)
        self.use_lowdpi_profile = is_hq_lowdpi_profile(self.lowdpi_cfg)
        if self.use_lowdpi_profile and self.train:
            self.pre_tensor_augment, self.post_tensor_augment = _build_lowdpi_profile_augs(self.train, self.lowdpi_cfg)

    def __getitem__(self, index):
        if not self.use_lowdpi_profile:
            return super().__getitem__(index)

        refer_rel, test_rel, label_int = self.pairs[index]
        refer_img = self.img_dict[refer_rel].copy()
        test_abs_path = os.path.join(self.image_root, test_rel)
        test_img = build_lowdpi_test_image_from_path(test_abs_path, self.lowdpi_cfg)

        if self.train and self.pre_tensor_augment is not None:
            refer_img = self.pre_tensor_augment(refer_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train and self.post_tensor_augment is not None:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)
        return image_pair, torch.tensor([[label_int]])


class SigDataset_GDPS_Kaggle(_BaseGDPSKaggle):
    """LowDPI-aware wrapper over WI GPDS Kaggle loader."""

    def __init__(self, opt, image_root, pair_root, train=True, image_size=256, mode='normalized', pair_filename=None):
        super().__init__(opt, image_root, pair_root, train=train, image_size=image_size, mode=mode, pair_filename=pair_filename)
        self.lowdpi_cfg = build_lowdpi_config(opt)
        self.use_lowdpi_profile = is_hq_lowdpi_profile(self.lowdpi_cfg)
        if self.use_lowdpi_profile and self.train:
            self.pre_tensor_augment, self.post_tensor_augment = _build_lowdpi_profile_augs(self.train, self.lowdpi_cfg)

    def __getitem__(self, index):
        if not self.use_lowdpi_profile:
            return super().__getitem__(index)

        refer_path, test_path = self.pairs[index]
        refer_img = self.img_dict[refer_path].copy()
        test_abs_path = os.path.join(self.image_root, test_path)
        test_img = build_lowdpi_test_image_from_path(test_abs_path, self.lowdpi_cfg)

        if self.train and self.pre_tensor_augment is not None:
            refer_img = self.pre_tensor_augment(refer_img)

        refer_img = self.basic_transforms(refer_img)
        test_img = self.basic_transforms(test_img)

        if self.train and self.post_tensor_augment is not None:
            refer_img = self.post_tensor_augment(refer_img)
            test_img = self.post_tensor_augment(test_img)

        image_pair = torch.cat((refer_img, test_img), dim=0)
        return image_pair, torch.tensor([[self.labels[index]]])
