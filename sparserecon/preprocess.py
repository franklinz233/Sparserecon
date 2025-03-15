from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import rembg
import torch
from PIL import Image

os.environ["OMP_NUM_THREADS"] = "10"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BLIP2:
    def __init__(self, device: str = "cuda"):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
        ).to(device)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> str:
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove backgrounds and recenter input images.")
    parser.add_argument("path", type=str, help="Path to an image or a directory of images.")
    parser.add_argument("--model", default="u2net", type=str, help="rembg model name.")
    parser.add_argument("--size", default=256, type=int, help="Output resolution.")
    parser.add_argument("--border_ratio", default=0.2, type=float, help="Output border ratio.")
    parser.add_argument("--no-recenter", dest="recenter", action="store_false", help="Skip recentering.")
    parser.set_defaults(recenter=True)
    return parser


def list_images(path: Path) -> tuple[list[Path], Path]:
    if path.is_dir():
        print(f"[INFO] processing directory {path}...")
        files = [
            Path(file)
            for file in glob.glob(str(path / "*"))
            if Path(file).is_file() and Path(file).suffix.lower() in IMAGE_EXTENSIONS
        ]
        return sorted(files), path

    if path.is_file():
        return [path], path.parent

    raise FileNotFoundError(f"Input path does not exist: {path}")


def recenter_image(
    image: np.ndarray,
    carved_image: np.ndarray,
    mask: np.ndarray,
    size: int,
    border_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.nonzero(mask)
    if len(coords[0]) == 0 or len(coords[1]) == 0:
        raise ValueError("Background removal produced an empty foreground mask.")

    final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = max(1, int(h * scale))
    w2 = max(1, int(w * scale))
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        carved_image[x_min:x_max, y_min:y_max],
        (w2, h2),
        interpolation=cv2.INTER_AREA,
    )

    xc = (x_min + x_max) // 2
    yc = (y_min + y_max) // 2
    crop_radius = int(max(h, w) / (1 - border_ratio)) // 2
    x_min, x_max = xc - crop_radius, xc + crop_radius
    y_min, y_max = yc - crop_radius, yc + crop_radius
    image_rgb = image[..., :3]
    height, width = image_rgb.shape[:2]

    canvas = np.zeros(
        (max(height, x_max) - min(0, x_min), max(width, y_max) - min(0, y_min), 3),
        dtype=image_rgb.dtype,
    )
    y_offset = -min(0, y_min)
    x_offset = -min(0, x_min)
    canvas[x_offset : x_offset + height, y_offset : y_offset + width] = image_rgb

    roi = canvas[x_offset + x_min : x_offset + x_max, y_offset + y_min : y_offset + y_max]
    final_rgb = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return final_rgb, final_rgba


def process_image(file: Path, out_dir: Path, session: rembg.sessions.BaseSession, opt: argparse.Namespace) -> None:
    out_base = file.stem
    out_rgba = out_dir / "processed" / f"{out_base}_rgba.png"
    out_rgb = out_dir / "source" / f"{out_base}.png"

    print(f"[INFO] loading image {file}...")
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {file}")

    print("[INFO] background removal...")
    carved_image = rembg.remove(image, session=session)
    mask = carved_image[..., -1] > 0

    if opt.recenter:
        print("[INFO] recenter...")
        final_rgb, final_rgba = recenter_image(image, carved_image, mask, opt.size, opt.border_ratio)
    else:
        final_rgba = carved_image
        alpha = (final_rgba[..., 3:] > 0).astype(np.float32)
        final_rgb = (final_rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)

    cv2.imwrite(str(out_rgba), final_rgba)
    cv2.imwrite(str(out_rgb), final_rgb)


def preprocess_dataset(opt: argparse.Namespace) -> None:
    files, out_dir = list_images(Path(opt.path))
    if not files:
        raise ValueError(f"No supported images found in {opt.path}")

    (out_dir / "processed").mkdir(parents=True, exist_ok=True)
    (out_dir / "source").mkdir(parents=True, exist_ok=True)

    session = rembg.new_session(model_name=opt.model)
    for file in files:
        process_image(file, out_dir, session, opt)


def main() -> None:
    preprocess_dataset(build_parser().parse_args())
