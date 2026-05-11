"""Cat face detection, cropping, and .npy export from the Cat Dataset .cat annotations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMG_SIZE = 64
BBOX_PAD_RATIO = 0.35


@dataclass(frozen=True)
class CatAnnotation:
    """9 facial keypoints parsed from a .cat file: (x, y) pairs."""

    left_eye: tuple[int, int]
    right_eye: tuple[int, int]
    mouth: tuple[int, int]
    left_ear_1: tuple[int, int]
    left_ear_2: tuple[int, int]
    left_ear_3: tuple[int, int]
    right_ear_1: tuple[int, int]
    right_ear_2: tuple[int, int]
    right_ear_3: tuple[int, int]

    @property
    def all_points(self) -> list[tuple[int, int]]:
        return [
            self.left_eye,
            self.right_eye,
            self.mouth,
            self.left_ear_1,
            self.left_ear_2,
            self.left_ear_3,
            self.right_ear_1,
            self.right_ear_2,
            self.right_ear_3,
        ]


def parse_cat_annotation(cat_path: Path) -> CatAnnotation:
    """Parse a .cat annotation file → CatAnnotation with 9 keypoints."""
    text = cat_path.read_text().strip()
    tokens = text.split()
    n_points = int(tokens[0])
    if n_points != 9:
        msg = f"Expected 9 keypoints, got {n_points} in {cat_path}"
        raise ValueError(msg)
    coords = [int(t) for t in tokens[1:]]
    if len(coords) != 18:
        msg = f"Expected 18 coordinate values, got {len(coords)} in {cat_path}"
        raise ValueError(msg)
    points = [(coords[i], coords[i + 1]) for i in range(0, 18, 2)]
    return CatAnnotation(
        left_eye=points[0],
        right_eye=points[1],
        mouth=points[2],
        left_ear_1=points[3],
        left_ear_2=points[4],
        left_ear_3=points[5],
        right_ear_1=points[6],
        right_ear_2=points[7],
        right_ear_3=points[8],
    )


def _keypoints_bbox(
    annotation: CatAnnotation, img_w: int, img_h: int, pad_ratio: float = BBOX_PAD_RATIO
) -> tuple[int, int, int, int]:
    """Compute padded square bounding box around all keypoints, clamped to image bounds."""
    xs = [p[0] for p in annotation.all_points]
    ys = [p[1] for p in annotation.all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min
    side = max(w, h)
    pad = int(side * pad_ratio)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = (side + 2 * pad) // 2

    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(img_w, cx + half)
    bottom = min(img_h, cy + half)

    return left, top, right, bottom


def crop_cat_face(
    img: Image.Image,
    annotation: CatAnnotation,
    size: int = IMG_SIZE,
) -> Image.Image:
    """Crop cat face from image using keypoint bbox, resize to `size` x `size`."""
    bbox = _keypoints_bbox(annotation, img.width, img.height)
    cropped = img.crop(bbox)
    return cropped.resize((size, size), Image.LANCZOS)


def process_dataset(
    raw_dir: Path,
    output_dir: Path,
    size: int = IMG_SIZE,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Process all images in raw_dir → cropped .npy arrays in output_dir.

    Returns dict with counts: {"train": N, "val": M, "skipped": K}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p.with_suffix(p.suffix + ".cat").exists()
    )

    if not image_paths:
        msg = f"No image+.cat pairs found in {raw_dir}"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(image_paths))
    n_val = max(1, int(len(image_paths) * val_fraction))
    val_indices = set(indices[:n_val].tolist())

    train_faces: list[NDArray[np.uint8]] = []
    val_faces: list[NDArray[np.uint8]] = []
    skipped = 0

    for i, img_path in enumerate(tqdm(image_paths, desc="Cropping faces", unit="img")):
        cat_path = img_path.with_suffix(img_path.suffix + ".cat")
        try:
            annotation = parse_cat_annotation(cat_path)
            img = Image.open(img_path).convert("RGB")
            face = crop_cat_face(img, annotation, size=size)
            arr = np.array(face, dtype=np.uint8)

            if i in val_indices:
                val_faces.append(arr)
            else:
                train_faces.append(arr)
        except Exception:
            logger.warning("Skipping %s", img_path.name, exc_info=True)
            skipped += 1
            continue

    train_arr = np.stack(train_faces)
    val_arr = np.stack(val_faces)

    np.save(output_dir / "train.npy", train_arr)
    np.save(output_dir / "val.npy", val_arr)

    logger.info(
        "Saved %d train, %d val faces (%d skipped) → %s",
        len(train_faces),
        len(val_faces),
        skipped,
        output_dir,
    )
    return {"train": len(train_faces), "val": len(val_faces), "skipped": skipped}
