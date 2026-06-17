"""
Patch splitter pipeline.

Reads HR/LR pairs from cache/degradation_manifest.json,
slices both into aligned patch pairs,
saves each pair as a single .pt file: {"hr": tensor, "lr": tensor}

Output layout:
    cache/patches/
        <img_stem>_patches/
            1.pt
            2.pt
            ...
    cache/patches_manifest.json  ← consumed directly by the DataLoader

Worker count: cpu_count - 2  (one free, one for orchestrator)
"""

import json
import logging
import multiprocessing
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve
from torchvision.transforms.functional import to_tensor

DEGRADATION_MANIFEST_PATH: Path = Path("cache/degradation_manifest.json")
PATCHES_DIR: Path = Path("cache/patches")
PATCHES_MANIFEST_PATH: Path = Path("cache/patches_manifest.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PatchConfig:
    hr_patch_size: int = 128
    stride: int = 64                       # 50% overlap — set equal to hr_patch_size for no overlap
    min_variance_threshold: float = 75.5   # patches below this are too flat to learn from


def load_valid_pairs_from_manifest(manifest_path: Path) -> list[tuple[Path, Path]]:
    manifest: dict = json.loads(manifest_path.read_text())
    valid_pairs: list[tuple[Path, Path]] = [
        (Path(entry["source"]), Path(entry["output"]))
        for entry in manifest["entries"]
        if entry.get("output") is not None and not entry.get("error")
    ]
    logger.info("Loaded %d valid HR/LR pairs from manifest", len(valid_pairs))
    return valid_pairs


def _patch_has_enough_texture(patch_array: np.ndarray, threshold: float) -> bool:
    grayscale: np.ndarray = (
        0.299 * patch_array[:, :, 0]
        + 0.587 * patch_array[:, :, 1]
        + 0.114 * patch_array[:, :, 2]
    )
    laplacian_kernel: np.ndarray = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float32)
    response: np.ndarray = convolve(grayscale.astype(np.float32), laplacian_kernel, mode="reflect")
    variance: float = float(response.var())
    return variance >= threshold


def _extract_patch_pairs(
    hr_array: np.ndarray,
    lr_array: np.ndarray,
    config: PatchConfig,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    image_height: int = hr_array.shape[0]
    image_width: int = hr_array.shape[1]
    patch_size: int = config.hr_patch_size

    if image_height < patch_size or image_width < patch_size:
        return []

    patch_pairs: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []

    top: int = 0
    while top + patch_size <= image_height:
        left: int = 0
        while left + patch_size <= image_width:
            hr_patch: np.ndarray = hr_array[top:top + patch_size, left:left + patch_size]
            lr_patch: np.ndarray = lr_array[top:top + patch_size, left:left + patch_size]

            if _patch_has_enough_texture(hr_patch, config.min_variance_threshold):
                patch_pairs.append((hr_patch, lr_patch, (top, left)))

            left += config.stride
        top += config.stride

    return patch_pairs



def _save_patch_pair(
    hr_patch: np.ndarray,
    lr_patch: np.ndarray,
    output_path: Path,
) -> None:
    # to_tensor converts HWC uint8 [0,255] → CHW float32 [0,1]
    hr_tensor: torch.Tensor = to_tensor(hr_patch)
    lr_tensor: torch.Tensor = to_tensor(lr_patch)
    torch.save({"hr": hr_tensor, "lr": lr_tensor}, output_path)



def _process_one_image_pair(
    args: tuple[Path, Path, Path, PatchConfig],
) -> dict:
    hr_path, lr_path, patches_root_dir, config = args

    image_stem: str = hr_path.stem
    image_patch_dir: Path = patches_root_dir / f"{image_stem}_patches"

    if image_patch_dir.exists() and any(image_patch_dir.iterdir()):
        existing_patches: list[Path] = sorted(
            image_patch_dir.glob("*.pt"),
            key=lambda patch_path: int(patch_path.stem),
        )
        return {
            "source_hr": str(hr_path),
            "source_lr": str(lr_path),
            "patch_dir": str(image_patch_dir),
            "patch_count": len(existing_patches),
            "patch_paths": [str(patch_path) for patch_path in existing_patches],
            "skipped": True,
        }

    try:
        hr_image: Image.Image = Image.open(hr_path).convert("RGB")
        lr_image: Image.Image = Image.open(lr_path).convert("RGB")

        hr_array: np.ndarray = np.array(hr_image, dtype=np.uint8)
        lr_array: np.ndarray = np.array(lr_image, dtype=np.uint8)

        if hr_array.shape != lr_array.shape:
            return {
                "source_hr": str(hr_path),
                "source_lr": str(lr_path),
                "patch_dir": None,
                "patch_count": 0,
                "patch_paths": [],
                "skipped": False,
                "error": f"shape mismatch — HR {hr_array.shape} vs LR {lr_array.shape}",
            }

        patch_pairs: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = (
            _extract_patch_pairs(hr_array, lr_array, config)
        )

        if not patch_pairs:
            return {
                "source_hr": str(hr_path),
                "source_lr": str(lr_path),
                "patch_dir": None,
                "patch_count": 0,
                "patch_paths": [],
                "skipped": False,
                "error": "no patches passed variance filter — image too small or too flat",
            }

        image_patch_dir.mkdir(parents=True, exist_ok=True)

        saved_patch_paths: list[str] = []
        for patch_index, (hr_patch, lr_patch, _coords) in enumerate(patch_pairs, start=1):
            patch_output_path: Path = image_patch_dir / f"{patch_index}.pt"
            _save_patch_pair(hr_patch, lr_patch, patch_output_path)
            saved_patch_paths.append(str(patch_output_path))

        return {
            "source_hr": str(hr_path),
            "source_lr": str(lr_path),
            "patch_dir": str(image_patch_dir),
            "patch_count": len(saved_patch_paths),
            "patch_paths": saved_patch_paths,
            "skipped": False,
        }

    except Exception as error:
        logger.error("Failed on %s — %s", hr_path.name, error)
        return {
            "source_hr": str(hr_path),
            "source_lr": str(lr_path),
            "patch_dir": None,
            "patch_count": 0,
            "patch_paths": [],
            "skipped": False,
            "error": str(error),
        }



def _write_patches_manifest(
    all_results: list[dict],
    config: PatchConfig,
    manifest_path: Path,
) -> None:
    successful_results: list[dict] = [
        result for result in all_results
        if not result.get("error") and result["patch_count"] > 0
    ]
    error_results: list[dict] = [
        result for result in all_results
        if result.get("error")
    ]
    skipped_results: list[dict] = [
        result for result in all_results
        if result.get("skipped")
    ]

    total_patches: int = sum(result["patch_count"] for result in all_results if not result.get("error"))

    all_patch_paths: list[str] = [
        patch_path
        for result in all_results
        if not result.get("error")
        for patch_path in result["patch_paths"]
    ]

    manifest_data: dict = {
        "patch_config": asdict(config),
        "summary": {
            "total_source_images": len(all_results),
            "successfully_patched": len(successful_results),
            "skipped_already_done": len(skipped_results),
            "errors": len(error_results),
            "total_patches": total_patches,
        },
        "error_log": [
            {"source_hr": result["source_hr"], "reason": result["error"]}
            for result in error_results
        ],
        "patches": all_patch_paths,
        "per_image": [
            {
                "source_hr": result["source_hr"],
                "source_lr": result["source_lr"],
                "patch_dir": result["patch_dir"],
                "patch_count": result["patch_count"],
            }
            for result in all_results
            if not result.get("error") and result["patch_count"] > 0
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    logger.info(
        "Patches manifest written → %s | total patches: %d",
        manifest_path,
        total_patches,
    )



def _run_with_progress(
    task_args: list[tuple[Path, Path, Path, PatchConfig]],
    worker_count: int,
) -> list[dict]:
    total_tasks: int = len(task_args)
    completed_count: int = 0
    total_patches_so_far: int = 0
    error_count: int = 0
    start_time: float = time.monotonic()
    all_results: list[dict] = []

    with multiprocessing.Pool(processes=worker_count) as pool:
        for result in pool.imap_unordered(_process_one_image_pair, task_args, chunksize=8):
            all_results.append(result)
            completed_count += 1

            if result.get("error"):
                error_count += 1
            else:
                total_patches_so_far += result["patch_count"]

            if completed_count % 200 == 0 or completed_count == total_tasks:
                elapsed: float = time.monotonic() - start_time
                rate: float = completed_count / elapsed if elapsed > 0 else 0.0
                remaining_seconds: float = (
                    (total_tasks - completed_count) / rate if rate > 0 else 0.0
                )
                logger.info(
                    "Progress: %d/%d | patches so far: %d | errors: %d | "
                    "%.1f img/s | ETA: %.0fs",
                    completed_count, total_tasks,
                    total_patches_so_far, error_count,
                    rate, remaining_seconds,
                )

    return all_results


def main() -> None:
    if not DEGRADATION_MANIFEST_PATH.exists():
        logger.error(
            "Degradation manifest not found at %s — run degrade_images.py first",
            DEGRADATION_MANIFEST_PATH,
        )
        return

    total_cores: int = multiprocessing.cpu_count()
    worker_count: int = max(1, total_cores - 2)
    logger.info("CPU cores available: %d | workers: %d", total_cores, worker_count)

    config: PatchConfig = PatchConfig()
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    valid_pairs: list[tuple[Path, Path]] = load_valid_pairs_from_manifest(
        DEGRADATION_MANIFEST_PATH
    )

    if not valid_pairs:
        logger.error("No valid HR/LR pairs found in manifest — nothing to process")
        return

    task_args: list[tuple[Path, Path, Path, PatchConfig]] = [
        (hr_path, lr_path, PATCHES_DIR, config)
        for hr_path, lr_path in valid_pairs
    ]

    logger.info("Starting patch extraction for %d image pairs", len(task_args))
    all_results: list[dict] = _run_with_progress(task_args, worker_count)

    _write_patches_manifest(all_results, config, PATCHES_MANIFEST_PATH)

    successful_count: int = sum(
        1 for result in all_results
        if not result.get("error") and result["patch_count"] > 0
    )
    total_patches: int = sum(
        result["patch_count"] for result in all_results
        if not result.get("error")
    )
    logger.info(
        "Done | %d/%d images patched | %d total patches | manifest → %s",
        successful_count, len(all_results),
        total_patches,
        PATCHES_MANIFEST_PATH,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
