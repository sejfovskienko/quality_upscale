"""
Synthetic degradation pipeline.
Reads all images from data/, applies randomized degradation per image,
saves results to cache/degradated_images/.

Worker count: cpu_count - 2  (one free, one for orchestrator)
"""

import io
import json
import logging
import multiprocessing
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import convolve


SOURCE_DIR: Path = Path("data")
OUTPUT_DIR: Path = Path("cache/degradated_images")
MANIFEST_PATH: Path = Path("cache/degradation_manifest.json")
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

@dataclass
class DegradationRangeConfig:
    gaussian_blur_sigma_range: tuple[float, float] = (0.2, 3.0)
    motion_blur_kernel_range: tuple[int, int] = (7, 21)
    blur_type_probabilities: dict[str, float] = field(default_factory=lambda: {
        "gaussian": 0.6,
        "motion":   0.2,
        "none":     0.2,
    })

    gaussian_noise_sigma_range: tuple[float, float] = (1.0, 30.0)
    poisson_noise_scale_range:  tuple[float, float] = (0.05, 2.0)
    noise_type_probabilities: dict[str, float] = field(default_factory=lambda: {
        "gaussian": 0.5,
        "poisson":  0.2,
        "none":     0.3,
    })

    jpeg_quality_range: tuple[int, int] = (30, 95)
    jpeg_apply_probability: float = 0.8

    downscale_mode_probabilities: dict[str, float] = field(default_factory=lambda: {
        "bicubic":  0.4,
        "bilinear": 0.3,
        "lanczos":  0.2,
        "area":     0.1,
    })


@dataclass
class SampledDegradationParams:
    blur_type:             str
    blur_sigma:            float | None
    motion_blur_kernel:    int   | None
    noise_type:            str
    gaussian_noise_sigma:  float | None
    poisson_noise_scale:   float | None
    apply_jpeg:            bool
    jpeg_quality:          int   | None
    downscale_mode:        str


def sample_degradation_params(config: DegradationRangeConfig) -> SampledDegradationParams:
    blur_type: str = random.choices(
        list(config.blur_type_probabilities.keys()),
        weights=list(config.blur_type_probabilities.values()),
    )[0]

    blur_sigma: float | None = (
        random.uniform(*config.gaussian_blur_sigma_range)
        if blur_type == "gaussian" else None
    )

    motion_blur_kernel: int | None = None
    if blur_type == "motion":
        raw: int = random.randint(*config.motion_blur_kernel_range)
        motion_blur_kernel = raw if raw % 2 != 0 else raw + 1

    noise_type: str = random.choices(
        list(config.noise_type_probabilities.keys()),
        weights=list(config.noise_type_probabilities.values()),
    )[0]

    gaussian_noise_sigma: float | None = (
        random.uniform(*config.gaussian_noise_sigma_range)
        if noise_type == "gaussian" else None
    )

    poisson_noise_scale: float | None = (
        random.uniform(*config.poisson_noise_scale_range)
        if noise_type == "poisson" else None
    )

    apply_jpeg: bool = random.random() < config.jpeg_apply_probability
    jpeg_quality: int | None = (
        random.randint(*config.jpeg_quality_range)
        if apply_jpeg else None
    )

    downscale_mode: str = random.choices(
        list(config.downscale_mode_probabilities.keys()),
        weights=list(config.downscale_mode_probabilities.values()),
    )[0]

    return SampledDegradationParams(
        blur_type=blur_type,
        blur_sigma=blur_sigma,
        motion_blur_kernel=motion_blur_kernel,
        noise_type=noise_type,
        gaussian_noise_sigma=gaussian_noise_sigma,
        poisson_noise_scale=poisson_noise_scale,
        apply_jpeg=apply_jpeg,
        jpeg_quality=jpeg_quality,
        downscale_mode=downscale_mode,
    )


def _apply_motion_blur(image_array: np.ndarray, kernel_size: int) -> np.ndarray:
    """Applies motion blur to the input image"""

    kernel: np.ndarray = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    blurred: np.ndarray = np.stack(
        [convolve(image_array[:, :, channel], kernel, mode="reflect") for channel in range(3)],
        axis=2,
    )
    return np.clip(blurred, 0.0, 255.0)


def apply_degradation(
    source_image: Image.Image,
    params: SampledDegradationParams,
) -> Image.Image:
    """"""
    result: Image.Image = source_image.convert("RGB")
    image_array: np.ndarray = np.array(result, dtype=np.float32)

    # --- blur ---
    if params.blur_type == "gaussian" and params.blur_sigma is not None:
        result = result.filter(ImageFilter.GaussianBlur(radius=params.blur_sigma))
        image_array = np.array(result, dtype=np.float32)

    elif params.blur_type == "motion" and params.motion_blur_kernel is not None:
        image_array = _apply_motion_blur(image_array, params.motion_blur_kernel)

    # --- noise ---
    if params.noise_type == "gaussian" and params.gaussian_noise_sigma is not None:
        noise: np.ndarray = np.random.normal(0.0, params.gaussian_noise_sigma, image_array.shape)
        image_array = np.clip(image_array + noise, 0.0, 255.0)

    elif params.noise_type == "poisson" and params.poisson_noise_scale is not None:
        scaled: np.ndarray = image_array * params.poisson_noise_scale
        poisson_noise: np.ndarray = np.random.poisson(scaled).astype(np.float32)
        image_array = np.clip(poisson_noise / (params.poisson_noise_scale + 1e-8), 0.0, 255.0)

    result = Image.fromarray(image_array.astype(np.uint8))

    # --- jpeg compression ---
    if params.apply_jpeg and params.jpeg_quality is not None:
        buffer: io.BytesIO = io.BytesIO()
        result.save(buffer, format="JPEG", quality=params.jpeg_quality)
        buffer.seek(0)
        result = Image.open(buffer).copy()

    return result


def _process_one_image(args: tuple[Path, Path, DegradationRangeConfig]) -> dict | None:
    source_path, output_dir, range_config = args

    output_path: Path = output_dir / source_path.name

    if output_path.exists():
        return {
            "source": str(source_path),
            "output": str(output_path),
            "skipped": True,
            "params": None,
        }

    try:
        source_image: Image.Image = Image.open(source_path)
        sampled_params: SampledDegradationParams = sample_degradation_params(range_config)
        degraded_image: Image.Image = apply_degradation(source_image, sampled_params)

        # save as PNG to avoid adding extra JPEG compression artifacts on top
        # of the ones already applied by the degradation pipeline
        output_png_path: Path = output_dir / (source_path.stem + ".png")
        degraded_image.save(output_png_path, format="PNG")

        return {
            "source": str(source_path),
            "output": str(output_png_path),
            "skipped": False,
            "params": asdict(sampled_params),
        }

    except Exception as error:
        logger.error("Failed on %s — %s", source_path.name, error)
        return {
            "source": str(source_path),
            "output": None,
            "skipped": False,
            "error": str(error),
            "params": None,
        }


def _collect_results(
    result_queue: multiprocessing.Queue,
    total_images: int,
    manifest_path: Path,
) -> None:
    manifest_entries: list[dict] = []
    processed_count: int = 0
    error_count: int = 0
    skip_count: int = 0
    start_time: float = time.monotonic()

    while processed_count + skip_count + error_count < total_images:
        result: dict | None = result_queue.get()

        if result is None:
            #sent poison pill to a worker pool 
            break

        if result.get("error"):
            error_count += 1
        elif result.get("skipped"):
            skip_count += 1
        else:
            processed_count += 1
            manifest_entries.append(result)

        completed: int = processed_count + skip_count + error_count
        if completed % 500 == 0 or completed == total_images:
            elapsed: float = time.monotonic() - start_time
            rate: float = completed / elapsed if elapsed > 0 else 0.0
            remaining_seconds: float = (total_images - completed) / rate if rate > 0 else 0.0
            logger.info(
                "Progress: %d/%d | processed: %d | skipped: %d | errors: %d | "
                "%.1f img/s | ETA: %.0fs",
                completed, total_images,
                processed_count, skip_count, error_count,
                rate, remaining_seconds,
            )

    manifest_data: dict = {
        "total_images": total_images,
        "processed": processed_count,
        "skipped": skip_count,
        "errors": error_count,
        "entries": manifest_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    logger.info("Manifest written to %s", manifest_path)


def main() -> None:
    total_cores: int = multiprocessing.cpu_count()
    worker_count: int = max(1, total_cores - 2)
    logger.info("CPU cores available: %d | workers: %d", total_cores, worker_count)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = [
        path for path in SOURCE_DIR.rglob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        logger.error("No images found in %s", SOURCE_DIR)
        return

    logger.info("Found %d images in %s", len(image_paths), SOURCE_DIR)

    range_config: DegradationRangeConfig = DegradationRangeConfig()

    task_args: list[tuple[Path, Path, DegradationRangeConfig]] = [
        (path, OUTPUT_DIR, range_config)
        for path in image_paths
    ]

    manager = multiprocessing.Manager()
    result_queue: multiprocessing.Queue = manager.Queue()

    start_time: float = time.monotonic()

    with multiprocessing.Pool(processes=worker_count) as pool:
        for result in pool.imap_unordered(_process_one_image, task_args, chunksize=10):
            result_queue.put(result)

    result_queue.put(None)

    _collect_results(result_queue, len(image_paths), MANIFEST_PATH)

    total_elapsed: float = time.monotonic() - start_time
    logger.info("Done in %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()