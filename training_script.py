"""
SRGAN Training Script
=====================
- Generator runs on main process (MPS if available, else CPU)
- Discriminator runs on a separate CPU core via multiprocessing
- Patches are read from cache/patches_manifest.json per_image entries
- Progress is logged to logs/training_metrics.csv
- Best model is saved to checkpoints/best_generator.pt
- Sample images saved to cache/training_outputs/samples/epoch_NNNN/

Improvements applied vs previous version:
    - fd exhaustion fix:    raise_file_descriptor_limit() called before any DataLoader spawning
                            persistent_workers=True on all DataLoaders so workers are not
                            torn down and recreated each epoch (was the main trigger)
                            safe worker count capped to min(requested, cpu//2, 4)
    - loss rebalancing:     pixel_loss_weight raised to 2.0, adversarial_loss_weight
                            lowered to 0.0005 — stops adv spikes from dominating training
    - multi-layer perc:     MultiLayerPerceptualLoss replaces single relu3_3 extraction;
                            combines relu2_2 + relu3_4 + relu4_4 with weighted L1
    - GAN stability:        label smoothing (real=0.9, fake=0.1) replaces hard 0/1 targets
                            R1 gradient penalty added to discriminator to prevent collapse
    - RRDB blocks:          ResidualDenseBlock replaces simple ResidualBlock;
                            DenseLayer with growth_rate=32 and 5 dense connections
                            num_residual_blocks raised to 16 in default config
    - pretrain warmup:      pixel-only pretrain now runs for 20 epochs before adv kicks in

What to look for during training:
    PSNR:               trending upward — above 25 dB after 30 epochs is good progress
    SSIM:               trending upward — above 0.85 after 30 epochs is good progress
    generator_loss:     trending downward over epochs (not every batch, epoch average)
    discriminator_loss: healthy range is 0.4 - 0.7
                        below 0.2 means discriminator dominates — lower its LR
                        above 0.8 means generator fools it too easily — lower generator LR
    real_score:         should stay near 0.7 - 0.9
    fake_score:         should slowly rise from near 0 toward 0.4 - 0.6 as generator improves
    sample images:      compare lr_input vs generated vs hr_target saved every N epochs
                        visible sharpening and noise removal = training is working
"""

import csv
import json
import logging
import multiprocessing
import os
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG19_Weights, vgg19
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import hflip, rotate, to_pil_image, vflip


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


PATCHES_MANIFEST_PATH = Path("cache/patches_manifest.json")
CHECKPOINTS_DIR = Path("checkpoints")
LOGS_DIR = Path("logs")
TRAINING_OUTPUTS_DIR = Path("cache/training_outputs")
SAMPLES_DIR = Path("cache/training_outputs/samples")
BEST_GENERATOR_PATH = Path("checkpoints/best_generator.pt")
BEST_DISCRIMINATOR_PATH = Path("checkpoints/best_discriminator.pt")
LATEST_CHECKPOINT_PATH = Path("checkpoints/latest_checkpoint.pt")
PRETRAIN_CHECKPOINT_PATH = Path("checkpoints/pretrain_latest.pt")
METRICS_CSV_PATH = Path("logs/training_metrics.csv")

R1_PENALTY_WEIGHT: float = 10.0


REAL_LABEL_SMOOTHED: float = 0.9
FAKE_LABEL_SMOOTHED: float = 0.1


# ---------------------------------------------------------------------------
# File descriptor limit
# ---------------------------------------------------------------------------

def raise_file_descriptor_limit(target_limit: int = 4096) -> None:
    """
    Raise the per-process file descriptor limit to avoid OSError: Too many open files.
    Each DataLoader worker opens file handles for patches, shared memory, and IPC pipes.
    On macOS the default soft limit is only 256, which exhausts quickly.
    """
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(target_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))
    logger.info(
        "File descriptor limit raised: %d → %d (hard cap: %d)",
        soft_limit, new_soft, hard_limit,
    )


def compute_safe_worker_count(requested_workers: int) -> int:
    """
    Cap worker count to min(requested, cpu_count // 2, 4).
    Each worker multiplies open file handles; on macOS this causes fd exhaustion fast.
    """
    cpu_count: int = os.cpu_count() or 1
    safe_count: int = min(requested_workers, cpu_count // 2, 4)
    if safe_count < requested_workers:
        logger.info(
            "Worker count reduced from %d to %d to avoid fd exhaustion",
            requested_workers, safe_count,
        )
    return safe_count


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def create_required_directories() -> None:
    for directory_path in [
        "checkpoints",
        "logs",
        "cache/training_outputs",
        "cache/training_outputs/samples",
    ]:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            logger.info("Created directory: %s", directory_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    batch_size: int = 16
    num_dataloader_workers: int = 4       # will be capped by compute_safe_worker_count
    patches_per_epoch: int = 5000
    learning_rate_generator: float = 1e-4
    learning_rate_discriminator: float = 1e-5
    lr_scheduler_step: int = 50
    lr_scheduler_gamma: float = 0.5
    pretrain_epochs: int = 20
    num_residual_blocks: int = 16         # raised from 8; using RRDB blocks now
    # --- loss weights rebalanced ---
    pixel_loss_weight: float = 2.0        # raised from 1.0 — anchors pixel fidelity
    perceptual_loss_weight: float = 0.006 # unchanged
    adversarial_loss_weight: float = 0.0005  # lowered from 0.01 — stops adv spikes
    discriminator_skip_threshold: float = 0.3
    validation_patch_count: int = 200
    save_samples_every_n_epochs: int = 5
    log_every_n_batches: int = 20


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_patch_paths_from_manifest(manifest_path: Path) -> list[str]:
    """Using manifest.json file loads every patch for image in format image_name_dir/patch_id.pt"""
    manifest = json.loads(manifest_path.read_text())
    per_image_entries = manifest["per_image"]
    all_patch_paths: list[str] = []

    for entry in per_image_entries:
        patch_dir = entry["patch_dir"]
        patch_count = entry["patch_count"]
        for patch_index in range(1, patch_count + 1):
            patch_file = os.path.join(patch_dir, f"{patch_index}.pt")
            if os.path.exists(patch_file):
                all_patch_paths.append(patch_file)

    logger.info("Loaded %d patch paths from manifest per_image entries", len(all_patch_paths))
    return all_patch_paths


class PatchDataset(Dataset):
    """Loads the dataset. randomly adds data augmentation. set to false no need in this step"""
    def __init__(self, patch_paths: list[str], augment: bool = False) -> None:
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_data = torch.load(self.patch_paths[index], weights_only=True)
        lr_tensor: torch.Tensor = patch_data["lr"]
        hr_tensor: torch.Tensor = patch_data["hr"]

        if self.augment:
            lr_tensor, hr_tensor = self._augment_pair(lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor

    def _augment_pair(
        self,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() > 0.5:
            lr_tensor = hflip(lr_tensor)
            hr_tensor = hflip(hr_tensor)

        if torch.rand(1).item() > 0.5:
            lr_tensor = vflip(lr_tensor)
            hr_tensor = vflip(hr_tensor)

        rotation_steps = int(torch.randint(0, 4, (1,)).item())
        if rotation_steps > 0:
            angle = rotation_steps * 90
            lr_tensor = rotate(lr_tensor, angle)
            hr_tensor = rotate(hr_tensor, angle)

        return lr_tensor, hr_tensor


# ---------------------------------------------------------------------------
# Generator — RRDB blocks
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """
    Single dense connection: concatenates its input with its output along the channel dim.
    Used as the building block inside ResidualDenseBlock.
    """
    def __init__(self, in_channels: int, growth_rate: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        new_features = self.activation(self.conv(features))
        return torch.cat([features, new_features], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) from ESRGAN.
    5 dense layers accumulate channels, then a 1x1 conv projects back to in_channels.
    Residual scaling of 0.2 prevents training instability.
    """
    def __init__(self, in_channels: int = 64, growth_rate: int = 32) -> None:
        super().__init__()
        self.dense_layers = nn.ModuleList([
            DenseLayer(in_channels + layer_index * growth_rate, growth_rate)
            for layer_index in range(5)
        ])
        self.final_conv = nn.Conv2d(in_channels + 5 * growth_rate, in_channels, kernel_size=1)
        self.residual_scale: float = 0.2

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        features = input_features
        for dense_layer in self.dense_layers:
            features = dense_layer(features)
        projected = self.final_conv(features)
        return input_features + self.residual_scale * projected


class SRGenerator(nn.Module):
    """The Main model class containing of 16 Residual Dense Blocks"""
    def __init__(self, num_residual_blocks: int = 16) -> None:
        super().__init__()
        self.entry_conv = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.entry_activation = nn.PReLU()

        blocks: list[nn.Module] = [
            ResidualDenseBlock(in_channels=64, growth_rate=32)
            for _ in range(num_residual_blocks)
        ]
        self.residual_blocks = nn.Sequential(*blocks)

        self.post_residual_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.post_residual_bn = nn.BatchNorm2d(64)
        self.exit_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        entry = self.entry_conv(lr_tensor)
        entry = self.entry_activation(entry)
        residual = self.residual_blocks(entry)
        post = self.post_residual_conv(residual)
        post = self.post_residual_bn(post)
        fused = entry + post
        return torch.sigmoid(self.exit_conv(fused))


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class DiscriminatorConvBlock(nn.Module):
    """Building blocks used inside the discriminator model"""
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(input_tensor)))


class SRDiscriminator(nn.Module):
    """Main discriminator model"""
    def __init__(self) -> None:
        super().__init__()
        self.entry_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.entry_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.block_one = DiscriminatorConvBlock(64, 64, stride=2)
        self.block_two = DiscriminatorConvBlock(64, 128, stride=1)
        self.block_three = DiscriminatorConvBlock(128, 128, stride=2)
        self.block_four = DiscriminatorConvBlock(128, 256, stride=1)
        self.block_five = DiscriminatorConvBlock(256, 256, stride=2)
        self.block_six = DiscriminatorConvBlock(256, 512, stride=1)
        self.block_seven = DiscriminatorConvBlock(512, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc_one = nn.Linear(512 * 4 * 4, 1024)
        self.fc_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc_two = nn.Linear(1024, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.entry_activation(self.entry_conv(input_tensor))
        output = self.block_one(output)
        output = self.block_two(output)
        output = self.block_three(output)
        output = self.block_four(output)
        output = self.block_five(output)
        output = self.block_six(output)
        output = self.block_seven(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.fc_activation(self.fc_one(output))
        return self.output_activation(self.fc_two(output))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class MultiLayerPerceptualLoss(nn.Module):
    """
    Combines VGG19 features from three depths: relu2_2, relu3_4, relu4_4.
    Shallower layers capture texture; deeper layers capture semantic structure.
    Using L1 instead of MSE reduces sensitivity to outlier activations.
    """
    def __init__(self) -> None:
        super().__init__()
        vgg_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.feature_extractor = create_feature_extractor(
            vgg_model,
            return_nodes={
                "features.9": "relu2_2",
                "features.18": "relu3_4",
                "features.27": "relu4_4",
            },
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        self.layer_weights: list[float] = [0.5, 1.0, 1.0]
        self.layer_keys: list[str] = ["relu2_2", "relu3_4", "relu4_4"]

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        generated_features = self.feature_extractor(generated)
        target_features = self.feature_extractor(target)

        total_loss: torch.Tensor = torch.tensor(0.0, device=generated.device)
        for layer_key, layer_weight in zip(self.layer_keys, self.layer_weights):
            total_loss = total_loss + layer_weight * F.l1_loss(
                generated_features[layer_key],
                target_features[layer_key],
            )
        return total_loss


def compute_r1_gradient_penalty(
    discriminator_output: torch.Tensor,
    real_images: torch.Tensor,
) -> torch.Tensor:
    """
    R1 regularization penalty.
    Penalizes the norm of discriminator gradients w.r.t. real images.
    Prevents discriminator loss from collapsing to near-zero on real samples.
    """
    gradients = torch.autograd.grad(
        outputs=discriminator_output.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    gradient_norm_squared = gradients.pow(2).reshape(gradients.shape[0], -1).sum(dim=1)
    return gradient_norm_squared.mean()


def compute_psnr(generated: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(generated.detach(), target.detach()).item()
    if mse == 0.0:
        return 100.0
    return float(10.0 * (torch.log10(torch.tensor(1.0 / mse))).item())


def compute_ssim(generated: torch.Tensor, target: torch.Tensor) -> float:
    constant_one = 0.01 ** 2
    constant_two = 0.03 ** 2
    mean_gen = generated.mean(dim=[2, 3], keepdim=True)
    mean_tgt = target.mean(dim=[2, 3], keepdim=True)
    var_gen = ((generated - mean_gen) ** 2).mean(dim=[2, 3])
    var_tgt = ((target - mean_tgt) ** 2).mean(dim=[2, 3])
    covariance = ((generated - mean_gen) * (target - mean_tgt)).mean(dim=[2, 3])
    numerator = (2 * mean_gen * mean_tgt + constant_one) * (2 * covariance + constant_two)
    denominator = (mean_gen ** 2 + mean_tgt ** 2 + constant_one) * (var_gen + var_tgt + constant_two)
    return float((numerator / denominator).mean().item())


# ---------------------------------------------------------------------------
# Logging / CSV
# ---------------------------------------------------------------------------

def init_csv_logger() -> None:
    if not os.path.exists(str(METRICS_CSV_PATH)):
        with open(str(METRICS_CSV_PATH), "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "epoch", "batch", "phase",
                "generator_total_loss", "pixel_loss", "perceptual_loss", "adversarial_loss",
                "discriminator_loss", "real_score", "fake_score", "psnr", "ssim",
            ])


def write_csv_row(
    epoch: int,
    batch: int,
    phase: str,
    generator_total_loss: float,
    pixel_loss: float,
    perceptual_loss: float,
    adversarial_loss: float,
    discriminator_loss: float,
    real_score: float,
    fake_score: float,
    psnr: float,
    ssim: float,
) -> None:
    with open(str(METRICS_CSV_PATH), "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            epoch, batch, phase,
            round(generator_total_loss, 6),
            round(pixel_loss, 6),
            round(perceptual_loss, 6),
            round(adversarial_loss, 6),
            round(discriminator_loss, 6),
            round(real_score, 4),
            round(fake_score, 4),
            round(psnr, 4),
            round(ssim, 4),
        ])


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

def save_sample_images(
    generator: SRGenerator,
    validation_paths: list[str],
    epoch: int,
    device: torch.device,
) -> None:
    epoch_dir = str(SAMPLES_DIR / f"epoch_{epoch:04d}")

    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)

    sampled_paths = random.sample(validation_paths, min(4, len(validation_paths)))
    sample_dataset = PatchDataset(sampled_paths, augment=False)

    sample_dataloader = DataLoader(sample_dataset, batch_size=4, shuffle=False, num_workers=0)
    lr_batch, hr_batch = next(iter(sample_dataloader))

    generator.eval()
    with torch.no_grad():
        sample_lr = lr_batch.to(device)
        sample_hr = hr_batch.to(device)
        generated = generator(sample_lr)

    for sample_index in range(sample_lr.shape[0]):
        lr_image = to_pil_image(sample_lr[sample_index].clamp(0, 1).cpu())
        generated_image = to_pil_image(generated[sample_index].clamp(0, 1).cpu())
        hr_image = to_pil_image(sample_hr[sample_index].clamp(0, 1).cpu())

        lr_image.save(os.path.join(epoch_dir, f"sample_{sample_index + 1}_lr_input.png"))
        generated_image.save(os.path.join(epoch_dir, f"sample_{sample_index + 1}_generated.png"))
        hr_image.save(os.path.join(epoch_dir, f"sample_{sample_index + 1}_hr_target.png"))

    generator.train()
    logger.info("Sample images saved to %s", epoch_dir)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    generator: SRGenerator,
    discriminator: SRDiscriminator,
    generator_optimizer: Adam,
    generator_scheduler: StepLR,
    epoch: int,
    best_psnr: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_psnr": best_psnr,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "generator_optimizer_state": generator_optimizer.state_dict(),
        "generator_scheduler_state": generator_scheduler.state_dict(),
    }
    torch.save(checkpoint, str(LATEST_CHECKPOINT_PATH))
    logger.info("Checkpoint saved — epoch %d", epoch)


def save_best_model(
    generator: SRGenerator,
    discriminator: SRDiscriminator,
    psnr: float,
) -> None:
    torch.save(generator.state_dict(), str(BEST_GENERATOR_PATH))
    torch.save(discriminator.state_dict(), str(BEST_DISCRIMINATOR_PATH))
    logger.info("Best model saved — PSNR: %.4f dB", psnr)


def load_checkpoint(
    generator: SRGenerator,
    generator_optimizer: Adam,
    generator_scheduler: StepLR,
) -> tuple[int, float]:
    if not os.path.exists(str(LATEST_CHECKPOINT_PATH)):
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0.0

    checkpoint = torch.load(str(LATEST_CHECKPOINT_PATH), weights_only=True)
    generator.load_state_dict(checkpoint["generator_state"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
    generator_scheduler.load_state_dict(checkpoint["generator_scheduler_state"])

    start_epoch: int = checkpoint["epoch"] + 1
    best_psnr: float = checkpoint["best_psnr"]
    logger.info("Resumed from epoch %d — best PSNR so far: %.4f dB", start_epoch, best_psnr)
    return start_epoch, best_psnr


# ---------------------------------------------------------------------------
# Discriminator worker (separate process)
# ---------------------------------------------------------------------------

def run_discriminator_worker(
    initial_state_dict: dict,
    batch_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    config: TrainingConfig,
) -> None:
    discriminator = SRDiscriminator()
    discriminator.load_state_dict(initial_state_dict)
    discriminator.train()

    optimizer = Adam(
        discriminator.parameters(),
        lr=config.learning_rate_discriminator,
        betas=(0.9, 0.999),
    )
    scheduler = StepLR(
        optimizer,
        step_size=config.lr_scheduler_step,
        gamma=config.lr_scheduler_gamma,
    )

    last_discriminator_loss: float = 1.0
    current_epoch: int = 0

    while True:
        message = batch_queue.get()

        if message is None:
            logger.info("Discriminator worker received stop signal — shutting down")
            break

        message_epoch: int = message.get("epoch", current_epoch)
        if message_epoch != current_epoch:
            scheduler.step()
            current_epoch = message_epoch

        real_hr: torch.Tensor = message["real_hr"]
        generated_hr: torch.Tensor = message["generated_hr"]

        # require_grad needed on real_hr to compute R1 gradient penalty
        real_hr = real_hr.requires_grad_(True)

        real_score = discriminator(real_hr)
        fake_score = discriminator(generated_hr.detach())

        # label smoothing: use 0.9 / 0.1 instead of 1.0 / 0.0
        real_targets = torch.full_like(real_score, REAL_LABEL_SMOOTHED)
        fake_targets = torch.full_like(fake_score, FAKE_LABEL_SMOOTHED)

        real_loss = F.binary_cross_entropy(real_score, real_targets)
        fake_loss = F.binary_cross_entropy(fake_score, fake_targets)

        r1_penalty = compute_r1_gradient_penalty(real_score, real_hr)
        discriminator_loss = (real_loss + fake_loss) * 0.5 + R1_PENALTY_WEIGHT * r1_penalty

        if last_discriminator_loss > config.discriminator_skip_threshold:
            optimizer.zero_grad()
            discriminator_loss.backward()
            optimizer.step()
        else:
            discriminator_loss = discriminator_loss.detach()

        last_discriminator_loss = discriminator_loss.item()

        result_queue.put({
            "discriminator_state": discriminator.state_dict(),
            "discriminator_loss": last_discriminator_loss,
            "real_score_mean": real_score.mean().item(),
            "fake_score_mean": fake_score.mean().item(),
        })


# ---------------------------------------------------------------------------
# Pretrain phase (pixel-only warmup)
# ---------------------------------------------------------------------------

def run_pretrain_phase(
    generator: SRGenerator,
    training_paths: list[str],
    config: TrainingConfig,
    device: torch.device,
    safe_worker_count: int,
) -> None:
    pretrain_start_epoch: int = 0

    if os.path.exists(str(PRETRAIN_CHECKPOINT_PATH)):
        pretrain_data = torch.load(str(PRETRAIN_CHECKPOINT_PATH), weights_only=True)
        generator.load_state_dict(pretrain_data["generator_state"])
        pretrain_start_epoch = pretrain_data["epoch"] + 1
        logger.info("Pretrain resumed from epoch %d", pretrain_start_epoch)

    if pretrain_start_epoch >= config.pretrain_epochs:
        logger.info("Pretrain already complete — skipping to adversarial phase")
        return

    optimizer = Adam(generator.parameters(), lr=config.learning_rate_generator)
    logger.info(
        "Starting pretrain phase — %d epochs remaining",
        config.pretrain_epochs - pretrain_start_epoch,
    )

    for epoch in range(pretrain_start_epoch, config.pretrain_epochs):
        sampled_paths = random.sample(
            training_paths,
            min(config.patches_per_epoch, len(training_paths)),
        )
        dataset = PatchDataset(sampled_paths, augment=True)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=safe_worker_count,
            persistent_workers=safe_worker_count > 0,  # reuse workers; no re-spawn between epochs
        )

        generator.train()
        epoch_loss: float = 0.0
        batch_count: int = 0
        epoch_start = time.monotonic()

        iterator = iter(dataloader)
        try:
            for batch_index, (lr_batch, hr_batch) in enumerate(iterator):
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                optimizer.zero_grad()
                generated = generator(lr_batch)
                pixel_loss = F.l1_loss(generated, hr_batch)
                pixel_loss.backward()
                optimizer.step()

                epoch_loss += pixel_loss.item()
                batch_count += 1

                if batch_index % config.log_every_n_batches == 0:
                    psnr = compute_psnr(generated.detach(), hr_batch)
                    ssim = compute_ssim(generated.detach(), hr_batch)
                    logger.info(
                        "[PRETRAIN] Epoch %d | Batch %d | pixel_loss: %.4f | PSNR: %.2f dB | SSIM: %.4f",
                        epoch, batch_index, pixel_loss.item(), psnr, ssim,
                    )
                    write_csv_row(
                        epoch, batch_index, "pretrain",
                        pixel_loss.item(), pixel_loss.item(), 0.0, 0.0,
                        0.0, 0.0, 0.0, psnr, ssim,
                    )
        finally:
            del iterator  # release worker file handles even if loop exits early

        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = time.monotonic() - epoch_start
        logger.info(
            "[PRETRAIN] Epoch %d done | avg pixel_loss: %.4f | elapsed: %.1f s",
            epoch, avg_loss, elapsed,
        )
        torch.save(
            {"epoch": epoch, "generator_state": generator.state_dict()},
            str(PRETRAIN_CHECKPOINT_PATH),
        )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(config: TrainingConfig) -> None:
    # Must be called before any DataLoader is created
    raise_file_descriptor_limit(target_limit=4096)

    create_required_directories()
    init_csv_logger()

    safe_worker_count = compute_safe_worker_count(config.num_dataloader_workers)
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print(f"MainTrainLoop>  CUDA is available")
        device = torch.device("cuda")

    #--- set to cuda device
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Using device: %s", device)

    all_patch_paths = load_patch_paths_from_manifest(PATCHES_MANIFEST_PATH)
    if len(all_patch_paths) == 0:
        logger.error("No patch files found — check your patches_manifest.json")
        return

    random.shuffle(all_patch_paths)
    validation_paths = all_patch_paths[:config.validation_patch_count]
    training_paths = all_patch_paths[config.validation_patch_count:]
    logger.info(
        "Training patches: %d | Validation patches: %d",
        len(training_paths), len(validation_paths),
    )

    generator = SRGenerator(num_residual_blocks=config.num_residual_blocks).to(device)
    discriminator = SRDiscriminator().to(device)
    perceptual_loss_fn = MultiLayerPerceptualLoss().to(device)

    generator_optimizer = Adam(
        generator.parameters(),
        lr=config.learning_rate_generator,
        betas=(0.9, 0.999),
    )
    generator_scheduler = StepLR(
        generator_optimizer,
        step_size=config.lr_scheduler_step,
        gamma=config.lr_scheduler_gamma,
    )

    start_epoch, best_psnr = load_checkpoint(
        generator,
        generator_optimizer,
        generator_scheduler,
    )

    run_pretrain_phase(generator, training_paths, config, device, safe_worker_count)

    batch_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)
    result_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=4)

    cpu_discriminator_state = {
        key: value.cpu()
        for key, value in discriminator.state_dict().items()
    }

    discriminator_process = multiprocessing.Process(
        target=run_discriminator_worker,
        args=(cpu_discriminator_state, batch_queue, result_queue, config),
        name="DiscriminatorWorker",
        daemon=True,
    )
    discriminator_process.start()
    logger.info("Discriminator worker started — PID %d", discriminator_process.pid)
    logger.info("Starting adversarial training from epoch %d", start_epoch)

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.monotonic()

        sampled_paths = random.sample(
            training_paths,
            min(config.patches_per_epoch, len(training_paths)),
        )
        train_dataset = PatchDataset(sampled_paths, augment=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=safe_worker_count,
            persistent_workers=safe_worker_count > 0,  # reuse workers; no re-spawn between epochs
            prefetch_factor=2 if safe_worker_count > 0 else None,
        )

        generator.train()

        epoch_generator_loss: float = 0.0
        epoch_discriminator_loss: float = 0.0
        epoch_psnr: float = 0.0
        epoch_ssim: float = 0.0
        batch_count: int = 0

        iterator = iter(train_dataloader)
        try:
            for batch_index, (lr_batch, hr_batch) in enumerate(iterator):
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                generator_optimizer.zero_grad()
                generated_hr = generator(lr_batch)

                batch_queue.put({
                    "real_hr": hr_batch.cpu(),
                    "generated_hr": generated_hr.detach().cpu(),
                    "epoch": epoch,
                })

                with torch.no_grad():
                    fake_score_for_generator = discriminator(generated_hr)

                adversarial_loss = F.binary_cross_entropy(
                    fake_score_for_generator,
                    torch.ones_like(fake_score_for_generator),
                )
                perceptual_loss = perceptual_loss_fn(generated_hr, hr_batch)
                pixel_loss = F.l1_loss(generated_hr, hr_batch)

                generator_total_loss = (
                    config.pixel_loss_weight * pixel_loss
                    + config.perceptual_loss_weight * perceptual_loss
                    + config.adversarial_loss_weight * adversarial_loss
                )

                generator_total_loss.backward()
                generator_optimizer.step()

                discriminator_result = result_queue.get()
                discriminator.load_state_dict(discriminator_result["discriminator_state"])
                discriminator_loss_value: float = discriminator_result["discriminator_loss"]
                real_score_mean: float = discriminator_result["real_score_mean"]
                fake_score_mean: float = discriminator_result["fake_score_mean"]

                psnr = compute_psnr(generated_hr.detach(), hr_batch)
                ssim = compute_ssim(generated_hr.detach(), hr_batch)

                epoch_generator_loss += generator_total_loss.item()
                epoch_discriminator_loss += discriminator_loss_value
                epoch_psnr += psnr
                epoch_ssim += ssim
                batch_count += 1

                if batch_index % config.log_every_n_batches == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | "
                        "G_loss: %.4f (px: %.4f | perc: %.6f | adv: %.4f) | "
                        "D_loss: %.4f | real: %.3f | fake: %.3f | "
                        "PSNR: %.2f dB | SSIM: %.4f",
                        epoch, batch_index, len(train_dataloader),
                        generator_total_loss.item(),
                        pixel_loss.item(),
                        perceptual_loss.item(),
                        adversarial_loss.item(),
                        discriminator_loss_value,
                        real_score_mean,
                        fake_score_mean,
                        psnr, ssim,
                    )
                    write_csv_row(
                        epoch, batch_index, "adversarial",
                        generator_total_loss.item(),
                        pixel_loss.item(),
                        perceptual_loss.item(),
                        adversarial_loss.item(),
                        discriminator_loss_value,
                        real_score_mean,
                        fake_score_mean,
                        psnr, ssim,
                    )
        finally:
            del iterator  # release worker file handles even if an exception causes early exit

        generator_scheduler.step()

        avg_generator_loss = epoch_generator_loss / max(batch_count, 1)
        avg_discriminator_loss = epoch_discriminator_loss / max(batch_count, 1)
        avg_psnr = epoch_psnr / max(batch_count, 1)
        avg_ssim = epoch_ssim / max(batch_count, 1)
        epoch_elapsed = time.monotonic() - epoch_start

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_best_model(generator, discriminator, best_psnr)

        save_checkpoint(
            generator, discriminator,
            generator_optimizer, generator_scheduler,
            epoch, best_psnr,
        )

        if epoch % config.save_samples_every_n_epochs == 0:
            save_sample_images(generator, validation_paths, epoch, device)

        logger.info(
            "=== Epoch %d/%d | avg G_loss: %.4f | avg D_loss: %.4f | "
            "avg PSNR: %.2f dB | best PSNR: %.2f dB | avg SSIM: %.4f | elapsed: %.1f s ===",
            epoch, config.num_epochs,
            avg_generator_loss, avg_discriminator_loss,
            avg_psnr, best_psnr,
            avg_ssim, epoch_elapsed,
        )

    batch_queue.put(None)
    discriminator_process.join(timeout=15)
    logger.info("Training complete — best PSNR: %.4f dB", best_psnr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(str(PATCHES_MANIFEST_PATH)):
        logger.error(
            "patches_manifest.json not found at %s — run split_patches.py first",
            PATCHES_MANIFEST_PATH,
        )
        return

    config = TrainingConfig()
    run_training(config)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()