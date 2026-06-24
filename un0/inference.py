"""Generate class-conditional image samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from un0.common import resolve_device, save_sample_grid, seed_everything
from un0.data import NUM_CLASSES
from un0.model import (
    PRETRAINED_NAMES,
    ConditionalImplicitKuramotoGenerator,
    build_cifar10_model,
    build_from_config,
    build_imagenet64_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Path to a local .pt checkpoint.")
    source.add_argument(
        "--pretrained",
        choices=PRETRAINED_NAMES,
        help="Load released weights from Hugging Face by name.",
    )
    parser.add_argument(
        "--family",
        choices=("cifar10", "imagenet64"),
        default="cifar10",
        help=(
            "Model family for a local --checkpoint (default: cifar10). "
            "Ignored with --pretrained, whose family is fixed by name."
        ),
    )
    parser.add_argument("--output", default="samples/grid.png")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(NUM_CLASSES)),
        help="Class ids to sample from (default: all 10).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="How many images to generate per class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def generate(args: argparse.Namespace) -> None:
    """Generate and save samples."""
    seed_everything(int(args.seed))
    device = resolve_device("auto")
    if args.pretrained is not None:
        model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained, device=device)
    else:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        # Rebuild the architecture from the checkpoint's own config, selecting the
        # builder by --family. `build_from_config` passes only the arch keys each
        # builder accepts and lets absent keys fall back to the builder defaults.
        build_fn = build_cifar10_model if args.family == "cifar10" else build_imagenet64_model
        model = build_from_config(build_fn, state.get("config") or {})
        model.load_state_dict(state["model"])
        model = model.to(device)

    classes = torch.tensor(args.classes, device=device, dtype=torch.long)
    class_ids = classes.repeat_interleave(int(args.samples_per_class))
    samples = model.sample(class_ids)
    # Infer the square image size from the flat (B, 3*H*W) sample width so the
    # grid reshapes correctly for either family (CIFAR-10 32px, ImageNet-64 64px).
    image_size = round((samples.shape[1] // 3) ** 0.5)
    save_sample_grid(
        samples,
        Path(args.output),
        image_size=image_size,
        nrow=int(args.samples_per_class),
    )


def main() -> None:
    parser = build_parser()
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
