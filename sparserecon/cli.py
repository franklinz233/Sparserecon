from __future__ import annotations

import argparse

from sparserecon.pipeline import run_reconstruction, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SparseRecon sparse-view reconstruction.")
    parser.add_argument("--output", default="output/demo", type=str, help="Output directory.")
    parser.add_argument("--category", default="jordan", type=str, help="Dataset name under data/demo.")
    parser.add_argument("--num_pts", default=25000, type=int, help="Number of points at initialization.")
    parser.add_argument("--num_views", default=8, type=int, help="Number of input images.")
    parser.add_argument("--mesh_format", default="obj", type=str, help="Output mesh format.")
    parser.add_argument("--enable_loop", action="store_true", help="Enable outlier detection and correction.")
    parser.add_argument("--config", default="navi.yaml", type=str, help="Config file name or path.")
    parser.add_argument("--render_video", action="store_true", help="Render an orbit video after each reconstruction.")
    return parser


def main() -> None:
    seed_everything(0)
    run_reconstruction(build_parser().parse_args())


if __name__ == "__main__":
    main()
