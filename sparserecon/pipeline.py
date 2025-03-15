from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from sparserecon.render_utils.util import align_to_mesh, render_and_compare
from sparserecon.visual_utils import vis_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "demo"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_command(command: list[str]) -> None:
    print("[CMD]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def resolve_config_path(config: str) -> Path:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / "configs" / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def omega_path(path: Path) -> str:
    return path.as_posix()


def run_stage(
    module: str,
    config_path: Path,
    camera_path: Path,
    save_path: Path,
    args: argparse.Namespace,
    force_all_views: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        "--config",
        omega_path(config_path),
        f"camera_path={omega_path(camera_path)}",
        "outdir=./",
        f"save_path={omega_path(save_path)}",
        f"all_views={force_all_views}",
        f"mesh_format={args.mesh_format}",
    ]

    if module.endswith("coarse_gaussian"):
        command.extend(["opt_cam=1", "sh_degree=3", f"num_pts={args.num_pts}"])

    run_command(command)


def reconstruct_round(cur_dir: Path, args: argparse.Namespace, force_all_views: bool = False) -> None:
    config_path = resolve_config_path(args.config)
    save_path = cur_dir / args.category

    stage1_mesh = cur_dir / f"{args.category}_mesh.{args.mesh_format}"
    if not stage1_mesh.exists():
        run_stage(
            "sparserecon.stages.coarse_gaussian",
            config_path,
            cur_dir / "cameras.json",
            save_path,
            args,
            force_all_views,
        )

    stage2_mesh = cur_dir / f"{args.category}.{args.mesh_format}"
    if not stage2_mesh.exists():
        run_stage(
            "sparserecon.stages.mesh_refinement",
            config_path,
            cur_dir / "cameras_updated.json",
            save_path,
            args,
            force_all_views,
        )

    video_path = cur_dir / f"{args.category}.mp4"
    if args.render_video and not video_path.exists():
        run_command(
            [
                sys.executable,
                "-m",
                "kiui.render",
                omega_path(stage2_mesh),
                "--save_video",
                omega_path(video_path),
                "--wogui",
                "--elevation",
                "-30",
            ]
        )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def mean_valid_loss(losses: np.ndarray, flags: np.ndarray) -> float:
    valid_count = flags.sum()
    if valid_count == 0:
        raise ValueError("No valid cameras remain for metric computation.")
    return float(np.sum(losses * flags) / valid_count)


def max_loop_count(num_views: int, enable_loop: bool) -> int:
    if not enable_loop:
        return 0
    if num_views <= 5:
        return 1
    if num_views < 8:
        return num_views - 4
    return 4


def resolve_output_root(output: str) -> Path:
    output_root = Path(output)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root


def run_reconstruction(args: argparse.Namespace) -> None:
    src_dir = DEFAULT_DATA_ROOT / args.category
    out_dir = resolve_output_root(args.output) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    source_camera_path = src_dir / "cameras.json"

    print(f"======== processing {src_dir} ========")
    if not source_camera_path.exists():
        raise FileNotFoundError(f"{source_camera_path} is missing.")

    threshold_lpips = 0.05
    max_cnt = max_loop_count(args.num_views, args.enable_loop)
    metrics_track = {}
    cnt = 0
    cur_dir = out_dir / f"round_{cnt}"
    cur_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_camera_path, cur_dir / "cameras.json")

    cnt_to_stop = 0
    mean_lpips_wo_max = None

    while True:
        metrics_track[cnt] = []
        reconstruct_round(cur_dir, args)
        camera_data = read_json(cur_dir / "cameras_updated.json")

        lpips_losses, mse_losses = vis_output(
            camera_data,
            mesh_path=str(cur_dir / f"{args.category}.{args.mesh_format}"),
            save_path=str(cur_dir / "vis.png"),
            num_views=args.num_views,
        )
        flags = np.array([int(v["flag"]) for v in camera_data.values()])

        mean_lpips = mean_valid_loss(lpips_losses, flags)
        metrics_track[cnt].append(mean_lpips)

        stop_with_lower_quality = False
        stop_with_full_iters = False
        if cnt != 0 and args.enable_loop:
            last_lpips = mean_lpips_wo_max
            diff_lpips = abs(last_lpips - mean_lpips)
            stop_with_lower_quality = mean_lpips > last_lpips or diff_lpips < threshold_lpips
            stop_with_full_iters = cnt >= max_cnt

        if stop_with_full_iters or stop_with_lower_quality:
            cnt_to_stop = cnt - 1 if stop_with_lower_quality else cnt
            shutil.copy2(
                out_dir / f"round_{cnt_to_stop}" / "cameras_updated.json",
                out_dir / "cameras_outlier_removal.json",
            )
            write_json(metrics_track, out_dir / "metrics.json")
            break

        if not args.enable_loop:
            write_json(metrics_track, out_dir / "metrics.json")
            break

        max_index = -1
        max_lpips_value = -float("inf")
        for i in range(args.num_views):
            if flags[i] == 1 and lpips_losses[i] > max_lpips_value:
                max_lpips_value = lpips_losses[i]
                max_index = i

        flags[max_index] = 0
        mean_lpips_wo_max = mean_valid_loss(lpips_losses, flags)
        metrics_track[cnt].append(mean_lpips_wo_max)

        camera_keys = list(camera_data.keys())
        assert camera_data[camera_keys[max_index]]["flag"] == 1
        camera_data[camera_keys[max_index]]["flag"] = 0

        cnt += 1
        cur_dir = out_dir / f"round_{cnt}"
        cur_dir.mkdir(parents=True, exist_ok=True)
        write_json(camera_data, cur_dir / "cameras.json")

    if cnt_to_stop == 0:
        return

    camera_path_outlier_removal = out_dir / "cameras_outlier_removal.json"
    camera_data_outlier_removal = read_json(camera_path_outlier_removal)

    camera_path_render_and_compare = out_dir / "cameras_render_and_compare.json"
    if not camera_path_render_and_compare.exists():
        mesh_path = out_dir / f"round_{cnt_to_stop}" / f"{args.category}.{args.mesh_format}"
        camera_data_render_and_compare = render_and_compare(
            copy.deepcopy(camera_data_outlier_removal),
            str(mesh_path),
            str(out_dir),
            num_views=args.num_views,
        )
        write_json(camera_data_render_and_compare, camera_path_render_and_compare)

    cur_dir = out_dir / "check_recovered_poses"
    cur_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(camera_path_render_and_compare, cur_dir / "cameras.json")
    reconstruct_round(cur_dir, args, force_all_views=True)

    camera_data = read_json(cur_dir / "cameras_updated.json")
    lpips_losses, mse_losses = vis_output(
        camera_data,
        mesh_path=str(cur_dir / f"{args.category}.{args.mesh_format}"),
        save_path=str(cur_dir / "vis.png"),
        num_views=args.num_views,
    )

    cur_dir = out_dir / "reconsider_init_poses"
    cur_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / f"round_{cnt_to_stop}" / f"{args.category}.{args.mesh_format}"
    camera_data_aligned = align_to_mesh(
        camera_data_outlier_removal,
        str(mesh_path),
        str(cur_dir),
        num_views=args.num_views,
    )
    write_json(camera_data_aligned, cur_dir / "cameras.json")

    reconstruct_round(cur_dir, args, force_all_views=True)
    camera_data_init = read_json(cur_dir / "cameras_updated.json")
    lpips_losses_init, mse_losses_init = vis_output(
        camera_data_init,
        mesh_path=str(cur_dir / f"{args.category}.{args.mesh_format}"),
        save_path=str(cur_dir / "vis.png"),
        num_views=args.num_views,
    )

    flags_sum = np.array([int(v["flag"]) for v in camera_data_outlier_removal.values()]).sum()
    cnt_valid_cameras = 0
    keep_init_poses = lpips_losses.mean() > lpips_losses_init.mean()

    if not keep_init_poses:
        for idx, (_, camera) in enumerate(camera_data_outlier_removal.items()):
            if int(camera["flag"]) == 1:
                continue
            if lpips_losses[idx] < lpips_losses_init[idx] and mse_losses[idx] < mse_losses_init[idx]:
                cnt_valid_cameras += 1
        keep_init_poses = cnt_valid_cameras + flags_sum != args.num_views

    output_path = out_dir / "cameras_final.json"
    if keep_init_poses:
        print("Keep the optimized initial camera poses.")
        write_json(camera_data_init, Path(str(output_path).replace(".json", "_init.json")))
    else:
        print("Replace the initial cameras with the recovered ones.")
        write_json(camera_data, Path(str(output_path).replace(".json", "_recovered.json")))
