from __future__ import annotations

import argparse
import functools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

from sparserecon.dust3r_utils import infer_dust3r
from sparserecon.pipeline import run_reconstruction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "demo"
EXAMPLE_CATEGORIES = ["toy", "butter", "robot", "jordan", "eagle"]

sys.path.insert(0, str(PROJECT_ROOT / "dust3r"))
from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402


def info_fn() -> None:
    gr.Info("Data preprocessing done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the SparseRecon Gradio demo.")
    parser.add_argument("--output", default="output/demo", type=str, help="Output directory.")
    parser.add_argument("--category", default="jordan", type=str, help="Default demo category.")
    parser.add_argument("--num_pts", default=25000, type=int, help="Number of points at initialization.")
    parser.add_argument("--num_views", default=8, type=int, help="Number of input images.")
    parser.add_argument("--mesh_format", default="glb", type=str, help="Output mesh format.")
    parser.add_argument("--enable_loop", action="store_true", default=True, help="Enable outlier correction.")
    parser.add_argument("--config", default="navi.yaml", type=str, help="Config file name or path.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--render_video", action="store_true", help="Render orbit videos during reconstruction.")
    return parser


def resolve_output_root(output: str) -> Path:
    output_root = Path(output)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return output_root


def load_examples() -> list[list[list[str]]]:
    examples_full = []
    for category in EXAMPLE_CATEGORIES:
        example_folder = DATA_ROOT / category / "processed"
        example_paths = sorted(str(path) for path in example_folder.glob("*.png"))
        examples_full.append([example_paths])
    return examples_full


def get_select_index(args: argparse.Namespace, examples_full: list, evt: gr.SelectData):
    index = evt.index
    args.num_views = len(examples_full[index][0])
    args.category = EXAMPLE_CATEGORIES[index]
    return examples_full[index][0], examples_full[index][0]


def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload input images.")


def extract_file_path(file_obj) -> Path:
    if isinstance(file_obj, str):
        return Path(file_obj)
    return Path(getattr(file_obj, "name"))


def preprocess_uploads(args: argparse.Namespace, dust3r_model, image_block: list):
    custom_data_dir = DATA_ROOT / "custom"
    custom_output_dir = resolve_output_root(args.output) / "custom"
    if custom_data_dir.exists():
        shutil.rmtree(custom_data_dir)
    if custom_output_dir.exists():
        shutil.rmtree(custom_output_dir)

    (custom_data_dir / "source").mkdir(parents=True, exist_ok=True)
    (custom_data_dir / "processed").mkdir(parents=True, exist_ok=True)

    file_names = []
    for file_obj in image_block:
        file_path = extract_file_path(file_obj)
        input_path = custom_data_dir / file_path.name
        img_pil = Image.open(file_path)

        try:
            img_pil.save(input_path)
        except OSError:
            img_pil.convert("RGB").save(input_path)

        file_names.append(str(custom_data_dir / "source" / f"{input_path.stem}.png"))
        subprocess.run(
            [sys.executable, "-m", "sparserecon.preprocess", str(input_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    camera_data = infer_dust3r(dust3r_model, file_names)
    with (custom_data_dir / "cameras.json").open("w", encoding="utf-8") as file:
        json.dump(camera_data, file, indent=4)

    args.num_views = len(file_names)
    args.category = "custom"

    return [str(custom_data_dir / "processed" / f"{extract_file_path(path).stem}_rgba.png") for path in image_block]


def run_single_reconstruction(args: argparse.Namespace, image_block: list) -> str:
    args.enable_loop = False
    run_reconstruction(args)
    return str(resolve_output_root(args.output) / args.category / "round_0" / f"{args.category}.glb")


def run_full_reconstruction(args: argparse.Namespace, image_block: list) -> str:
    args.enable_loop = True
    run_reconstruction(args)

    category_dir = resolve_output_root(args.output) / args.category
    if (category_dir / "cameras_final_recovered.json").exists():
        return str(category_dir / "check_recovered_poses" / f"{args.category}.glb")
    if (category_dir / "cameras_final_init.json").exists():
        return str(category_dir / "reconsider_init_poses" / f"{args.category}.glb")
    return str(category_dir / "round_1" / f"{args.category}.glb")


def main() -> None:
    args = build_parser().parse_args()
    title = "SparseRecon: Sparse-View Object Reconstruction"
    description = """
    <div>
    <a style="display:inline-block" href="https://github.com/franklinz233/Sparserecon"><img src="https://img.shields.io/badge/project-SparseRecon-2f6f6d"></a>
    </div>
    SparseRecon reconstructs textured 3D assets from sparse, unposed object images. It uses DUSt3R for custom pose initialization and a two-stage reconstruction backend adapted from SparseAGS.
    """
    image_guide = "Once the preprocessed images appear, click **Run Single 3D Reconstruction**. If the result is poor, try **Outlier Removal & Correction**."

    examples_full = load_examples()
    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_224_linear"
    ).to("cuda")
    print("Loaded DUSt3R model.")

    preprocess_fn = functools.partial(preprocess_uploads, args, dust3r_model)
    select_fn = functools.partial(get_select_index, args, examples_full)
    run_single_fn = functools.partial(run_single_reconstruction, args)
    run_full_fn = functools.partial(run_full_reconstruction, args)

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# " + title)
        gr.Markdown(description)

        with gr.Row(variant="panel"):
            with gr.Column(scale=5):
                image_block = gr.Files(file_count="multiple")
                preprocess_btn = gr.Button("Preprocess Images")
                gr.Markdown(
                    "Upload images and click **Preprocess Images** to initialize poses with DUSt3R, "
                    "or select a preprocessed example below."
                )

                gallery = gr.Gallery(
                    value=[example[0][0] for example in examples_full if example[0]],
                    label="Examples",
                    show_label=True,
                    elem_id="gallery",
                    columns=[5],
                    rows=[1],
                    object_fit="contain",
                    height="256",
                    preview=None,
                    allow_preview=None,
                )

                preprocessed_data = gr.Gallery(
                    label="Preprocessed images",
                    show_label=True,
                    elem_id="preprocessed-gallery",
                    columns=[4],
                    rows=[2],
                    object_fit="contain",
                    height="256",
                    preview=None,
                    allow_preview=None,
                )

                with gr.Row(variant="panel"):
                    run_single_btn = gr.Button("Run Single 3D Reconstruction")
                    outlier_detect_btn = gr.Button("Outlier Removal & Correction")
                gr.Markdown(image_guide, visible=True)

            with gr.Column(scale=5):
                obj_single_recon = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model (Single Reconstruction)",
                )
                obj_outlier_detect = gr.Model3D(
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    label="3D Model (Full Method)",
                )

        gallery.select(select_fn, None, outputs=[image_block, preprocessed_data])
        preprocess_btn.click(
            preprocess_fn,
            inputs=[image_block],
            outputs=[preprocessed_data],
            queue=False,
            show_progress="full",
        ).success(info_fn, None, None)

        run_single_btn.click(check_img_input, inputs=[image_block], queue=False).success(
            run_single_fn,
            inputs=[image_block],
            outputs=[obj_single_recon],
        )
        outlier_detect_btn.click(check_img_input, inputs=[image_block], queue=False).success(
            run_full_fn,
            inputs=[image_block],
            outputs=[obj_outlier_detect],
        )

    demo.queue().launch(share=args.share)
