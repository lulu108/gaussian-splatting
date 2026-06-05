import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def image_stats(label, image_or_array):
    arr = np.asarray(image_or_array)
    print(f"{label}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean():.6f}")
    return arr


def composite_rgba(rgba, white_background=True):
    bg = np.array([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0], dtype=np.float32)
    norm = rgba.astype(np.float32) / 255.0
    comp = norm[..., :3] * norm[..., 3:4] + bg * (1.0 - norm[..., 3:4])
    return np.array(comp * 255.0, dtype=np.uint8)


def mean_abs_diff(a, b):
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()


def simple_psnr(a, b):
    mse = np.square(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0).mean()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def save_view_original_image(source_path, debug_dir):
    from scene.dataset_readers import readNerfSyntheticInfo
    from utils.camera_utils import cameraList_from_camInfos

    scene_info = readNerfSyntheticInfo(str(source_path), white_background=True, depths="", eval=True)
    camera_args = SimpleNamespace(
        resolution=-1,
        data_device="cuda",
        train_test_exp=False,
        white_background=True,
    )
    test_cameras = cameraList_from_camInfos(
        scene_info.test_cameras[:1],
        resolution_scale=1.0,
        args=camera_args,
        is_nerf_synthetic=True,
        is_test_dataset=True,
    )
    view = test_cameras[0]
    view_arr = (
        view.original_image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    out_path = debug_dir / "debug_view_original_image.png"
    Image.fromarray(view_arr, "RGB").save(out_path)
    return out_path, view_arr


def save_manual_white_gt(source_path, debug_dir):
    transforms_path = source_path / "transforms_test.json"
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    first_frame = transforms["frames"][0]
    image_path = source_path / f"{first_frame['file_path']}.png"
    rgba = np.asarray(Image.open(image_path).convert("RGBA"))
    manual = composite_rgba(rgba, white_background=True)
    out_path = debug_dir / "debug_manual_white_gt.png"
    Image.fromarray(manual, "RGB").save(out_path)
    return out_path, manual


def main():
    parser = argparse.ArgumentParser(description="Debug NeRF Synthetic Lego RGBA/background/render pipeline.")
    parser.add_argument("--source", "-s", default="data/lego", help="Path to NeRF Synthetic Lego dataset.")
    parser.add_argument("--model", "-m", default="output/lego_white_baseline", help="Path to trained model output.")
    parser.add_argument("--iteration", type=int, default=60000, help="Rendered iteration to inspect.")
    parser.add_argument("--debug_dir", default="tools/debug_lego_pipeline_outputs", help="Directory for debug PNGs.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    model_path = Path(args.model)
    debug_dir = Path(args.debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    print("== Raw RGBA input ==")
    raw_path = source_path / "test" / "r_0.png"
    raw_image = Image.open(raw_path)
    rgba = np.asarray(raw_image.convert("RGBA"))
    alpha = rgba[..., 3]
    white = composite_rgba(rgba, white_background=True)
    black = composite_rgba(rgba, white_background=False)
    print(f"RGBA mode: {raw_image.mode}")
    image_stats("raw RGB", rgba[..., :3])
    print(f"alpha: min={alpha.min()} max={alpha.max()} mean={alpha.mean():.6f}")
    image_stats("white composite", white)
    image_stats("black composite", black)

    print("\n== Render output ==")
    method_dir = model_path / "test" / f"ours_{args.iteration}"
    render_path = method_dir / "renders" / "00000.png"
    gt_path = method_dir / "gt" / "00000.png"
    if render_path.exists() and gt_path.exists():
        render_arr = image_stats("render 00000", Image.open(render_path).convert("RGB"))
        gt_arr = image_stats("gt 00000", Image.open(gt_path).convert("RGB"))
        print(f"render/gt mean abs diff: {mean_abs_diff(render_arr, gt_arr):.6f}")
        print(f"render/gt simple PSNR: {simple_psnr(render_arr, gt_arr):.6f}")
    else:
        print(f"Missing render or gt: {render_path} / {gt_path}")

    print("\n== Camera pipeline GT ==")
    manual_path, manual_arr = save_manual_white_gt(source_path, debug_dir)
    print(f"saved manual white GT: {manual_path}")
    try:
        view_path, view_arr = save_view_original_image(source_path, debug_dir)
    except Exception as exc:
        print(f"无法读取真实 view.original_image: {type(exc).__name__}: {exc}")
        print("判断: 当前 Python/CUDA 环境无法构造 Camera；请在训练环境中重跑以检查真实 view.original_image")
        return

    print(f"saved view.original_image: {view_path}")
    view_manual_mad = mean_abs_diff(view_arr, manual_arr)
    print(f"view/manual mean abs diff: {view_manual_mad:.6f}")
    print(f"view/manual simple PSNR: {simple_psnr(view_arr, manual_arr):.6f}")
    if view_manual_mad <= 1.0 / 255.0:
        print("判断: view.original_image 等于手动白背景 GT")
    else:
        print("判断: view.original_image 不等于手动白背景 GT")


if __name__ == "__main__":
    main()
