#!/usr/bin/env python3
import os
import json
import subprocess
from PIL import Image
import numpy as np

# ===================== 配置区 =====================
DATA_ROOT = "data/lego"           # Lego 数据集路径
OUTPUT_DIR = "output/lego_white_baseline"  # 输出路径
ITERATIONS = 60000                # 训练迭代次数，可调整为30000/60000/100000
WHITE_BACKGROUND = True
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1. 清理旧输出 ----------
print("Cleaning old output...")
subprocess.run(["rm", "-rf", OUTPUT_DIR], check=True)

# ---------- 2. 训练模型 ----------
train_cmd = [
    "python", "train.py",
    "-s", DATA_ROOT,
    "-m", OUTPUT_DIR,
    "--eval",
    "--white_background" if WHITE_BACKGROUND else "",
    "--iterations", str(ITERATIONS)
]
print("Training model with 3DGS...")
subprocess.run([arg for arg in train_cmd if arg], check=True)

# ---------- 3. 渲染训练好的模型 ----------
render_cmd = [
    "python", "render.py",
    "-m", OUTPUT_DIR,
    "--skip_train",
    "--white_background" if WHITE_BACKGROUND else ""
]
print("Rendering trained model...")
subprocess.run([arg for arg in render_cmd if arg], check=True)

# ---------- 4. 自动生成白背景 GT ----------
gt_dir = os.path.join(OUTPUT_DIR, "test", f"ours_{ITERATIONS}", "gt")
os.makedirs(gt_dir, exist_ok=True)

with open(os.path.join(DATA_ROOT, "transforms_test.json"), "r") as f:
    meta = json.load(f)

print(f"Generating white-background GT in {gt_dir} ...")
for idx, frame in enumerate(meta["frames"]):
    img_path = os.path.join(DATA_ROOT, frame["file_path"] + ".png")
    img = Image.open(img_path).convert("RGBA")
    rgba = np.array(img).astype(np.float32) / 255.0

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    white = np.ones_like(rgb)

    comp = rgb * alpha + white * (1 - alpha)
    comp = (comp * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(comp).save(os.path.join(gt_dir, f"{idx:05d}.png"))

print("White-background GT generation done.")

# ---------- 5. 计算 metrics ----------
metrics_cmd = [
    "python", "metrics.py",
    "-m", OUTPUT_DIR
]
print("Computing metrics...")
subprocess.run(metrics_cmd, check=True)

print("All steps finished. Baseline ready for evaluation and presentation.")