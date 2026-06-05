# NeRF Synthetic Lego 白背景训练与评估流程

本文档用于复现 `data/lego` 上的白背景训练、渲染和指标评估，并检查
`view.original_image`、render 保存的 GT、metrics 读取的 GT 是否一致。

## 0. 目标

训练命令使用：

```bash
python train.py \
  -s data/lego \
  -m output/lego_white_fixed_30k \
  --eval \
  --white_background \
  --iterations 30000
```

预期修复后：

- NeRF Synthetic/Blender RGBA 图像会按 `--white_background` 合成为 RGB 白背景 GT。
- `train.py` 中的 `viewpoint_cam.original_image` 是白背景 GT。
- `render.py` 保存到 `test/ours_xxxxx/gt` 的 GT 与训练 GT 同源。
- `metrics.py` 按同名文件读取 `renders/00000.png` 与 `gt/00000.png`。
- Lego PSNR 应明显高于此前约 15 的异常结果。

## 1. 补丁状态检查

在仓库根目录执行：

```bash
grep -n "def load_image\|cam_info.image\|image: object\|is_test=is_test, image=image\|image_path = cam_name\|Path(args.source).resolve" \
  scene/dataset_readers.py utils/camera_utils.py tools/debug_lego_pipeline.py
```

正确状态应满足：

```text
utils/camera_utils.py: 有 def load_image
scene/dataset_readers.py: 有 image_path = cam_name
tools/debug_lego_pipeline.py: 有 Path(args.source).resolve()
```

不应出现：

```text
cam_info.image
image: object
is_test=is_test, image=image
```

注意：如果看到下面这一行是正常的，它只是把 PIL 图像传给 `Camera(...)`：

```python
image=image, invdepthmap=invdepthmap
```

## 2. 清理旧输出和缓存

旧模型是按错误 GT 训练出来的，不能复用。重新训练前先清理：

```bash
rm -rf output/lego_white_fixed_30k
rm -rf output/lego_white_fixed_60k
find scene utils tools -name "__pycache__" -type d -exec rm -rf {} +
```

## 3. 训练前诊断

先不要直接长训，先确认 `view.original_image` 等于手动白底合成 GT：

```bash
python tools/debug_lego_pipeline.py \
  -s data/lego \
  -m output/lego_white_fixed_30k \
  --iteration 30000
```

训练前 render/gt 还不存在，所以看到 `Missing render or gt` 是正常的。

关键输出必须类似：

```text
view/manual mean abs diff: 0.000000
判断: view.original_image 等于手动白背景 GT
```

如果这里不是 0，先不要训练，继续检查补丁是否同步。

## 4. 正式训练 30000 iter

```bash
mkdir -p output

python train.py \
  -s data/lego \
  -m output/lego_white_fixed_30k \
  --eval \
  --white_background \
  --iterations 30000 \
  2>&1 | tee output/lego_white_fixed_30k_train.log
```

## 5. Render Test

```bash
python render.py \
  -m output/lego_white_fixed_30k \
  --skip_train \
  --white_background \
  --iteration 30000 \
  2>&1 | tee output/lego_white_fixed_30k_render.log
```

输出目录：

```text
output/lego_white_fixed_30k/test/ours_30000/renders
output/lego_white_fixed_30k/test/ours_30000/gt
```

## 6. Metrics

```bash
python metrics.py \
  -m output/lego_white_fixed_30k \
  2>&1 | tee output/lego_white_fixed_30k_metrics.log
```

结果文件：

```text
output/lego_white_fixed_30k/results.json
output/lego_white_fixed_30k/per_view.json
```

## 7. 训练后诊断

训练、render、metrics 跑完后，再检查 render/gt 和 camera GT：

```bash
python tools/debug_lego_pipeline.py \
  -s data/lego \
  -m output/lego_white_fixed_30k \
  --iteration 30000 \
  2>&1 | tee output/lego_white_fixed_30k_debug.log
```

重点看：

```text
view/manual mean abs diff
render/gt mean abs diff
render/gt simple PSNR
```

其中 `view/manual mean abs diff` 必须接近 0。

## 8. 打包 Render/GT 对比图和日志

```bash
zip -r output/lego_white_fixed_30k_test_pairs.zip \
  output/lego_white_fixed_30k/test/ours_30000/renders \
  output/lego_white_fixed_30k/test/ours_30000/gt \
  output/lego_white_fixed_30k/results.json \
  output/lego_white_fixed_30k/per_view.json \
  output/lego_white_fixed_30k_*_log
```

## 9. 可选：正式训练 60000 iter

如果 30000 iter 已经确认链路正确，可以跑 60000 iter：

```bash
rm -rf output/lego_white_fixed_60k

python train.py \
  -s data/lego \
  -m output/lego_white_fixed_60k \
  --eval \
  --white_background \
  --iterations 60000 \
  --test_iterations 7000 30000 60000 \
  --save_iterations 7000 30000 60000 \
  2>&1 | tee output/lego_white_fixed_60k_train.log
```

Render：

```bash
python render.py \
  -m output/lego_white_fixed_60k \
  --skip_train \
  --white_background \
  --iteration 60000 \
  2>&1 | tee output/lego_white_fixed_60k_render.log
```

Metrics：

```bash
python metrics.py \
  -m output/lego_white_fixed_60k \
  2>&1 | tee output/lego_white_fixed_60k_metrics.log
```

诊断：

```bash
python tools/debug_lego_pipeline.py \
  -s data/lego \
  -m output/lego_white_fixed_60k \
  --iteration 60000 \
  2>&1 | tee output/lego_white_fixed_60k_debug.log
```

打包：

```bash
zip -r output/lego_white_fixed_60k_test_pairs.zip \
  output/lego_white_fixed_60k/test/ours_60000/renders \
  output/lego_white_fixed_60k/test/ours_60000/gt \
  output/lego_white_fixed_60k/results.json \
  output/lego_white_fixed_60k/per_view.json \
  output/lego_white_fixed_60k_*_log
```

## 10. 如果 PSNR 仍低于 20

按下面顺序排查，不要先增加 iterations：

1. 检查补丁状态：

   ```bash
   grep -n "def load_image\|cam_info.image\|image: object\|is_test=is_test, image=image\|image_path = cam_name\|Path(args.source).resolve" \
     scene/dataset_readers.py utils/camera_utils.py tools/debug_lego_pipeline.py
   ```

2. 检查 `view.original_image` 是否等于手动白底 GT：

   ```bash
   grep -n "view/manual mean abs diff\|判断" output/lego_white_fixed_30k_debug.log
   ```

3. 检查 render 和 gt 的统计：

   ```bash
   grep -n "render/gt mean abs diff\|render/gt simple PSNR" output/lego_white_fixed_30k_debug.log
   ```

4. 确认 render 使用的是新训练输出：

   ```bash
   ls output/lego_white_fixed_30k/point_cloud/iteration_30000
   ls output/lego_white_fixed_30k/test/ours_30000/renders/00000.png
   ls output/lego_white_fixed_30k/test/ours_30000/gt/00000.png
   ```

5. 如果 `view/manual mean abs diff` 是 0，但指标仍异常低，再继续查相机位姿、frame 顺序、颜色范围和训练日志。

