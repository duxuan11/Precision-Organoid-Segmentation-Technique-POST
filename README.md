<h1 align="center">Precision Organoid Segmentation Technique (POST)</h1>

<div align='center'>
    <a href='https://scholar.google.com' target='_blank'><strong>Xuan Du</strong></a><sup> 1,3</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>YuChen Li</strong></a><sup>1,3</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Jiaping Song</strong></a><sup>4</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zilin Zhang</strong></a><sup>1</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Jing Zhang</strong></a><sup>4</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Yanhui Li</strong></a><sup>2*</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zaozao Chen</strong></a><sup>1*</sup>,&thinsp;
    <a href='https://scholar.google.com' target='_blank'><strong>Zhongze Gu</strong></a><sup>1*</sup>,&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>Southeast University&ensp;  <sup>2 </sup>Nanjing University&ensp;  <sup>3 </sup>Avatarget Biotechnology Co.&ensp;  <br/> <sup>4 </sup>Anhui Science and Technology University&ensp;
    <br />
</div>
<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <a href='https://www.sciopen.com/article/pdf/10.26599/AIR.2024.9150'><img src='https://img.shields.io/badge/Journal-Paper-red'></a>&ensp;
  <a href='https://arxiv.org/pdf/2401.'><img src='https://img.shields.io/badge/arXiv-Paper-green'></a>&ensp;
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-blue'></a>&ensp;
</div>

|            *Sample_1*            |             *Sample_2*            |             *Sample_3*            |
| :------------------------------: | :-------------------------------: | :-------------------------------: |
| <img src="https://drive.google.com/uc?id=19ShmEUc_lpIKASy6zUq5F89q3HRsdfvr" /> |  <img src="https://drive.google.com/uc?id=1tJEwPgfiK18r34n_qronNpuQhjd20NhV" /> |  <img src="https://drive.google.com/uc?id=1KtoevItyasK2samFIE0RfrQQ7cHo0QBZ" /> |
|            *Sample_4*            |             *Sample_5*            |             *Sample_6*            |
| <img src="https://drive.google.com/uc?id=1I5UW-xzCLMQD0IhlRwdCHqz8nFMxUa8X" /> |  <img src="https://drive.google.com/uc?id=1BbR6YjzI8xKwCQL-t-5pul5PcLRrJNrs" /> |  <img src="https://drive.google.com/uc?id=1puGcK3elZ3__nfKTa2iVcMajLhLaG1qQ" /> |
<br/>
This repo is the official implementation of "[**Precision Organoid Segmentation Technique (POST): an Accurate Algorithm for Segmentation of any Challenging Bright-field Organoid Images.**]

## 📑 Documentation 
> This is about the organoid/organ-on-a-chip intelligent segmentation algorithm.

### 🚀 Installation 

#### Environment Setup (uv)

依赖由 [**uv**](https://docs.astral.sh/uv/) 管理：Python 版本写在 `.python-version`（3.10），
依赖声明在 `pyproject.toml`，精确版本锁在 `uv.lock`。原来的 `requirements.txt` 已被移除，
请不要再用 `pip install -r requirements.txt`。

**1. Install uv**
```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and Install this repository:**
```bash
git clone https://github.com/duxuan11/Precision-Organoid-Segmentation-Technique-POST.git
cd Precision-Organoid-Segmentation-Technique-POST
```

**3. Create the environment** —— 二选一（两个 extra 互斥，见 `[tool.uv] conflicts`）：
```bash
uv sync --extra gpu   # CUDA 11.8 + cuDNN 8：onnxruntime-gpu + torch(cu118)，推荐
uv sync --extra cpu   # 纯 CPU：onnxruntime + torch(cpu)
```
`uv` 会自动下载 Python 3.10 并在仓库根目录创建 `.venv/`，同时安装 `dev` 依赖组
（含测试用来合成假 ONNX 图的 `onnx`；只想要运行期依赖时加 `--no-dev`）。
从 CPU 切到 GPU（或反向）时再执行一次 `uv sync --extra gpu`，uv 会自动移除多余的包。

**4. Run** —— `uv run` 会自动使用 `.venv`，不必手动 activate：
```bash
uv run python inference.py --yolo_weight weights/yolo8_3_25.onnx \
    --seg_encoder_weight weights/tinysam_encoder.onnx \
    --seg_decoder_weight weights/tinysam_decoder.onnx \
    --img_path img/example1.tiff --output_path result.png
```
也可以在 shell 里激活：Linux/macOS `source .venv/bin/activate`，Windows
`.venv\Scripts\activate`（PowerShell 用 `.venv\Scripts\Activate.ps1`）。

其它常用命令：
```bash
uv lock --upgrade            # 升级锁文件里的版本
uv run python -m unittest discover -s tests -v   # 跑测试
uv sync --extra gpu --reinstall  # 依赖状态异常时强制重建
```

## 🚀 Quick Start: Inference & Visualization

### Inference
Before run this script, you should download pre-trained models, and put them into weights file. 
```shell
python inference.py --yolo_weight weights/yolo8_3_25.onnx --seg_encoder_weight weights/tinysam_encoder.onnx --seg_decoder_weight weights/tinysam_decoder.onnx --img_path img/example1.tiff --output_path result.png
```

#### Speed / memory tuning flags

All of these have sensible defaults, the flags below only exist to trade speed against memory
on your own hardware.

| Flag | Default | Effect |
| --- | --- | --- |
| `--batch_size` | `8` | Number of image slices pushed through the YOLO ONNX model in one `session.run()`. Needs an ONNX graph exported with a dynamic batch dimension (`model.export(format="onnx", dynamic=True)`); otherwise the code automatically falls back to one slice at a time. |
| `--standard_pred` | off | Also run the detector on the whole image. It costs roughly one extra full pass and is only useful for very large objects the slices might miss. |
| `--seg_bbox_batch_size` | `50` | Boxes decoded per TinySAM decoder call. Lower it if you run out of VRAM. |
| `--seg_mask_scale` | `1.0` | Resolution factor of the decoded masks. `0.5` gives ~4x less decoder output (VRAM/host memory) and barely changes the drawn contours. |
| `--seg_encoder_device` | `cpu` | Where the TinySAM image encoder runs. Try `cuda:0` if the host CPU is slow. |
| `--seg_decoder_device` | `cuda:0` | Where the TinySAM mask decoder runs. |

Other changes that are always active: the input image is decoded once and shared between both
stages, all slices are pre-processed as a single batch, the TinySAM decoder is called directly
instead of through one joined thread per batch, mask bounding boxes are computed with
`cv2.boundingRect` instead of `np.argwhere`, and NMS runs on a vectorised numpy/torchvision
implementation.

Verify the refactor on your machine with:
```shell
uv run python -m unittest discover -s tests -v          # equivalence tests (no weights required)
uv run python tools/bench_slice_batching.py --img img/example1.tiff   # detector batching benchmark
uv run python tools/bench_seg_decoder.py --boxes 300                  # mask decoder benchmark
```

#### Segmentation quality flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--enhance` | `none` | Contrast enhancement for the **TinySAM encoder input only** (the overlay still shows the original image). `clahe` = percentile clipping (kills specular highlights/bubble glare) + adaptive histogram equalisation, `lowcontrast` = the same but gated to images that look low contrast. Ported from [cellSAM](https://github.com/vanvalenlab/cellSAM). Kept `none` by default because the shipped encoder was fine-tuned with a fixed mean/std normalisation - validate on your own data before changing it. |
| `--seg_iou_threshold` | `0.5` | Drop masks whose predicted IoU (the decoder's own score) is below this, so a failed mask prompt never reaches the result. `0` disables the gate. Graphs that return no score tensor disable it automatically. |
| `--seg_min_area` | `25` | Drop masks smaller than this many pixels. `0` keeps everything. |
| `--mask_output` | none | Write an **instance label map** (`0` = background, one id per mask) next to the visualization. |

#### Instance label map

`--output_path` keeps producing the blended PNG/SVG-style overlay for humans.
`--mask_output` additionally writes a machine-readable mask for counting, area and
morphology measurements:

```shell
python inference.py --yolo_weight weights/yolo8_3_25.onnx \
    --seg_encoder_weight weights/tinysam_encoder.onnx \
    --seg_decoder_weight weights/tinysam_decoder.onnx \
    --img_path img/example1.tiff --output_path result.png \
    --mask_output result_mask.png
```

* `.npy` keeps the native `uint32` ids;
* any other extension is written as a **16-bit** image (`cv2.imread(path, cv2.IMREAD_UNCHANGED)`),
  and ids above 65535 raise an error instead of wrapping around;
* ids are assigned in box order and a later mask overwrites an earlier one where they
  overlap, the same rule the overlay uses;
* the map is always at the **original image resolution** (nearest-neighbour upscaled from the
  1024px decoder canvas, so no new label values are invented).

In Python:

```python
from POST.tinysam.tiny_sam import TinySam, save_label_map
from inference import tinysam_inference, object_detection_inference, read_image

image = read_image("img/example1.tiff")
boxes = object_detection_inference(image, detector)
tinysam = TinySam(encoder_path=..., decoder_path=..., enhance="clahe")
blend, label_map = tinysam_inference(image, boxes, tinysam, output_path="result.png")
save_label_map("result_mask.png", label_map)   # or .npy for uint32 ids
```

### Training
**1. YOLO Packages:**
`ultralytics==8.0.224` 已包含在 `pyproject.toml` 里，`uv sync --extra gpu`（或 `--extra cpu`）
之后就装好了，不需要再单独 `pip install`。
**2. Run Training Script:**
```shell
uv run python train.py --model_path weights/yolo8m.pt --data_path_yaml organoid.yaml --epochs 100 --img_size 1024 --batch_size 16 --workers 8
```
`train.py` now honours `--device` (previously it was hard-coded to GPU 0), uses mixed precision by
default (`--no_amp` to disable) and accepts `--workers` / `--cache` to speed up the dataloader.
You can refer to the training process of [yolov8](https://github.com/ultralytics/ultralytics) and [yolov9](https://github.com/WongKinYiu/yolov9), and download the pre-trained model.Then use these trained models in this code.
**Notes:** Since Tinysam has already been fine-tuned on a large organoid dataset and converted to ONNX, no additional training is required.
### Pre-trained Models
Download the pre-trained model weights from: [Google Drive](https://drive.google.com/drive/folders/1-Dd-zFxHM2GfprqbEv2Tv0_mLNu88SuW?usp=sharing)<br>
Download the yolov8 pre-trained model weights from: [Github](https://docs.ultralytics.com/zh/models/yolov8/#performance-metrics)
## License
Commercial use of the Software, including without limitation use in products or services, use by for-profit entities, or use for any purpose intended to generate revenue, is strictly prohibited without the prior written permission of the copyright holder. For commercial licensing inquiries, please contact [duxuan9470@gmail.com].

## Acknowledgement
Our project is developed based on [yolov8](https://github.com/ultralytics/ultralytics),[yolov9](https://github.com/WongKinYiu/yolov9), and [TinySAM](https://github.com/xinghaochen/TinySAM).

## 📖 Citation

If you find our work useful, please cite our paper:
```bash
@inproceedings{Du2025POST,
  title={Precision Organoid Segmentation Technique (POST): an Accurate Algorithm for Segmentation of any Challenging Organoids.},
  author={Xuan Du and Yuchen Li and Jiaping Song and Zilin Zhang and Jing Zhang and Yanhui Li and Zaozao Chen and Zhongze Gu},
  booktitle={Bio-Design and Manufacturing},
  year={2025}
}
```