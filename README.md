# SAM3 → TensorRT

Export Meta AI's Segment Anything 3 (SAM3) model to ONNX, then build a TensorRT engine for real-time segmentation. This repo includes a CUDA inference library and demo apps for semantic and instance segmentation.

## Table of Contents
- [Project Overview](#project-overview)
- [Benchmarks](#benchmarks)
- [Demos](#demos)
- [Repo Layout](#repo-layout)
- [Quickstart](#quickstart)
  - [On x86](#on-x86)
  - [On Jetson/Spark](#on-jetsonspark)
- [Extensions](#extensions)
- [Troubleshooting](#troubleshooting)
- [Development guide](#development-guide)
  - [CUDA Library Notes](#cuda-library-notes)
  - [Benchmarking](#benchmarking)
  - [ONNX Export Details](#onnx-export-details)
  - [TensorRT Notes](#tensorrt-notes)
  - [License](#license)
- [Disclaimer](#disclaimer)

## Project Overview
- Python tooling to export SAM3 to a clean ONNX graph.
- TensorRT-ready workflows for building optimized engines.
- A C++/CUDA library for high-performance inference with demo apps.
- Support for Promptable concept segmentation (PCS), the latest feautre in SAM3.
- Zero-copy support on unified-memory platforms (Jetson, DGX Spark). Great for robotics/real-time interaction.
- Everything runs inside a reproducible docker environment (x86, Jetson, Spark).
- MIT license for the love of everything nice :)

## Benchmarks
The numbers show end to end image processing latency per image (4K resolution) in ms excluding image load/save time.

| Hardware | HF+PyTorch | TensorRT+CUDA | Speedup | Notes |
| --- | --- | --- | --- | --- |
| Jetson Orin NX | 6600 ms | 950 ms | 6.95x | Uses zero-copy |
| Jetson Thor | 1709.8 ms | 125.9 ms | 13.58x | R38.2.2, CUDA 13.0, TRT 10.13.3, MAXN, zero-copy, 10x 4K images |
| DGX Spark |  |  |  | Please contribute |
| RTX 3090 | 438 ms | 75 ms | 5.82x |  |
| RTX 5090 | 120.9 ms | 24.8 ms | 4.88x | COCO val2017, TRT 10.14.1 |
| A10 | 545.3 ms | 161.1 | 3.38x | GPU hits 100% utilization |
| A100 | 314.1 ms | 48.8 ms | 6.43x | 40GB SXM4 variant |
| H100 | 265.3 ms | 34.6 ms | 7.66x | PCIe variant |
| H100 | 213.2 ms | 24.9 ms | 8.56x | SXM5 variant |
| GH200 | 142.3 ms | 23.3 ms | 6.11x | arm64+H100 iGPU, without zero-copy |
| GH200 | 142.3 ms | 26.4 ms | 5.39x | using zero-copy |
| B200 | 160.0 ms | 17.7 ms | 9.03x | SXM6 variant |

Note: the HF+PyTorch path is GPU-backed too, so these numbers compare two GPU implementations rather than CPU vs GPU.

Please contribute your results and I will be happy to add them here. Use [this guide](#benchmarking) to run the benchmarks yourself.

## Demos
Video demo (click to play):
[![Semantic segmentation demo video](https://img.youtube.com/vi/hHvhQ514Evs/maxresdefault.jpg)](https://youtube.com/shorts/hHvhQ514Evs?feature=share)

Semantic segmentation produced by the C++ demo app (`prompt='dog'`)

<img src="demo/semantic_puppies.png" width="640" alt="Semantic segmentation demo">

Instance segmentation results (`prompt='box'`)

<img src="demo/instance_box.jpeg" width="800" alt="Instance segmentation demo">

## Repo Layout
- `python/` - ONNX export and visualization scripts.
- `cpp/` - C++/CUDA library and apps (TensorRT inference).
- `docker/` - Container setup (`Dockerfile.x86`, with an aarch64 variant expected).
- `demo/` - Example outputs from the C++ demo app.

## Quickstart

1) Request access to the gated model
   - Visit https://huggingface.co/facebook/sam3 and request access.
   - Ensure your `HF_TOKEN` has permission.
   - Set `HF_TOKEN` as environment variable in the host. Docker will pick it up from there.

2) Build the Docker container for your platform (all commands below run inside it)

### On x86
```bash
docker build -t sam3-trt -f docker/Dockerfile.x86 .
```

### On Jetson/Spark

For aarch64 platforms with shared CPU/GPU memory, the C++ library in this repo supports zero-copy inference paths.

Build and run the aarch64 container:
```bash
docker build -t sam3-trt-aarch64 -f docker/Dockerfile.aarch64 .
```

3) Export `HF_TOKEN` and run the docker container 

```bash
export HF_TOKEN=<YOUR TOKEN>
docker run -it --rm \
  --network=host \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --env HF_TOKEN \
  -v "$PWD":/workspace \
  -w /workspace \
  sam3-trt bash
```

4) Export the fixed `door` + `door handle` model to ONNX on Windows

Edit `MODEL_DIR`, `SAMPLE_IMAGE`, and `IMAGE_SIZE` at the top of
`python/onnxexport.py` if needed. The script loads the Hugging Face
`model.safetensors` checkpoint in `MODEL_DIR`; it does not load `sam3.pt`.

```powershell
conda activate sam3
cd C:\code\xxx\sam3-main\SAM3-TensorRT
python python\onnxexport.py
```

This produces
`python/onnx_weights_fixed/sam3_door_door_handle.onnx`. Depending on the ONNX
exporter's size handling, it may also create external weight files. Copy the
entire `onnx_weights_fixed` directory to the Jetson; files under `_weights`
are external ONNX initializers, not duplicate models.

At the end of export, the script removes the six decoder `If` nodes emitted
for `squeeze(-1)` that TensorRT 10.3 cannot parse. The expected message for
this SAM3 configuration is `replaced 6 decoder squeeze If node(s)`.

The exported graph has one input and two outputs:

- `pixel_values`: `[1, 3, 672, 672]`
- `semantic_logits`: `[2, 1, 192, 192]` (`0=door`, `1=door handle`)
- `presence_logits`: `[2, 1]`

The image encoder runs once. Its feature maps are reused by a two-item prompt
batch for the detector and mask heads. Both text embeddings are constants in
the ONNX graph, so C++ does not pass token IDs at runtime.

5) Build a TensorRT engine on the Jetson

TensorRT engines are platform- and TensorRT-version-specific, so build the
engine on the target Jetson rather than copying an engine built on Windows.

```bash
trtexec \
  --onnx=python/onnx_weights_fixed/sam3_door_door_handle.onnx \
  --saveEngine=sam3_door_door_handle_fp16.plan \
  --fp16 \
  --verbose
```

6) Build the C++/CUDA library and sample app
```bash
mkdir cpp/build && cd cpp/build
cmake ..
make
```

7) Run the demo app
```bash
./sam3_pcs_app <image_dir> <engine_path.engine>
```

By default, each image produces a single-channel, original-resolution PNG in
`results/masks/`: `0=door`, `128=door handle`, `255=background`. An image with
no selected foreground produces an all-255 mask. PNG preserves these label
values exactly. Existing stem-based naming is preserved: `frame.jpg` becomes
`results/masks/frame_mask.png`, with an optional overlay at
`results/vis/frame_vis.png`.

Input subdirectories are scanned recursively and their paths relative to the
input root are preserved: `A/B/frame.jpg` becomes
`results/masks/A/B/frame_mask.png` and, when enabled,
`results/vis/A/B/frame_vis.png`. Images in different subdirectories may share
the same name. Keep source stems unique within each input subdirectory.
Keep `results/` outside the input directory so generated images are not read
as inputs. All input images, including those in subdirectories, must use the
same resolution, as required by the existing reusable inference buffers.

Visualization saving is disabled by default (`save_vis=false`). Enable it
with the fourth argument:

```bash
# Save masks and additional color overlays
./sam3_pcs_app <image_dir> <engine_path.engine> 0 1
```

Overlays go to `results/vis/` and are generated from the final mask after the
same inference. The demo's `vis_alpha` controls their fixed foreground opacity;
background pixels keep their original color. `results/` is relative to the
working directory. The third argument remains `benchmark`: `1` disables all
output saving, even when `save_vis` is enabled. Changing `save_vis` on the command
line needs neither recompilation nor an engine rebuild.

The demo applies separate presence and pixel-mask thresholds to the two
classes:

```cpp
const SAM3_CLASS_THRESHOLDS door_thresholds = {0.50F, 0.50F};
const SAM3_CLASS_THRESHOLDS handle_thresholds = {0.45F, 0.55F};
SAM3_PCS pcs(engine_path, 0.3F, door_thresholds, handle_thresholds);
```

If only one class passes its own presence and mask thresholds, only that class
is drawn. If both masks cover the same pixel, `door handle` has priority. The
CUDA postprocessor supports a three-channel overlay with
`VIS_SEMANTIC_SEGMENTATION`, or an efficient single-channel label map with
`VIS_CLASS_MAP` (`0=door`, `128=door handle`, `255=background`). Construct the
label-map result as `CV_8UC1` before calling `pin_opencv_matrices`. Use
`VIS_NONE` only when CPU code needs the raw logits. After a `VIS_NONE` call,
use `semantic_logits_host()` and `presence_logits_host()` rather than relying
on TensorRT output ordering.


## Extensions
This is a very raw project and provides the crucial backend TensorRT/CUDA bits necessary for anything. From here, please feel free to fan out into any application you like. Pull requests are very welcome! Here are some ideas I can think of:
- ROS2 wrapper for real-time robotics pipelines.
- Interactive voice-based segmentation app. Have someone speak into a microphone, use a TTS model to transcribe it and feed into the engine, which then produces the segmentation mask live. I don't have the time to build it but I hope you can.
- Live camera input and overlays. You will need a beefy GPU. SAM3 doesn't run realtime on a Jetson nano.

## Troubleshooting
- **Access errors:** Make sure your `HF_TOKEN` has access to `facebook/sam3`.
- **ONNX export fails:** Install `transformers` from source if SAM3 is missing.
- **TensorRT `/sam3/detr_decoder/If` parse error:** Re-export with the current
  `python/onnxexport.py`; it applies the TensorRT compatibility rewrite after
  export.
- **Missing external ONNX data:** If export created an `_weights` directory,
  copy it together with the `.onnx` file without changing their relative paths.
- **C++ build errors:** Confirm CUDA, TensorRT, and OpenCV are installed and discoverable via `pkg-config`.

## Development guide

### CUDA Library Notes
- The shared library target is `sam3_trt`.
- Demo app: `sam3_pcs_app` (semantic/instance visualization modes).
- Outputs include semantic segmentation and instance segmentation mask logits. If you choose `SAM3_VISUALIZATION::VIS_NONE` in your application, you need to apply sigmoid yourself.
- The library does not support building engines. Use `trtexec` instead.

### Benchmarking
Use the same image directory and prompt for all runs. Both paths time the model pipeline and exclude image load/save.

Huggingface + PyTorch:
```bash
python python/basic_script.py <image_dir>
```

TensorRT + CUDA (benchmark mode disables output writes):
```bash
./sam3_pcs_app <image_dir> <engine_path.engine> 1
```

### ONNX Export Details
- Export size is fixed by `IMAGE_SIZE` in `python/onnxexport.py`; TensorRT does
  not change it later. The C++ preprocessor resizes source images to this
  engine input size.
- Export uses CUDA when available and falls back to CPU.
- SAM3 is large and may export with external weight shards; keep the entire
  `onnx_weights_fixed/` directory together.

### TensorRT Notes
- Use `trtexec` for quick engine builds and benchmarking.
- FP16 is the usual starting point; INT8/FP8/INT4 require calibration or compatible tooling.

### License
- MIT (see `LICENSE`).

If this saved you time, drop a ⭐ so others can find it and ship SAM-3 faster.

# Disclaimer
All views expressed here are my own. This project is not affiliated with my employer.
