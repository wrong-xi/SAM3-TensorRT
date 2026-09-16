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

4) Export the fixed `door handle` model to ONNX on Windows

The `feat/fixed-door-handle` branch accepts exactly one fixed prompt. The
two-class implementation remains on `jetson-xxx`. The models and C++ interfaces
are not interchangeable: this branch requires a semantic-only engine and
rejects both the old two-class engine and the earlier single-class engine
with a presence output. Re-export ONNX, rebuild the engine, and recompile C++.

Edit `MODEL_DIR`, `SAMPLE_IMAGE`, and `IMAGE_SIZE` at the top of
`python/onnxexport.py` if needed. The script loads the Hugging Face
`model.safetensors` checkpoint in `MODEL_DIR`; it does not load `sam3.pt`.

```powershell
conda activate sam3
cd C:\code\xxx\sam3-main\SAM3-TensorRT
python python\onnxexport.py
```

This produces
`python/onnx_weights_handle/sam3_door_handle_semantic.onnx`. The new filename
preserves the previous `sam3_door_handle.onnx` for comparison. Depending on the ONNX
exporter's size handling, it may also create external weight files. Copy the
entire `onnx_weights_handle` directory to `/data/code/SAM3-TensorRT/onnx_weights_handle`
on the Jetson; files under `_weights`
are external ONNX initializers, not duplicate models.

At the end of export, the script fixes decoder `If` nodes emitted for
`squeeze(-1)` when needed for TensorRT compatibility. Single-prompt export can
already eliminate these nodes; a replacement count of zero is not an error.

The exported graph has one input and one output:

- `pixel_values`: `[1, 3, 672, 672]`
- `semantic_logits`: `[1, 1, 192, 192]` (the only channel is `door handle`)

The image encoder runs once, followed by a single-item prompt batch. Text
features for `door handle` are precomputed constants: there is no runtime text
input or text encoder. Only the semantic result is exported, so the unused
presence scoring branch is pruned. The old two-class ONNX is also left intact.

5) Build a TensorRT engine on the Jetson

TensorRT engines are platform- and TensorRT-version-specific, so build the
engine on the target Jetson rather than copying an engine built on Windows.

```bash
cd /data/code/SAM3-TensorRT
/usr/src/tensorrt/bin/trtexec \
  --onnx=onnx_weights_handle/sam3_door_handle_semantic.onnx \
  --saveEngine=onnx_weights_handle/sam3_672_door_handle_semantic_fp16.engine \
  --fp16 \
  --skipInference
```

6) Build the C++/CUDA library and sample app
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j4
```

7) Run the demo app
```bash
./cpp/build/sam3_pcs_app <image_dir> onnx_weights_handle/sam3_672_door_handle_semantic_fp16.engine
```

By default, each image produces a single-channel, original-resolution PNG in
`results/masks/`: `128=door handle`, `255=background` (no label 0). An image with
no selected foreground produces an all-255 mask. PNG preserves these label
values exactly. Existing stem-based naming is preserved: `frame.jpg` becomes
`results/masks/frame_mask.png`, and its optional overlay is `results/vis/frame_vis.png`.
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
./cpp/build/sam3_pcs_app <image_dir> <single_class_engine.engine> 0 1
```

Overlays go to `results/vis/` and are generated from the final mask after the
same inference. The demo's `vis_alpha` controls their fixed foreground opacity;
background pixels keep their original color. `results/` is relative to the
working directory. The third argument remains `benchmark`: `1` disables all
output saving, even when `save_vis` is enabled. Changing `save_vis` on the command
line needs neither recompilation nor an engine rebuild.

The demo applies only a pixel-mask threshold to the handle:

```cpp
const float handle_mask_threshold = 0.5F;
SAM3_PCS pcs(engine_path, 0.6F, handle_mask_threshold);
```

There is no whole-image presence gate. A pixel is selected when its probability
is greater than or equal to `handle_mask_threshold`. Changing this C++ constant
requires recompilation, but does not require another engine build. Sigmoid runs on
the low-resolution map once, and GPU bilinear interpolation (`align_corners=False`)
precedes pixel thresholding. Scratch storage is one reusable mask plane
(144 KiB at 192x192). The CUDA postprocessor supports
a three-channel overlay with
`VIS_SEMANTIC_SEGMENTATION`, or an efficient single-channel label map with
`VIS_CLASS_MAP` (`128=door handle`, `255=background`). Construct the
label-map result as `CV_8UC1` before calling `pin_opencv_matrices`. Use
`VIS_NONE` only when CPU code needs the raw logits. After a `VIS_NONE` call,
use `semantic_logits_host()` rather than relying
on TensorRT output ordering.

This restores the earlier pixel-only selection policy. Validate both images
with handles and images without handles: removing the extra presence gate can
increase false positives compared with a correctly working gated model.


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
- This branch exports semantic logits only; it does not provide presence scores or instance masks. With `SAM3_VISUALIZATION::VIS_NONE`, apply sigmoid yourself.
- The library does not support building engines. Use `trtexec` instead.

### Benchmarking
Use the same image directory and prompt for all runs. The C++ benchmark mode
excludes image load/save but includes preprocessing, inference and mask
postprocessing. Normal mode also times PNG saving and optional visualization.
Keep each input directory at one image resolution: the app pins buffers based
on its first image.

The C++ app warms up the complete inference pipeline five times on the first
image, without saving output or adding timing samples. It then processes every
input image, including that first image. Cumulative averages are printed every
10 images, with a final summary even for fewer than 10 images.

GPU timings use reusable, timing-enabled `cudaEvent` objects on `sam3_stream`.
There is one checked stream synchronization per frame, after all GPU work and
copies are submitted; there are no synchronizations between measured stages.

| Field (ms/image) | Measurement |
| --- | --- |
| `preprocess` | CUDA image resize, color conversion and normalization |
| `TensorRT` | GPU stream interval surrounding `enqueueV3`, not CPU enqueue duration |
| `postprocess` | Low-resolution sigmoid, bilinear upsampling and mask/overlay selection |
| `H2D`, `D2H` | Explicit image/output copies on dGPU, excluded from the three stages |
| `total` | GPU stream interval spanning copies and the three stages |
| `CPU wall (infer+save)` | `steady_clock` elapsed time including inference, synchronization and optional output saving; excludes image reading/decoding |

On Jetson's zero-copy path, `H2D` and `D2H` are zero because there are no explicit
copies. Shared-memory access costs remain part of the kernels that perform them.
Event intervals measure elapsed GPU-stream time, not the sum of individual
kernel active times: GPU contention and host submission gaps can affect them.
The total may differ slightly from the sum because of empty-stage event gaps
and display rounding. Avoid other GPU workloads when comparing measurements.

After a successful `infer_on_image`, `last_gpu_timings()` returns the latest
frame's `Sam3GpuTimings` (milliseconds). `VIS_NONE` has zero postprocess time;
`run_blind_inference()` reports only TensorRT and total time. Statistics are
collected by the app, so library users need not perform the app's warmup policy.
This instrumentation changes only C++; reuse the current semantic-only engine
after recompiling the application and library.

Huggingface + PyTorch:
```bash
python python/basic_script.py <image_dir>
```

TensorRT + CUDA (benchmark mode disables output writes):
```bash
./cpp/build/sam3_pcs_app <image_dir> <single_class_engine.engine> 1 0
```

Timing integration tests (requires a compatible engine, a PNG image, Python
with NumPy/OpenCV, and the C++ dependencies):

```bash
python3 cpp/tests/test_timing_cli.py --app cpp/build/sam3_pcs_app \
  --engine <single_class_engine.engine> --image <sample.png> --work-dir cpp/build -v
g++ -std=c++17 cpp/tests/test_gpu_timing.cpp -Icpp/include -I/usr/local/cuda/include \
  $(pkg-config --cflags --libs opencv4) -Lcpp/build -lsam3_trt \
  -L/usr/local/cuda/lib64 -lcudart -lnvinfer -o cpp/build/test_gpu_timing
LD_LIBRARY_PATH="$PWD/cpp/build:$LD_LIBRARY_PATH" \
  ./cpp/build/test_gpu_timing <single_class_engine.engine> <sample.png>
```

### ONNX Export Details
- Export size is fixed by `IMAGE_SIZE` in `python/onnxexport.py`; TensorRT does
  not change it later. The C++ preprocessor resizes source images to this
  engine input size.
- Export uses CUDA when available and falls back to CPU.
- SAM3 is large and may export with external weight shards; keep the entire
  `onnx_weights_handle/` directory together.

### TensorRT Notes
- Use `trtexec` for quick engine builds and benchmarking.
- FP16 is the usual starting point; INT8/FP8/INT4 require calibration or compatible tooling.

### License
- MIT (see `LICENSE`).

If this saved you time, drop a ⭐ so others can find it and ship SAM-3 faster.

# Disclaimer
All views expressed here are my own. This project is not affiliated with my employer.
