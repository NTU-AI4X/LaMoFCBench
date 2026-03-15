# LaMoFCBench

LaMoFCBench is a benchmark and evaluation toolkit for universal feature coding across multiple large-model modalities.

## 1. Project Overview

This repository currently covers four task groups:

- Common Vision Understanding (CVU), model family: DINOv3-ViT7B
- Common Language Understanding (CLU), model families: Qwen3-8B, FalconMamba-7B
- Common Audio Understanding (CAU), model family: KimiAudio-7B
- Controllable Text-to-Image (CTTI), model family: StableDiffusion3.5 + ControlNet

Core directories:

- `coding/`: feature coding pipeline (`feature_coding.py`) and batch launcher (`feature_coding.sh`)
- `machine/`: downstream task evaluation scripts
- `lmfc_utils/handlers/`: feature parsers/packers/unpackers
- `lmfc_utils/custom_codecs/`: learned image codec wrappers used by feature coding
- `lmfc_utils/transform_mapping/`: quantization mapping files

## 2. Data and Feature Resources

All hosted resources are under:

- <https://www.modelscope.cn/collections/yooweey/LaMoFCBench>

Main datasets:

- Raw datasets: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-RawDatasets>
- Raw extracted features:
  - DINOv3: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3>
  - Qwen3/FalconMamba: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-LargeLanguageModel>
  - KimiAudio: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-KimiAudio>
  - SD3.5 + ControlNet: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-StableDiffusion3.5Large>
- Post-coding features:
  - DINOv3:
    - <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3TotalCls-AfterCodec>
    - <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3TotalSegHyperprior-AfterCodec>
    - <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3TotalSegELIC-AfterCodec>
    - <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3TotalDepHyperprior-AfterCodec>
    - <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-DINOv3TotalDepELIC-AfterCodec>
  - Qwen3: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-Qwen3LLM-AfterCodec>
  - FalconMamba: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-FalconMambaLLM-AfterCodec>
  - KimiAudio: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-KimiAudio-AfterCodec>
  - SD3.5 + ControlNet: <https://www.modelscope.cn/datasets/yooweey/FeatureCoding-StableDiffusion3.5ControlNet-AfterCodec>

## 3. Quick Start

### 3.1 Environment

Recommended baseline:

- Python 3.10+
- PyTorch + CUDA (for GPU runs)
- `compressai`, `einops`, `zstandard`, `tabulate`
- task-specific dependencies used by scripts under `machine/`

### 3.2 Minimal Roundtrip Command

Run from the `coding` directory:

```bash
cd coding
python feature_coding.py roundtrip <INPUT_FILE_OR_DIR> \
  --output <OUTPUT_DIR> \
  --handler qwen \
  --strategy individual \
  --arch hyperprior-featurecoding \
  --checkpoint <CHECKPOINT_PATH>
```

Notes:

- valid `--handler` values come from `lmfc_utils/handlers/__init__.py`
- default mapping config is `lmfc_utils/transform_mapping/10samples-8bits/mapping.json`

## 4. Feature Coding Pipeline

Entry: `coding/feature_coding.py`

Workflow:

1. Parse raw features with a handler.
2. Pack tensors by strategy.
3. Apply nonlinear quantization + learned codec compression.
4. Decode and unpack features.
5. Compute BPFP/EBPFP/MSE.
6. Save reconstructed feature payloads (`.zst`) with metadata.

## 5. Downstream Evaluation Scripts

Main scripts:

- CVU:
  - `machine/cvu/dinov3cls.py`
  - `machine/cvu/dinov3dep.py`
  - `machine/cvu/dinov3seg.py`
- CLU:
  - `machine/clu/qwen3.py`
  - `machine/clu/falconmamba.py`
- CAU:
  - `machine/cau/kimiaudio.py`
- CTTI:
  - `machine/ctti/sd35cond.py`

These scripts load reconstructed features from `--load_root`, inject them into task inference, and report task-specific metrics.

## 6. Reproducibility

### 6.1 Pin Repository Version

```bash
git rev-parse HEAD
git branch --show-current
git status --short
```

Record at least:

- commit hash
- branch name
- whether the working tree is clean

### 6.2 Record Experiment Metadata

For every run, keep:

- checkpoint path
- handler + strategy
- transform mapping file
- dataset split / sample count
- output log path and generated `.zst` path

## 7. Troubleshooting

### 7.1 Handler Naming

- `feature_coding.py` validates handlers against `AVAILABLE_HANDLERS`.
- for DINOv3 total features, handler name is `dinov3-total`.

### 7.2 Runtime Environment

- many scripts are designed for offline/local-cache workflows.
- verify model weights, dataset paths, and cache directories before running.

### 7.3 Working Directory

- current `feature_coding.py` import path setup assumes running from `coding/`.
- if you run from another directory, resolve import paths first.

## 8. Suggested Execution Order

1. Prepare raw features and codec checkpoints.
2. Run `coding/feature_coding.py` to generate reconstructed features.
3. Run task scripts under `machine/`.
4. Compare metrics with baseline.
5. Archive logs, configs, and Git commit hash.
