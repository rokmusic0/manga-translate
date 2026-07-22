# ComicTrans

**See [report](report/main.pdf)**.

A local manga translation pipeline that:

1. detects Japanese text regions in manga pages,
2. extracts the Japanese text with OCR,
3. translates that text into English with a local LLM,
4. removes the original text from the image, and
5. redraws the English inside the detected regions.

The project is designed as a modular local pipeline so each stage can be inspected, debugged, and benchmarked separately.

## How it works

The pipeline has 4 stages:

- **OCR / text-region detection** with **PaddleOCR-VL**
- **Translation** with a local model served through **llama.cpp**
- **Text removal** by painting over detected text boxes
- **Redraw** by wrapping and centering English text back into the boxes

Important detail: the current translation step is **text-only**. The translator receives OCR-extracted Japanese text, not the original page image.

## Models used

### OCR

The OCR stage uses:

- `PaddlePaddle/PaddleOCR-VL-1.5`

This is configured in `ocr.py` through `PaddleOCRVL(...)` with an `mlx-vlm-server` backend.

### Translation

The translation stage talks to an OpenAI-compatible local server provided by `llama.cpp`.

The helper scripts expect:

- a GGUF model at `LLAMA_MODEL_PATH`
- a multimodal projector at `LLAMA_MMPROJ_PATH`
- server alias `llama`

The project was developed using a local **Gemma 4** GGUF served by `llama-server`.

Even though the helper script starts a multimodal model, the checked-in translation pipeline currently sends **text only** to the model.

## Requirements

### Recommended / tested environment

- **macOS / Apple Silicon**
- **Python 3.13**
- **uv** for environment management
- **llama.cpp** with `llama-server`

### Linux note

`setup.sh` contains a Linux dependency branch, but the checked-in OCR backend is `mlx-vlm-server` and `start_paddlepaddleocr.sh` starts `mlx_vlm.server`. So the provided helper scripts are best suited to **macOS / Apple Silicon**.

## Repository layout

### Input and data directories

- `images/` - sample input images for normal runs
- `data/open-mantra/` - OpenMantra benchmark dataset
- `images_google_translated/` - comparison outputs
- `images_chatgpt_translated/` - comparison outputs

### Output directories

- `output/` - default output for normal runs
- `benchmark_runs/open-mantra/` - saved benchmark runs
- `report/` - report sources and generated report files

### Main code files

- `main.py` - normal manga translation pipeline
- `ocr.py` - OCR setup and parsing
- `translate.py` - translation logic
- `edit_image.py` - text removal and redraw
- `benchmark_openmantra.py` - benchmark runner for OpenMantra
- `analyze_openmantra_benchmark.py` - benchmark analysis

## Setup

### 1. Install system prerequisites

You need:

- Python **3.13**
- `uv`
- `llama.cpp` with `llama-server` available on your `PATH`

Example on macOS with Homebrew:

```bash
brew install uv
brew install llama.cpp
```

### 2. Create the virtual environment and install Python dependencies

From the project root:

```bash
uv venv
source .venv/bin/activate
uv sync
```

This installs the dependencies declared in `pyproject.toml`.

### 3. Install OCR-specific dependencies

Run:

```bash
bash setup.sh
```

What this installs:

- `paddlepaddle` for your platform
- `paddleocr[doc-parser]`
- `mlx-vlm` on macOS

### 4. Configure environment variables

The runtime uses these variables:

- `LLAMA_MODEL_PATH` - absolute path to the GGUF model
- `LLAMA_MMPROJ_PATH` - absolute path to the multimodal projector GGUF
- `LLAMA_MODEL_NAME` - model name/alias used by the client, default: `llama`
- `LLAMA_BASE_URL` - default: `http://localhost:8080/v1`
- `OCR_VL_SERVER_URL` - default: `http://localhost:8111/`

A sample config is provided in [`.env.example`](.env.example).

Notes:

- `main.py` loads environment variables via `python-dotenv`
- `LLAMA_MODEL_NAME` must match the alias exposed by `llama-server`
- if you use `start_llamacpp.sh`, the alias is `llama`

### 5. Prepare the benchmark dataset (optional)

You only need this if you want to run the OpenMantra benchmark.

The benchmark expects the dataset at:

```text
data/open-mantra/
```

Expected structure:

```text
data/open-mantra/
├── annotation.json
├── README.md
├── LICENSE.md
└── images/
    ├── balloon_dream/
    ├── boureisougi/
    ├── rasetugari/
    ├── tencho_isoro/
    └── tojime_no_siora/
```

This repository already contains that directory. If it is missing in your copy, clone the OpenMantra dataset into `data/open-mantra/`.

## Start the required local services

You need **two local services** running before executing the pipeline.

### OCR service

```bash
source .venv/bin/activate
bash start_paddlepaddleocr.sh
```

This starts the OCR backend on port `8111` by default.

### Translation service

In another terminal:

```bash
source .venv/bin/activate
bash start_llamacpp.sh
```

This starts `llama-server` and exposes the model alias `llama`.

The checked-in script uses:

- `--alias llama`
- `--ctx-size 65536`
- `--parallel 32`
- `--no-cache-prompt`

## How to run the project

### Run on a single image

```bash
source .venv/bin/activate
python main.py images/attack_on_titan.png
```

### Run on multiple images

```bash
python main.py images/attack_on_titan.png images/demon_slayer.png
```

### Run on a directory of images

```bash
python main.py images/
```

### Use a custom output directory

```bash
python main.py images/ -o my_output
```

### Enable verbose logging

```bash
python main.py images/ --verbose
```

## Normal run outputs

For an input like `images/attack_on_titan.png`, the default output layout is:

```text
output/
├── attack_on_titan.png
├── pipeline.log
├── problem_images.log
└── attack_on_titan/
    ├── attack_on_titan_ocr.json
    ├── attack_on_titan_ocr.md
    ├── attack_on_titan_translation.md
    ├── attack_on_titan_ocr.png
    └── attack_on_titan_text_removed.png
```

What each artifact means:

- `*_ocr.json` - raw OCR output
- `*_ocr.md` - OCR text in markdown form
- `*_translation.md` - OCR text paired with translations
- `*_ocr.png` - original image with OCR bounding boxes drawn
- `*_text_removed.png` - image after text removal
- `output/<name>.png` - final translated image

## Benchmarking on OpenMantra

Run the full benchmark:

```bash
python benchmark_openmantra.py
```

Useful examples:

```bash
python benchmark_openmantra.py --limit-pages 10
python benchmark_openmantra.py --book balloon_dream
python benchmark_openmantra.py --book balloon_dream --book boureisougi
python benchmark_openmantra.py --verbose
```

Defaults:

- dataset dir: `data/open-mantra`
- output root: `benchmark_runs/open-mantra`

Each run creates a timestamped directory such as:

```text
benchmark_runs/open-mantra/20260518_115900/
```

Each run contains:

- `benchmark_results.json`
- `stage_timings.json`
- `translation_issues.json`
- `pipeline.log`
- `problem_images.log`
- per-page OCR / translation / render artifacts under `pages/...`

## Analyze a saved benchmark run

```bash
python analyze_openmantra_benchmark.py benchmark_runs/open-mantra/<run_name>
```

This writes:

- `analysis.json`
- `analysis.md`

## What the benchmark measures

The benchmark code evaluates:

- **text-region detection**: precision / recall / F1
- **OCR quality**: character error rate and exact match rate
- **translation quality**: BLEU and chrF
- **coverage**: fraction of ground-truth regions that got a non-empty translation
- **timings**: OCR, translation, artifact generation, and total runtime

## Troubleshooting

### `llama-server` not found

Install `llama.cpp` and make sure `llama-server` is on your `PATH`.

### OCR service not reachable

Make sure `start_paddlepaddleocr.sh` is running and that `OCR_VL_SERVER_URL` matches the server address.

### Translation service not reachable

Make sure `start_llamacpp.sh` is running and that `LLAMA_BASE_URL` matches the server address.

### Wrong model name

If the server alias and client model name do not match, translation requests will fail.

- server alias in `start_llamacpp.sh`: `llama`
- client default in `translate.py`: `LLAMA_MODEL_NAME=llama`

### Missing fonts

The redraw step tries several common system fonts and falls back to PIL's default font. If the final text looks bad on Linux, install a standard font such as DejaVu Sans.

## Minimal run checklist

To run the project successfully, you need:

1. Python 3.13 + `uv`
2. dependencies installed with `uv sync`
3. OCR dependencies installed with `bash setup.sh`
4. `llama.cpp` installed
5. environment variables configured
6. OCR service running
7. `llama-server` running
8. input images in `images/` or another directory
9. optionally, `data/open-mantra/` for benchmarking

Once the services are up, the simplest command is:

```bash
python main.py images/
```
