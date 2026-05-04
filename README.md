# ComicTrans

Translate Japanese manga into English using deep learning.

## Setup on macos

### PaddlePaddle setup

Read https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html#31-starting-the-vlm-inference-service, and prepare the environment.

### Llama.cpp setup

Read https://github.com/ggml-org/llama.cpp, and install llama.cpp on your machine. Download a multimodal model from huggingface, and convert it to gguf. This model will be used to translate the text.

## Run services

See [start_paddlepaddleocr.sh](start_paddlepaddleocr.sh) for starting the PaddlePaddle OCR service, and [start_llamacpp.sh](start_llamacpp.sh) for starting the llama.cpp service.
