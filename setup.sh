#!/usr/bin/env bash
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    uv pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    uv pip install "mlx-vlm>=0.3.11"
else # Linux
    uv pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
    uv pip install "numpy<2.4"
fi
uv pip install "paddleocr[doc-parser]"
