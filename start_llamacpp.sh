#!/usr/bin/env bash
# llama-server splits total ctx across parallel slots.
# To allow up to 32 concurrent requests with ~2048 ctx each, use 32 * 2048 for --ctx-size.
llama-server \
  --model "$LLAMA_MODEL_PATH" \
  --mmproj "$LLAMA_MMPROJ_PATH" \
  --temperature 0.0 \
  --alias llama \
  --ctx-size 65536 \
  --parallel 32 \
  --no-cache-prompt
