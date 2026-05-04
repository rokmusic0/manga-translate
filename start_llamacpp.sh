#!/usr/bin/env bash
llama-server --model $LLAMA_MODEL_PATH --mmproj $LLAMA_MMPROJ_PATH --alias llama --ctx-size 8192
