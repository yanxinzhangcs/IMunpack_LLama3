#!/bin/bash
export PYTHONPATH=/workspace/llama-models:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-3B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun --standalone --nproc_per_node=1 models/scripts/example_text_completion.py $CHECKPOINT_DIR
