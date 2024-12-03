#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.2-11B-Vision-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun models/scripts/example_text_completion.py $CHECKPOINT_DIR
