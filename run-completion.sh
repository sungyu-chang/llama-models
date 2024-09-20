#!/bin/bash

CHECKPOINT_DIR=~/.llama/checkpoints/Meta-Llama3.1-8B

PYTHONPATH=$(git rev-parse --show-toplevel) torchrun models/scripts/example_text_completion.py $CHECKPOINT_DIR
