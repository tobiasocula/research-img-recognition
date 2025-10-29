#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root
cd "$(dirname "$0")"

# Safer allocator + force smaller-memory conv algorithms
export TF_GPU_ALLOCATOR=cuda_malloc
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=64
export TF_CUDNN_USE_AUTOTUNE=0
export TF_DETERMINISTIC_OPS=1
export TF_XLA_FLAGS=--tf_xla_auto_jit=0

# Activate venv and run training
source venv/bin/activate
python make_nn.py --text_dir input_images_train_cropped --bars_dir bars_targets_train_cropped


