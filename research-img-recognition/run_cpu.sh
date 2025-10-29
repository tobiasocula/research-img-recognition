#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root
cd "$(dirname "$0")"

# Disable GPU usage - force CPU only
export CUDA_VISIBLE_DEVICES=""

# Optional: clear other GPU-related variables if set
unset TF_GPU_ALLOCATOR
unset XLA_FLAGS
unset TF_CUDNN_WORKSPACE_LIMIT_IN_MB
unset TF_CUDNN_USE_AUTOTUNE
unset TF_DETERMINISTIC_OPS
unset TF_XLA_FLAGS

# Activate venv and run training
source venv/bin/activate
python make_nn.py --text_dir input_images_train_cropped --bars_dir bars_targets_train_cropped
