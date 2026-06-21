source .venv/bin/activate

export UV_CACHE_DIR=$PWD/.cache/uv
export PIP_CACHE_DIR=$PWD/.cache/pip
export TMPDIR=$PWD/tmp
export TORCH_CUDA_ARCH_LIST="9.0"

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
