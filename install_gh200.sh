#!/bin/bash
set -e

uv pip install "torch==2.4.1" --torch-backend=cu124
uv pip install ninja
uv pip install -r requirements.txt