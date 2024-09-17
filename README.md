# Candle Serve

## Pre-reqs
```bash
uv tool install huggingface_hub
huggingface-cli login
```

## Verify CUDA

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
nvcc --version
```

## Usage

```bash
cargo run --features cuda -- --prompt "Write me a poem about Machine Learning."
```
