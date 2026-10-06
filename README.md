# BuildSmallLLM

Notebooks (in `src/`) for building, training, and deploying a small LLM from scratch.

## Requirements

- [uv](https://docs.astral.sh/uv/) (and optionally [mise](https://mise.jdx.dev/))
- Python 3.12

PyTorch is **not** installed by default. You pick exactly one build — CPU, CUDA, or ROCm — with a uv "extra".

## PyTorch extras

| Extra   | Build                | Use when                                                  | Platforms             |
|---------|----------------------|-----------------------------------------------------------|-----------------------|
| `cpu`   | CPU only             | No GPU, CI, or macOS (macOS gets the PyPI wheel with MPS) | Linux, Windows, macOS |
| `cu126` | CUDA 12.6            | Older NVIDIA drivers (`nvidia-smi` reports CUDA 12.x)     | Linux, Windows        |
| `cu130` | CUDA 13.0            | NVIDIA driver supports CUDA 13.0 or 13.1                  | Linux, Windows        |
| `cu132` | CUDA 13.2            | Recent NVIDIA drivers (CUDA 13.2+)                        | Linux, Windows        |
| `rocm`  | ROCm 7.14            | AMD GPUs / APUs                                           | Linux                 |

The "CUDA Version" in the top-right of `nvidia-smi` is the newest CUDA build your driver can run. Pick the highest extra that doesn't exceed it.

### Install with uv

```bash
uv sync --extra cpu      # CPU only
uv sync --extra cu126    # NVIDIA, CUDA 12.6
uv sync --extra cu130    # NVIDIA, CUDA 13.0
uv sync --extra cu132    # NVIDIA, CUDA 13.2
uv sync --extra rocm     # AMD, ROCm 7.14
```

The extras conflict with each other, so passing two (e.g. `--extra cpu --extra rocm`) is an error. To switch backends, re-run `uv sync` with the other extra.

> **Note:** a plain `uv sync` with no `--extra` removes PyTorch from `.venv`. Always pass your extra to `uv sync`. `uv run` doesn't remove packages, so it's safe to use after the initial sync.

### Install with mise

`mise.toml` pins Python 3.12 and uv, creates and activates `.venv` when you `cd` into the project, and adds tasks:

```bash
mise trust          # first time only
mise install        # installs Python 3.12 + uv
mise run setup      # auto-detects NVIDIA / AMD / CPU and runs `uv sync --extra <backend>`
mise run check      # prints the torch version and the accelerator it can see
mise run kernel     # (optional) registers .venv as a Jupyter kernel
```

To skip auto-detection and pick a backend yourself:

```bash
mise run setup cpu      # or cu126 | cu130 | cu132 | rocm
```

## Running the notebooks

Open any notebook in `src/` with VS Code / Jupyter and select the `.venv` interpreter (or the `BuildSmallLLM (.venv)` kernel after `mise run kernel`).

## Notes

- **ROCm:** the `pytorch-rocm` index in `pyproject.toml` is intentionally not `explicit`. The ROCm 7.14 torch wheels depend on `rocm` / `rocm-sdk-*` runtime packages that are only published on that index. Because of this, some common dependencies (e.g. `numpy`, `jinja2`) are locked from the PyTorch mirror rather than PyPI. The files are the same.
- **ROCm:** a `rocSHMEM ... Could not open libnuma` message on import is harmless. Install your distro's `numactl` package to silence it.
