# browser-dl-bench

A small speed test: the same neural-network models run in PyTorch and in a web browser, on the same machine, so the times can be compared.

## What it tests
Models with random weights (two by default; `--models` also takes the classifier stand-ins `resnet18_80` and `resnet50_80`: ResNet-18 and ResNet-50 on 80x80 patches with a 2-class head, as in the group's cell classifiers). Only speed is measured, not accuracy.
- **ResNet-18**, 224x224 input, as a well-known baseline.
- **IFCRN**, 256x256 input: a small U-Net for nucleus detection (Edgren Schüllerqvist, 2025, section 6.1), 102,049 parameters. The thesis doesn't state the padding or the output activation, so `models.py` assumes "same" padding and none.

Each model runs with batch sizes 1, 8 and 16 on:
- **PyTorch**: CPU, plus CUDA, XPU or MPS if the machine has one.
- **Browser**: WebGPU (GPU) and WASM (CPU), through ONNX Runtime Web.

Method: 5 warm-up runs, then the median of 30, in fp32. Each timing includes copying the output back to the CPU, so the GPU has finished.

## Run it
Requirements: Python 3.9+, Node.js with npm, and Chrome, Chromium or Edge with WebGPU.
```bash
git clone https://github.com/Rishit-Reddy/browser-dl-bench.git && cd browser-dl-bench
python3 run.py          # on Windows: python run.py
```
`run.py` creates `.venv`, installs PyTorch and the Node package, exports the models to ONNX, times PyTorch, starts a local server, opens the browser with a clean temporary profile, waits for the browser results, and writes `results/benchmark.png`, `results/summary.md` and the raw JSON. Keep the browser window in the foreground until it says DONE; it closes itself afterwards. A first run takes about 5 minutes, mostly installs.

Everything is a flag, so you only run what you need (`python run.py --help` lists them all):

| Flag | What it does | Default |
|---|---|---|
| `--models` | `resnet18`, `ifcrn` | both |
| `--batches` | batch sizes, e.g. `1,8,16` | `1,8,16` |
| `--engines` | `pytorch-cpu`, `pytorch-gpu` (CUDA, XPU or MPS, whichever exists), `wasm`, `webgpu` | all four |
| `--runs`, `--warmup` | timed runs and warm-up runs per setting | 30, 5 |
| `--merge` | keep earlier rows in `results/` for settings you don't re-run now | off (results are replaced) |
| `--force-export` | re-export the ONNX models even if `models/` has them | off |
| `--plot-only` | redraw the charts from `results/` without running anything | |
| `--no-install` | skip creating `.venv` and installing packages | |
| `--no-browser` | print the URL and open it yourself (keep the tab in the foreground) | |
| `--browser-path`, `--browser-flags` | which browser, and its flags (on Linux the default is `--enable-unsafe-webgpu --enable-features=Vulkan`) | auto |
| `--port` | local server port | 8765 |
| `--torch-index-url` | PyTorch wheel index, e.g. `https://download.pytorch.org/whl/cpu` for a small CPU-only build, or a CUDA/ROCm index | PyPI |

Examples:
```bash
python run.py --engines webgpu                      # only WebGPU, both models
python run.py --engines webgpu --models ifcrn --batches 16 --runs 50
python run.py --engines webgpu --merge              # re-run WebGPU, keep the earlier WASM and PyTorch rows
python run.py --engines pytorch-cpu,pytorch-gpu     # PyTorch only; no browser is opened
```
The old environment variables (`BROWSER=none`, `BROWSER_PATH`, `BROWSER_FLAGS`, `PORT`, `TORCH_INDEX_URL`) still work as defaults for the matching flags.

## Memory sweep
`python run.py memory` (all settings below are flags: `--sweeps calibration,batch,size,long`, `--mem-batches`, `--mem-sizes`, `--mem-iters`, `--long-iters`) measures how much GPU memory IFCRN uses on browser WebGPU as the batch size and the tile size grow, and whether memory keeps growing over a long run. It starts a fresh browser for every setting (about 10 minutes in total) and writes `results/memory.png` and `results/memory.md`. macOS and Linux only, and the browser must be one that `run.py` launches.

WebGPU doesn't tell the page how much GPU memory it uses, so the server reads it from outside, from the browser's processes. This is the weak part:
- On macOS it uses `footprint`, which includes GPU memory. The first panel of the chart is a calibration (allocate 1 GB of buffers, see what the number does). On my M2 it went up by about 1.1 GB, so it works there, and it was slow to come back down after the buffers were freed.
- Plain RSS did not move at all in that calibration on the M2, so don't use it for GPU memory on Apple silicon.
- On Linux the script falls back to the GPU process's RSS, which may not include video memory. Look at the calibration panel first. If it doesn't track the allocation, the other panels are not about GPU memory.
- Renderer (page) memory is read as RSS and is noisy. Ignore it for small differences.
- The numbers are totals for the GPU process, including about 0.5 GB that the browser uses before the model loads. "Growth" is the peak minus the reading just before the model loads.

## Platform notes
Only tested on macOS (Apple M2, Chrome). The rest is expected to work but is unverified.
- **WebGPU**: needs a hardware-accelerated adapter. If the browser falls back to a software adapter, the "browser GPU" numbers are not GPU numbers; the chart title says so. On Linux this usually needs a working Vulkan driver and the flags above. Check `chrome://gpu`.
- **PyTorch GPU**: NVIDIA works with the default Linux build. On Windows the default wheel is CPU-only, so pass `--torch-index-url` with a CUDA index. AMD needs a ROCm build. Intel GPUs need an XPU build.
- **Linux download size**: the default PyTorch build includes CUDA and is a few GB.
- **Snap Chromium** may refuse the temporary profile. Use `--no-browser` instead.

## Reading the results
- Browser and PyTorch use different software (ONNX Runtime Web vs PyTorch), so the comparison is partly a runtime comparison.
- Weights are random, so accuracy is not measured.
- Only the model is timed, not tile fetching, decoding or post-processing.
- Numbers vary between runs, especially on laptops without fans. Run it a few times on a plugged-in machine.

A reference run on a MacBook Air M2 is in `reference/macbook-air-m2/`.

## Files
- `run.py`: the one command
- `models.py`: the two models
- `bench_torch.py`: exports the ONNX files and times PyTorch
- `export_dynamic.py`, `web/memory.html`, `plot_memory.py`: the memory sweep
- `web/ort.html`: the browser benchmark; `server.js`: a small local server that receives its results
- `plot.py`: builds the chart and tables from `results/`
