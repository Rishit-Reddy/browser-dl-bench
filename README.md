# browser-dl-bench

A small speed test: the same neural-network models run in PyTorch and in a web browser, on the same machine, so the times can be compared.

## What it tests
Two models with random weights. Only speed is measured, not accuracy.
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

Options (environment variables):

| Variable | Use |
|---|---|
| `BROWSER=none` | print the URL instead of opening a browser; open it yourself and keep the tab in the foreground |
| `BROWSER_PATH` | path to the browser executable |
| `BROWSER_FLAGS` | browser flags (on Linux the default is `--enable-unsafe-webgpu --enable-features=Vulkan`) |
| `PORT` | server port (default 8765) |
| `TORCH_INDEX_URL` | PyTorch wheel index, e.g. `https://download.pytorch.org/whl/cpu` for a small CPU-only build, or a CUDA/ROCm index |

## Platform notes
Only tested on macOS (Apple M2, Chrome). The rest is expected to work but is unverified.
- **WebGPU**: needs a hardware-accelerated adapter. If the browser falls back to a software adapter, the "browser GPU" numbers are not GPU numbers; the chart title says so. On Linux this usually needs a working Vulkan driver and the flags above. Check `chrome://gpu`.
- **PyTorch GPU**: NVIDIA works with the default Linux build. On Windows the default wheel is CPU-only, so set `TORCH_INDEX_URL` to a CUDA index. AMD needs a ROCm build. Intel GPUs need an XPU build.
- **Linux download size**: the default PyTorch build includes CUDA and is a few GB.
- **Snap Chromium** may refuse the temporary profile. Use `BROWSER=none` instead.

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
- `web/ort.html`: the browser benchmark; `server.js`: a small local server that receives its results
- `plot.py`: builds the chart and tables from `results/`
