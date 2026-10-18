# browser-dl-bench

I want to know if deep-learning inference can run in a web browser fast enough to move part of the CytoBrowser pipeline off the server. This repo is a small speed test for that. It runs the same models in PyTorch and in the browser, on the same laptop, and compares the times.

I'm sending it so you can run it on your laptop and send me the numbers. It takes about 5 minutes.

## What it tests
Two models, both with random weights. Only speed is measured, not accuracy.
- **ResNet-18**, 224x224, as a well-known baseline.
- **IFCRN**, 256x256: Olle's nucleus detector, built from the description in his thesis (section 6.1). It has 102,049 parameters. The thesis doesn't say how the convolutions are padded or whether the output has an activation, so `models.py` assumes "same" padding and none.

Each model runs with batch sizes 1, 8 and 16 on:
- **PyTorch**: CPU, plus CUDA / XPU / MPS if the machine has one.
- **Browser**: WebGPU (the GPU) and WASM (the CPU), through ONNX Runtime Web.

Method: 5 warm-up runs, then the median of 30. Every timing includes copying the output back to the CPU, so the GPU has finished. Everything is fp32.

## Run it (Linux)
You need Python 3.9+, Node.js with npm, and Chrome, Chromium or Edge.
```bash
sudo apt install python3-venv nodejs npm     # Debian/Ubuntu; use your own package manager otherwise
git clone https://github.com/Rishit-Reddy/browser-dl-bench.git && cd browser-dl-bench
python3 run.py
```
`run.py` does everything: it creates `.venv`, installs PyTorch and the Node package, exports the models to ONNX, times PyTorch, starts a local server, opens the browser with a clean temporary profile, waits for the browser results, and writes the charts. Keep the browser window in the foreground until it says DONE. It closes itself after.

When it finishes, send me the whole `results/` folder (`summary.md`, `benchmark.png`, `torch.json`, `browser.json`) and your laptop model and GPU.

## Things that can go wrong on Linux
I have only run this on my MacBook Air M2. The Linux notes below are what I expect to matter, but I haven't tested them. Please tell me what breaks.

- **WebGPU.** Chrome on Linux often needs flags and a working Vulkan driver. `run.py` passes `--enable-unsafe-webgpu --enable-features=Vulkan` by default. Open `chrome://gpu` and check that WebGPU says hardware accelerated. If the browser falls back to a software adapter, the "browser GPU" numbers are not GPU numbers. The chart title says so when that happens. You can change the flags with `BROWSER_FLAGS="..."`.
- **Which browser.** `python3 run.py` looks for `google-chrome`, `chromium`, `microsoft-edge` and similar. Use `BROWSER_PATH=/path/to/browser` to pick one. A snap-installed Chromium may refuse the temporary profile. If it does, run `BROWSER=none python3 run.py`, open the printed URL yourself, and keep the tab in the foreground.
- **PyTorch download size.** On Linux the default PyTorch build includes CUDA and is a few GB. For CPU only, use the smaller build: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu python3 run.py`.
- **GPU support in PyTorch.** NVIDIA is picked up automatically with the default build. For AMD you need the ROCm build from pytorch.org (set `TORCH_INDEX_URL`). An Intel or integrated GPU will usually show PyTorch on CPU only, so the browser's GPU row has nothing to compare against.
- **Port in use.** Use `PORT=8766 python3 run.py`.

## For a fair run
Plug the laptop in, close other heavy programs, and run it two or three times. My browser numbers moved by about 1.5x between runs on a fanless laptop, so one run is not enough.

## What the numbers mean (and don't)
- It shows how fast each setup runs the model. It does not show detection accuracy, because the weights are random.
- PyTorch and ONNX Runtime Web are different software, so the comparison is partly a runtime comparison and not only PyTorch vs browser.
- It is only the model. Fetching tiles, decoding them and finding peaks in the output are not included.

My own result is in `reference/macbook-air-m2/` (`summary.md` and `benchmark.png`), so you can compare. On my laptop the browser's WebGPU ran IFCRN at about 12-13 ms per tile, and PyTorch on the Apple GPU at about 7 ms.

## Files
- `run.py`: the one command.
- `models.py`: the two models.
- `bench_torch.py`: exports the ONNX files and times PyTorch.
- `web/ort.html`: the browser benchmark. `server.js`: a tiny local server that receives its results.
- `plot.py`: makes the chart and tables from `results/`.
