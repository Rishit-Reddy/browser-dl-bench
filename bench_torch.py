"""PyTorch side. Reads results/bench_cfg.json (written by run.py; defaults if absent). Exports the ONNX models the browser needs
(only missing files, unless force_export) and times PyTorch on the selected devices. Writes results/torch.json."""
import json, os, statistics as st, time, torch
from models import MODELS
C = json.load(open("results/bench_cfg.json")) if os.path.exists("results/bench_cfg.json") else {}
models, batches = C.get("models", list(MODELS)), C.get("batches", [1, 8, 16])
engines = C.get("engines", ["pytorch-cpu", "pytorch-gpu", "wasm", "webgpu"]); N, WARM = C.get("runs", 30), C.get("warmup", 5)
os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True); torch.manual_seed(0)

def bench(name, m, s, dev, b):
    m = m.to(dev); x = torch.randn(b, 3, s, s, device=dev)
    with torch.no_grad():
        t = time.perf_counter(); m(x).cpu(); first = (time.perf_counter() - t) * 1000
        for _ in range(WARM): m(x).cpu()
        ts = []
        for _ in range(N):
            t = time.perf_counter(); m(x).cpu(); ts.append((time.perf_counter() - t) * 1000)  # .cpu() waits for the GPU and copies the output back
    ts.sort(); return {"model": name, "engine": f"pytorch-{dev}", "batch": b, "median_ms": st.median(ts), "p90_ms": ts[min(N - 1, int(.9 * N))], "first_ms": first, "runs": N}

devs = []
if "pytorch-cpu" in engines: devs.append("cpu")
if "pytorch-gpu" in engines:
    if torch.cuda.is_available(): devs.append("cuda")                                    # NVIDIA, or AMD with a ROCm build of PyTorch
    if hasattr(torch, "xpu") and torch.xpu.is_available(): devs.append("xpu")             # Intel GPUs
    if torch.backends.mps.is_available(): devs.append("mps")                              # Apple silicon
    if not any(d != "cpu" for d in devs): print("no PyTorch GPU found (CUDA / XPU / MPS); skipping pytorch-gpu")
names = {"cuda": torch.cuda.get_device_name(0) if "cuda" in devs else None, "xpu": torch.xpu.get_device_name(0) if "xpu" in devs else None}
out = []
for name in models:
    mk, s = MODELS[name]; m = mk().eval()
    if any(e in engines for e in ("wasm", "webgpu")):
        for b in batches:
            f = f"models/{name}_b{b}.onnx"
            if os.path.exists(f) and not C.get("force_export"): continue
            kw = dict(opset_version=17, input_names=["x"], output_names=["y"])
            try: torch.onnx.export(m, torch.randn(b, 3, s, s), f, dynamo=False, **kw)
            except TypeError: torch.onnx.export(m, torch.randn(b, 3, s, s), f, **kw)  # newer torch without the dynamo flag
    for d in devs:
        for b in batches:
            r = bench(name, m, s, d, b); out.append(r); print(r, flush=True)
if devs: json.dump({"torch": torch.__version__, "devices": devs, "device_names": names, "results": out}, open("results/torch.json", "w"), indent=1)
