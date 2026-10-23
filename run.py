#!/usr/bin/env python3
"""PyTorch vs browser inference benchmark, and a browser GPU-memory sweep. Windows, Linux or macOS.
Needs Python 3.9+, Node/npm, and Chrome/Edge/Chromium with WebGPU for the browser parts.

  python run.py                                    # full benchmark, everything
  python run.py --engines webgpu --models ifcrn    # only WebGPU on IFCRN
  python run.py --engines webgpu --merge           # re-run WebGPU, keep the earlier WASM / PyTorch rows
  python run.py memory --sweeps batch --mem-batches 4,8,16
  python run.py --plot-only                        # redraw charts from results/ without running anything
The environment variables BROWSER=none, BROWSER_PATH, BROWSER_FLAGS, PORT and TORCH_INDEX_URL still work as defaults for the matching flags."""
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent; os.chdir(ROOT)
WIN = os.name == "nt"
MODELS = {"resnet18": 224, "ifcrn": 256}  # name -> input size; keep in sync with models.py
ENGINES = ("pytorch-cpu", "pytorch-gpu", "wasm", "webgpu")
SWEEPS = ("calibration", "batch", "size", "long")
def ints(s): return [int(x) for x in s.split(",") if x]
def words(s): return [x.strip() for x in s.split(",") if x.strip()]
ints.__name__ = "comma-separated integers"; words.__name__ = "comma-separated names"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", nargs="?", default="bench", choices=("bench", "memory"), help="bench (default): speed comparison. memory: browser GPU memory sweep (macOS/Linux)")
g = ap.add_argument_group("what to test (bench)")
g.add_argument("--models", type=words, default=list(MODELS), help=f"comma list of {','.join(MODELS)} (default: all)")
g.add_argument("--batches", type=ints, default=[1, 8, 16], help="comma list of batch sizes (default 1,8,16)")
g.add_argument("--engines", type=words, default=list(ENGINES), help="comma list of pytorch-cpu, pytorch-gpu (CUDA/XPU/MPS, whichever exists), wasm, webgpu (default: all)")
g.add_argument("--runs", type=int, default=30, help="timed runs per setting (default 30)")
g.add_argument("--warmup", type=int, default=5, help="warm-up runs per setting (default 5)")
g.add_argument("--merge", action="store_true", help="keep earlier rows in results/ for settings not re-run now (otherwise the results files are replaced)")
g.add_argument("--force-export", action="store_true", help="re-export the ONNX models even if models/ already has them")
m = ap.add_argument_group("what to test (memory)")
m.add_argument("--sweeps", type=words, default=list(SWEEPS), help=f"comma list of {','.join(SWEEPS)} (default: all)")
m.add_argument("--mem-batches", type=ints, default=[1, 2, 4, 8, 16, 32, 64], help="batch sizes for the batch sweep (256x256 tiles)")
m.add_argument("--mem-sizes", type=ints, default=[128, 256, 512, 1024, 2048, 4096], help="tile sides in px for the size sweep (batch 1)")
m.add_argument("--mem-iters", type=int, default=10, help="timed runs per memory setting (default 10)")
m.add_argument("--long-iters", type=int, default=600, help="batches of 16 in the long run (default 600)")
o = ap.add_argument_group("environment")
o.add_argument("--no-browser", action="store_true", default=os.environ.get("BROWSER") == "none", help="don't launch a browser: print the URL and open it yourself (bench only)")
o.add_argument("--browser-path", default=os.environ.get("BROWSER_PATH"), help="browser executable (default: find Chrome, Edge or Chromium)")
o.add_argument("--browser-flags", default=os.environ.get("BROWSER_FLAGS", ""), help='extra browser flags, e.g. "--enable-unsafe-webgpu" (Linux default: --enable-unsafe-webgpu --enable-features=Vulkan)')
o.add_argument("--port", default=os.environ.get("PORT", "8765"), help="local server port (default 8765)")
o.add_argument("--torch-index-url", default=os.environ.get("TORCH_INDEX_URL"), help="PyTorch wheel index, e.g. https://download.pytorch.org/whl/cpu or a CUDA index")
o.add_argument("--no-install", action="store_true", help="skip creating .venv and installing packages (assumes they are already set up)")
o.add_argument("--plot-only", action="store_true", help="just redraw the charts from results/")
a = ap.parse_args()
for x, allowed, flag in ((a.models, MODELS, "--models"), (a.engines, ENGINES, "--engines"), (a.sweeps, SWEEPS, "--sweeps")):
    bad = [i for i in x if i not in allowed]
    if bad: ap.error(f"{flag}: unknown {bad}; choose from {list(allowed)}")

venv_py = str(ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python"))
py = venv_py if os.path.exists(venv_py) else sys.executable
def run(cmd, **kw): subprocess.run(cmd, check=True, **kw)
PLOT = "plot.py" if a.mode == "bench" else "plot_memory.py"
os.makedirs("results", exist_ok=True)
if a.plot_only: run([py, PLOT]); sys.exit()

def find_browser():
    if a.browser_path: return a.browser_path
    mac = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    win = [os.path.join(os.environ.get(v, ""), p) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA") for p in (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe")]
    for p in (mac if sys.platform == "darwin" else win if WIN else []):
        if os.path.exists(p): return p
    for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "chrome"):
        if shutil.which(n): return shutil.which(n)

if not a.no_install:
    if not os.path.exists(venv_py): run([sys.executable, "-m", "venv", ".venv"])
    py = venv_py
    pip = [py, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
    if a.torch_index_url: pip += ["--index-url", a.torch_index_url, "--extra-index-url", "https://pypi.org/simple"]
    run(pip)
    if not os.path.isdir("node_modules"): run(["npm", "install", "--silent"], shell=WIN)  # shell=True so Windows finds npm.cmd

browser_engines = [e for e in a.engines if e in ("wasm", "webgpu")]; torch_engines = [e for e in a.engines if e.startswith("pytorch")]
if a.mode == "memory" and (WIN or a.no_browser): sys.exit("the memory sweep reads the browser's process memory with ps, so it needs macOS/Linux and a browser that run.py launches")
old = {}
if a.mode == "bench":
    json.dump({"models": a.models, "sizes": MODELS, "batches": a.batches, "engines": a.engines, "runs": a.runs, "warmup": a.warmup, "force_export": a.force_export}, open("results/bench_cfg.json", "w"))
    for f, used in (("results/torch.json", torch_engines), ("results/browser.json", browser_engines)):
        if used and os.path.exists(f):
            if a.merge: old[f] = json.load(open(f))
            os.remove(f)
    missing = [(n, b) for n in a.models for b in a.batches if not os.path.exists(f"models/{n}_b{b}.onnx")]
    if torch_engines or (browser_engines and (missing or a.force_export)): run([py, "bench_torch.py"])
else:
    cfgs = [{"kind": "calibration"}] * ("calibration" in a.sweeps)
    cfgs += [{"kind": "batch", "b": b, "s": 256, "iters": a.mem_iters} for b in a.mem_batches] * ("batch" in a.sweeps)
    cfgs += [{"kind": "size", "b": 1, "s": s, "iters": a.mem_iters} for s in a.mem_sizes] * ("size" in a.sweeps)
    cfgs += [{"kind": "long", "b": 16, "s": 256, "iters": a.long_iters}] * ("long" in a.sweeps)
    json.dump(cfgs, open("results/memory_cfgs.json", "w")); run([py, "export_dynamic.py"])

server = subprocess.Popen(["node", "server.js"], env={**os.environ, "PORT": a.port})

def browse(url, out, wait):
    """Opens the page in a fresh browser (temporary profile) and waits for `out` to appear."""
    exe = None if a.no_browser else find_browser()
    browser = None
    if exe:
        flags = a.browser_flags.split()
        if sys.platform.startswith("linux") and not flags: flags = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]  # WebGPU is still behind flags on many Linux setups
        browser = subprocess.Popen([exe, "--user-data-dir=" + tempfile.mkdtemp(), "--no-first-run", "--no-default-browser-check", *flags, url])  # fresh profile: no extensions, flags honoured
        open("results/.browser_pid", "w").write(str(browser.pid))  # lets the server sample the browser's memory
    else: print(f"Open {url} in a WebGPU-capable browser and keep the tab in the foreground.")
    try:
        t0 = time.time()
        while not (os.path.exists(out) and os.path.getsize(out) > 0):
            if time.time() - t0 > wait: sys.exit("timed out waiting for the browser results")
            if server.poll() is not None: sys.exit("server stopped unexpectedly")
            time.sleep(1)
        time.sleep(1)
    finally:
        if browser: browser.terminate(); time.sleep(2)

try:
    time.sleep(1)
    if server.poll() is not None: sys.exit(f"server failed to start (port {a.port} in use? try --port 8766)")
    if a.mode == "bench":
        if browser_engines:
            print("Waiting for the browser run. Keep its window in the foreground...")
            browse(f"http://localhost:{a.port}/web/ort.html", "results/browser.json", 3600)
    else:
        parts = []
        for i, c in enumerate(cfgs):
            f = f"results/memory_{i}.json"
            if os.path.exists(f): os.remove(f)
            print(f"setting {i + 1}/{len(cfgs)}: {c}", flush=True)
            browse(f"http://localhost:{a.port}/web/memory.html?cfg={i}", f, 600); parts.append(json.load(open(f)))
        json.dump({"results": parts}, open("results/memory.json", "w"))
finally:
    server.terminate()

for f, prev in old.items():  # --merge: put back earlier rows for settings that were not re-run
    if not os.path.exists(f): json.dump(prev, open(f, "w"), indent=1); continue
    new = json.load(open(f)); key = lambda r: (r["model"], r["engine"], r["batch"])
    rows = {key(r): r for r in prev["results"]}; rows.update({key(r): r for r in new["results"]})
    new["results"] = list(rows.values()); json.dump(new, open(f, "w"), indent=1)
run([py, PLOT])
