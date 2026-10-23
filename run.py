#!/usr/bin/env python3
"""One command on Windows, Linux or macOS:  python run.py   (needs Python 3.9+, Node/npm, and Chrome/Edge/Chromium with WebGPU).
Installs into .venv and node_modules, exports the models, times PyTorch, serves web/ort.html, opens the browser, waits for its
results, then writes results/benchmark.png and results/summary.md.
Env: BROWSER=none (just print the URL) | BROWSER_PATH=<browser exe> | BROWSER_FLAGS="--flag ..." | PORT | TORCH_INDEX_URL (e.g. a CUDA wheel index)"""
import os, pathlib, platform, shutil, subprocess, sys, tempfile, time

MODES = {"bench": dict(prep="bench_torch.py", page="web/ort.html", out="results/browser.json", plot="plot.py", wait=900),
         "memory": dict(prep="export_dynamic.py", page="web/memory.html", out="results/memory.json", plot="plot_memory.py", wait=2400)}  # python run.py memory
MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
if MODE not in MODES: sys.exit("usage: python run.py [bench|memory]")
M = MODES[MODE]
ROOT = pathlib.Path(__file__).resolve().parent; os.chdir(ROOT)
WIN, PORT = os.name == "nt", os.environ.get("PORT", "8765")
py = str(ROOT / ".venv" / ("Scripts/python.exe" if WIN else "bin/python"))

def run(cmd, **kw): subprocess.run(cmd, check=True, **kw)

def find_browser():
    if os.environ.get("BROWSER_PATH"): return os.environ["BROWSER_PATH"]
    mac = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    win = [os.path.join(os.environ.get(v, ""), p) for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA") for p in (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe")]
    paths = mac if sys.platform == "darwin" else win if WIN else []
    for p in paths:
        if os.path.exists(p): return p
    for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "chrome"):
        if shutil.which(n): return shutil.which(n)

if not os.path.exists(py): run([sys.executable, "-m", "venv", ".venv"])
pip = [py, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
if os.environ.get("TORCH_INDEX_URL"): pip += ["--index-url", os.environ["TORCH_INDEX_URL"], "--extra-index-url", "https://pypi.org/simple"]
run(pip)
if not os.path.isdir("node_modules"): run(["npm", "install", "--silent"], shell=WIN)  # shell=True so Windows finds npm.cmd
if MODE == "memory" and (os.name == "nt" or os.environ.get("BROWSER") == "none"): sys.exit("the memory sweep reads the browser's process memory with ps, so it needs macOS/Linux and a browser that run.py launches")
os.makedirs("results", exist_ok=True)
if os.path.exists(M["out"]): os.remove(M["out"])
run([py, M["prep"]])

server = subprocess.Popen(["node", "server.js"], env={**os.environ, "PORT": PORT})

def browse(url, out, wait):
    """Opens the page in a fresh browser (temporary profile) and waits for `out` to appear. Returns once it has."""
    exe = None if os.environ.get("BROWSER") == "none" else find_browser()
    browser = None
    if exe:
        flags = os.environ.get("BROWSER_FLAGS", "").split()
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
    if server.poll() is not None: sys.exit(f"server failed to start (port {PORT} in use? try PORT=8766 python run.py)")
    base = f"http://localhost:{PORT}/{M['page']}"
    if MODE == "bench":
        print("Waiting for the browser run (about 2 minutes). Keep its window in the foreground...")
        browse(base, M["out"], M["wait"])
    else:  # memory: one fresh browser per setting, so the GPU process starts clean each time
        import json
        cfgs = json.load(open("web/memory_cfgs.json")); only = os.environ.get("ONLY")
        parts = []
        for i, c in enumerate(cfgs):
            if only and c["kind"] != only: continue
            f = f"results/memory_{i}.json"
            if os.path.exists(f): os.remove(f)
            print(f"setting {i + 1}/{len(cfgs)}: {c}", flush=True)
            browse(f"{base}?cfg={i}", f, 600); parts.append(json.load(open(f)))
        json.dump({"results": parts}, open(M["out"], "w"))
finally:
    server.terminate()
run([py, M["plot"]])
