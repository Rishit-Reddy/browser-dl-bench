#!/usr/bin/env python3
"""One command on Windows, Linux or macOS:  python run.py   (needs Python 3.9+, Node/npm, and Chrome/Edge/Chromium with WebGPU).
Installs into .venv and node_modules, exports the models, times PyTorch, serves web/ort.html, opens the browser, waits for its
results, then writes results/benchmark.png and results/summary.md.
Env: BROWSER=none (just print the URL) | BROWSER_PATH=<browser exe> | BROWSER_FLAGS="--flag ..." | PORT | TORCH_INDEX_URL (e.g. a CUDA wheel index)"""
import os, pathlib, platform, shutil, subprocess, sys, tempfile, time

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
if os.path.exists("results/browser.json"): os.remove("results/browser.json")
run([py, "bench_torch.py"])

server = subprocess.Popen(["node", "server.js"], env={**os.environ, "PORT": PORT}); browser = None
try:
    time.sleep(1)
    if server.poll() is not None: sys.exit(f"server failed to start (port {PORT} in use? try PORT=8766 python run.py)")
    url = f"http://localhost:{PORT}/web/ort.html"
    exe = None if os.environ.get("BROWSER") == "none" else find_browser()
    if exe:
        flags = os.environ.get("BROWSER_FLAGS", "").split()
        if sys.platform.startswith("linux") and not flags: flags = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"]  # WebGPU is still behind flags on many Linux setups
        browser = subprocess.Popen([exe, "--user-data-dir=" + tempfile.mkdtemp(), "--no-first-run", "--no-default-browser-check", *flags, url])  # fresh profile: no extensions, flags honoured
    else: print(f"Open {url} in a WebGPU-capable browser and keep the tab in the foreground.")
    print("Waiting for the browser run (about 2 minutes). Keep its window in the foreground...")
    t0 = time.time()
    while not (os.path.exists("results/browser.json") and os.path.getsize("results/browser.json") > 0):
        if time.time() - t0 > 900: sys.exit("timed out waiting for the browser results")
        if server.poll() is not None: sys.exit("server stopped unexpectedly")
        time.sleep(2)
    time.sleep(1)
finally:
    if browser: browser.terminate()
    server.terminate()
run([py, "plot.py"])
