"""Reads results/torch.json and results/browser.json; writes results/benchmark.png and results/summary.md."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("results/torch.json"))["results"]
B = json.load(open("results/browser.json")); R += B["results"]
names = {"pytorch-cpu": "PyTorch CPU", "pytorch-cuda": "PyTorch CUDA (GPU)", "pytorch-xpu": "PyTorch XPU (GPU)", "pytorch-mps": "PyTorch MPS (GPU)",
         "browser-wasm": "Browser WASM (CPU)", "browser-webgpu": "Browser WebGPU (GPU)"}
cols = {"pytorch-cpu": "#9e9e9e", "pytorch-cuda": "#2ca02c", "pytorch-xpu": "#9467bd", "pytorch-mps": "#d62728", "browser-wasm": "#aec7e8", "browser-webgpu": "#1f77b4"}
eng = [(e, names[e], cols[e]) for e in names if any(r["engine"] == e and "error" not in r for r in R)]  # only engines that ran
models = [("resnet18", "ResNet-18 (baseline), 224x224"), ("ifcrn", "IFCRN (Olle's detector), 256x256")]
batches = sorted({r["batch"] for r in R})
get = lambda m, e, b: next((r["median_ms"] / b for r in R if r["model"] == m and r["engine"] == e and r["batch"] == b and "error" not in r), float("nan"))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); w = 0.8 / len(eng)
for a, (m, title) in zip(ax, models):
    for i, (e, n, c) in enumerate(eng):
        bars = a.bar([j + (i - (len(eng) - 1) / 2) * w for j in range(len(batches))], [get(m, e, b) for b in batches], w, label=n, color=c); a.bar_label(bars, fmt="%.1f", fontsize=7)
    a.set_xticks(range(len(batches))); a.set_xticklabels([f"batch {b}" for b in batches]); a.set_ylabel("ms per image (lower is better)"); a.set_title(title, fontsize=10)
ax[0].legend(fontsize=8)
g = B.get("gpu") or {}; T = json.load(open("results/torch.json")); dn = ", ".join(f"{k}: {v}" for k, v in T.get("device_names", {}).items() if v)
fig.suptitle(f"PyTorch vs browser, fp32, random weights, median of 30 runs, output copied back to CPU. Browser GPU: {g.get('vendor')} {g.get('arch')}" + (f"; PyTorch {dn}" if dn else "") + (" (SOFTWARE ADAPTER)" if g.get("software") else ""), fontsize=9)
fig.tight_layout(); fig.savefig("results/benchmark.png", dpi=140)

rows = ["| model | engine | " + " | ".join(f"batch {b} (ms/img)" for b in batches) + " |", "|---|---|" + "---|" * len(batches)]
for m, _ in models:
    for e, n, _ in eng: rows.append(f"| {m} | {n} | " + " | ".join(f"{get(m, e, b):.2f}" for b in batches) + " |")
open("results/summary.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows)); print("\nwrote results/benchmark.png, results/summary.md")
