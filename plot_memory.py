"""Reads results/memory.json; writes results/memory.png and results/memory.md.
GPU memory = GPU-process footprint on macOS (`footprint`); elsewhere GPU-process RSS, which may not include VRAM (see the calibration panel)."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("results/memory.json"))["results"]; mb = lambda kb: kb / 1024
fp = any("fp_gpu_kb" in (r.get("baseline") or r.get("steps", [{}])[0]) for r in R if "error" not in r)
G = "fp_gpu_kb" if fp else "gpu_kb"; gname = "GPU process (footprint)" if fp else "GPU process (RSS)"
from statistics import median
def agg(rs):  # median over repeats of the same setting
    ok = [r for r in rs if "error" not in r]
    if not ok or "baseline" not in ok[0]: return rs[0]  # calibration and failures are kept as they are
    o = dict(ok[0])
    for part in ("baseline", "peak", "final"): o[part] = {k: median(r[part][k] for r in ok) for k in ok[0][part]}
    gr = [r["peak"][G] - r["baseline"][G] for r in ok]
    o.update(ms_per_tile=median(r["ms_per_tile"] for r in ok), n=len(ok), growth_min=min(gr), growth_max=max(gr)); return o
groups = {}
for r in R: groups.setdefault((r["cfg"]["kind"], r["cfg"].get("b"), r["cfg"].get("s")) if r["cfg"]["kind"] in ("batch", "size") else id(r), []).append(r)
R = [agg(v) for v in groups.values()]
kind = lambda k: [r for r in R if r["cfg"]["kind"] == k]
cal = next(iter(kind("calibration")), None); long_ = next(iter(kind("long")), None)
batch = [r for r in kind("batch") if "error" not in r]; size = kind("size"); ok = [r for r in size if "error" not in r]

fig, ax = plt.subplots(2, 2, figsize=(12, 8)); ax = ax.ravel()
if cal and "steps" in cal:
    st = cal["steps"][:-1]; xs = [s["mb"] for s in st]; base = st[0][G]
    ax[0].plot(xs, [mb(s[G] - base) for s in st], "o-", label=gname); ax[0].plot(xs, [mb(s["gpu_kb"] - st[0]["gpu_kb"]) for s in st], "s--", color="gray", label="GPU process (RSS)")
    ax[0].plot(xs, xs, "k:", label="y = x"); ax[0].set_xlabel("WebGPU buffers allocated (MB)"); ax[0].set_ylabel("growth seen (MB)"); ax[0].legend(fontsize=8)
ax[0].set_title("Calibration: does this method see GPU memory?", fontsize=10)
for a, rs, xk, xl, t in ((ax[1], batch, "b", "batch size (256x256 tiles)", "Memory vs batch size"), (ax[2], ok, "s", "tile side (px), batch 1", "Memory vs tile size")):
    xs = [r["cfg"][xk] for r in rs]
    a.plot(xs, [mb(r["peak"][G] - r["baseline"][G]) for r in rs], "o-", color="#f58518", label="GPU, peak growth")
    a.plot(xs, [mb(r["peak"]["renderer_kb"] - r["baseline"]["renderer_kb"]) for r in rs], "s-", color="#54a24b", label="renderer (RSS), peak growth")
    a.plot(xs, [mb(r["peak"][G]) for r in rs], "o:", color="#f58518", alpha=.5, label="GPU, absolute peak")
    a.set_xscale("log", base=2); a.set_xlabel(xl); a.set_ylabel("MB"); a.set_title(t, fontsize=10); a.legend(fontsize=8)
for r in size:
    if "error" in r: ax[2].axvline(r["cfg"]["s"], color="r", ls="--", lw=1); ax[2].text(r["cfg"]["s"], 0, " failed", color="r", fontsize=8, rotation=90, va="bottom")
if long_ and "series" in long_ and long_["series"]:
    s = long_["series"]; ax[3].plot([p["iter"] for p in s], [mb(p[G]) for p in s], color="#f58518", label=gname); ax[3].plot([p["iter"] for p in s], [mb(p["renderer_kb"]) for p in s], color="#54a24b", label="renderer (RSS)")
    ax[3].set_xlabel("batches of 16 processed"); ax[3].set_ylabel("MB"); ax[3].legend(fontsize=8)
ax[3].set_title("Long run: does memory keep growing?", fontsize=10)
fig.suptitle("IFCRN on browser WebGPU, fresh browser per setting. Growth = peak minus the reading just before the model loads", fontsize=10); fig.tight_layout(); fig.savefig("results/memory.png", dpi=140)

rows = ["| setting | runs | ms/tile | GPU growth MB (range) | **GPU growth per tile MB** | GPU peak MB | GPU peak per tile MB | GPU after idle MB |", "|---|---|---|---|---|---|---|---|"]
for r in kind("batch") + size:
    c = r["cfg"]; name = f"batch {c['b']}, {c['s']}px" if c["kind"] == "batch" else f"tile {c['s']}px, batch 1"
    ntile = c["b"] * (c["s"] / 256) ** 2  # 256x256-tile equivalents
    rows.append(f"| {name} | | FAILED: {r['error'][:60]} | | | | | |" if "error" in r else f"| {name} | {r['n']} | {r['ms_per_tile']:.1f} | {mb(r['peak'][G] - r['baseline'][G]):.0f} ({mb(r['growth_min']):.0f}-{mb(r['growth_max']):.0f}) | **{mb(r['peak'][G] - r['baseline'][G]) / ntile:.0f}** | {mb(r['peak'][G]):.0f} | {mb(r['peak'][G]) / ntile:.0f} | {mb(r['final'][G]):.0f} |")
if long_ and long_.get("series"): s = long_["series"]; rows.append(f"\nLong run, {gname}: {mb(s[0][G]):.0f} MB after {s[0]['iter']} batches, {mb(s[-1][G]):.0f} MB after {s[-1]['iter']}; max {mb(max(p[G] for p in s)):.0f} MB.")
open("results/memory.md", "w").write("\n".join(rows) + "\n"); print("\n".join(rows)); print("\nwrote results/memory.png, results/memory.md")

if batch:  # per-tile view
    f2, a2 = plt.subplots(figsize=(8, 4.8)); xs = [str(r["cfg"]["b"]) for r in batch]; w = 0.38
    g = [mb(r["peak"][G] - r["baseline"][G]) / r["cfg"]["b"] for r in batch]; pk = [mb(r["peak"][G]) / r["cfg"]["b"] for r in batch]
    b1 = a2.bar([i - w / 2 for i in range(len(xs))], g, w, color="#f58518", label="growth per tile (peak - before model load)"); b2 = a2.bar([i + w / 2 for i in range(len(xs))], pk, w, color="#f5b97a", label="absolute peak / batch (includes ~0.5 GB browser baseline)")
    a2.bar_label(b1, fmt="%.0f", fontsize=8); a2.bar_label(b2, fmt="%.0f", fontsize=8); a2.set_xticks(range(len(xs))); a2.set_xticklabels([f"batch {x}" for x in xs]); a2.set_ylabel("GPU memory per tile (MB)")
    a3 = a2.twinx(); a3.plot(range(len(xs)), [r["ms_per_tile"] for r in batch], "ko-", label="ms per tile"); a3.set_ylabel("ms per tile"); a3.set_ylim(0, max(r["ms_per_tile"] for r in batch) * 1.3)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels(); a2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    a2.set_title(f"IFCRN 256x256 on browser WebGPU: GPU memory per tile (median of {batch[0]['n']} fresh-browser runs)", fontsize=9); f2.tight_layout(); f2.savefig("results/memory_per_tile.png", dpi=140)
