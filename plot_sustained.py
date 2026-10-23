"""Reads results/sustained.json (+ sustained_system.json on macOS); writes results/sustained.png and results/sustained.md."""
import json, os
from statistics import median
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = json.load(open("results/sustained.json")); S = D["series"]; C = D["cfg"]
t = [p["t_s"] / 60 for p in S]; y = [p["ms_per_patch"] for p in S]; n = len(y); k = max(1, n // 10)
first, last = median(y[:k]), median(y[-k:]); slow = (last / first - 1) * 100
roll = [median(y[max(0, i - 5):i + 6]) for i in range(n)]
fig, ax = plt.subplots(figsize=(9, 4.8)); ax.plot(t, y, color="#aec7e8", lw=0.8, label="per window of 10 batches"); ax.plot(t, roll, color="#1f77b4", lw=2, label="rolling median")
ax.axhline(first, color="gray", ls=":", lw=1); ax.set_ylim(0, max(y) * 1.15); ax.set_xlabel("minutes"); ax.set_ylabel("ms per patch (lower is better)"); ax.legend(fontsize=8)
ax.set_title(f"{C['model']} on browser WebGPU, batch {C['batch']}, {C['patches']:,} patches: first 10% {first:.2f} ms, last 10% {last:.2f} ms ({slow:+.0f}%)", fontsize=9)
fig.tight_layout(); fig.savefig("results/sustained.png", dpi=140)
lines = [f"- {C['model']}, batch {C['batch']}, {C['patches']:,} patches in {D['total_s'] / 60:.1f} min ({C['patches'] / D['total_s']:.0f} patches/s on average)",
         f"- ms per patch: first 10% of the run {first:.2f}, last 10% {last:.2f} ({slow:+.0f}%); fastest window {min(y):.2f}, slowest {max(y):.2f}"]
if os.path.exists("results/sustained_system.json"):
    sysd = json.load(open("results/sustained_system.json"))["samples"]
    warn = [s for s in sysd if "No thermal warning" not in s["therm"] or "No performance warning" not in s["therm"] or "No CPU power" not in s["therm"]]
    lines += [f"- power: {sysd[0]['batt']} -> {sysd[-1]['batt']}", f"- macOS thermal / performance warnings recorded: {'none in ' + str(len(sysd)) + ' samples' if not warn else str(len(warn)) + ' of ' + str(len(sysd)) + ' samples, e.g. ' + warn[0]['therm'][:200]}"]
open("results/sustained.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("\nwrote results/sustained.png, results/sustained.md")
