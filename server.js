// Tiny static server. POST /results?name=X saves results/X.json.
const http = require("http"), fs = require("fs"), path = require("path");
const { execFileSync } = require("child_process");
const root = __dirname, types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".jpg": "image/jpeg", ".json": "application/json" };
// Memory of the browser process tree (root pid in results/.browser_pid), from `ps` (macOS/Linux). RSS counts shared pages in every process.
function sample() {
  let root; try { root = +fs.readFileSync(path.join(__dirname, "results", ".browser_pid"), "utf8"); } catch { return { error: "no browser pid" }; }
  const rows = execFileSync("ps", ["-axo", "pid=,ppid=,rss=,command="], { maxBuffer: 1 << 26 }).toString().split("\n")
    .map((l) => l.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/)).filter(Boolean).map((m) => ({ pid: +m[1], ppid: +m[2], kb: +m[3], cmd: m[4] }));
  const tree = new Set([root]); for (let grew = true; grew; ) { grew = false; for (const r of rows) if (!tree.has(r.pid) && tree.has(r.ppid)) { tree.add(r.pid); grew = true; } }
  const o = { total_kb: 0, gpu_kb: 0, renderer_kb: 0, other_kb: 0, n: 0 };
  for (const r of rows) if (tree.has(r.pid)) { const k = /\(GPU\)|--type=gpu-process/.test(r.cmd) ? "gpu" : /\(Renderer\)|--type=renderer/.test(r.cmd) ? "renderer" : "other"; o.total_kb += r.kb; o[k + "_kb"] += r.kb; o.n++; }
  if (process.platform === "darwin") { // RSS misses GPU memory on Apple silicon; `footprint` (phys_footprint) is what Activity Monitor shows
    o.fp_total_kb = 0; o.fp_gpu_kb = 0; o.fp_renderer_kb = 0;
    for (const r of rows) if (tree.has(r.pid) && /\(GPU\)|\(Renderer\)|--type=gpu-process|--type=renderer/.test(r.cmd)) {
      try { const m = execFileSync("footprint", ["-p", String(r.pid)], { stdio: ["ignore", "pipe", "ignore"] }).toString().match(/Footprint:\s+([\d.]+)\s*(KB|MB|GB)/);
        const kb = m ? +m[1] * { KB: 1, MB: 1024, GB: 1048576 }[m[2]] : 0; o.fp_total_kb += kb; o[/GPU|gpu/.test(r.cmd) ? "fp_gpu_kb" : "fp_renderer_kb"] += kb; } catch {}
    }
  }
  return o;
}
http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname === "/sample") { res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store"); return res.end(JSON.stringify(sample())); }
  if (u.pathname === "/results" && req.method === "POST") {
    const name = (u.searchParams.get("name") || "results").replace(/[^\w-]/g, "");
    fs.mkdirSync(path.join(root, "results"), { recursive: true });
    const f = fs.createWriteStream(path.join(root, "results", name + ".json"));
    req.pipe(f); return f.on("finish", () => res.end("saved"));
  }
  const file = path.join(root, decodeURIComponent(u.pathname));
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.statusCode = 404; return res.end(); }
  const buf = fs.readFileSync(file);
  res.setHeader("Access-Control-Allow-Origin", "*"); res.setHeader("Cache-Control", "no-store");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin"); res.setHeader("Cross-Origin-Embedder-Policy", "require-corp"); res.setHeader("Cross-Origin-Resource-Policy", "cross-origin"); // needed for multi-threaded WASM
  res.setHeader("Content-Type", types[path.extname(file)] || "application/octet-stream"); res.setHeader("Content-Length", buf.length);
  res.end(buf);
}).listen(+process.env.PORT || 8765, () => console.log("listening"));
