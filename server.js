// Tiny static server. POST /results?name=X saves results/X.json.
const http = require("http"), fs = require("fs"), path = require("path");
const root = __dirname, types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".jpg": "image/jpeg", ".json": "application/json" };
http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");
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
