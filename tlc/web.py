from __future__ import annotations

import json
import time
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tlc.detect import classify_scene_details
from tlc.engine import (
    classify_proba,
    get_default_model_path,
    load_metadata,
    resolve_threshold,
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TLC - Traffic Light Classifier</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .glow-red { box-shadow: 0 0 30px rgba(239, 68, 68, 0.85), inset 0 0 10px rgba(255, 255, 255, 0.6); }
    .glow-yellow { box-shadow: 0 0 30px rgba(234, 179, 8, 0.85), inset 0 0 10px rgba(255, 255, 255, 0.6); }
    .glow-green { box-shadow: 0 0 30px rgba(34, 197, 94, 0.85), inset 0 0 10px rgba(255, 255, 255, 0.6); }
    .housing { background: linear-gradient(145deg, #1e293b, #0f172a); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col items-center p-4 md:p-8 font-sans">
  <header class="w-full max-w-4xl flex items-center justify-between pb-6 mb-6 border-b border-slate-800">
    <div class="flex items-center space-x-3">
      <div class="w-10 h-10 rounded-xl bg-red-600/20 border border-red-500/30 flex items-center justify-center">
        <span class="w-3.5 h-3.5 rounded-full bg-red-500 animate-pulse"></span>
      </div>
      <div>
        <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
          TLC <span class="text-xs font-medium px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">Production Classifier</span>
        </h1>
        <p class="text-xs text-slate-400">Zero-Torch ONNX &bull; MobileNetV3 + YOLOv8 Detection &bull; 3 Output States</p>
      </div>
    </div>
    <div class="text-right hidden sm:block">
      <div class="text-xs text-slate-400">Threshold: <span id="hdr-th" class="font-mono text-emerald-400">0.160</span></div>
      <div class="text-xs text-slate-500">Val Red Recall: 98.2% (Calibrated)</div>
    </div>
  </header>

  <main class="w-full max-w-4xl grid grid-cols-1 md:grid-cols-12 gap-6">
    <div class="md:col-span-7 flex flex-col space-y-4">
      <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-red-500/60 rounded-2xl p-8 bg-slate-900/60 transition-all cursor-pointer flex flex-col items-center justify-center min-h-[200px] text-center group">
        <input type="file" id="file-input" multiple accept="image/*" class="hidden">
        <div class="w-12 h-12 rounded-full bg-slate-800 group-hover:bg-red-500/10 flex items-center justify-center mb-3 text-slate-400 group-hover:text-red-400 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
          </svg>
        </div>
        <p class="text-sm font-medium text-slate-200">Drag & drop traffic light or street images here</p>
        <p class="text-xs text-slate-400 mt-1">or <span class="text-red-400 underline font-semibold">browse files</span> (supports JPG, PNG, WEBP)</p>
      </div>

      <div class="bg-slate-900/40 border border-slate-800/80 rounded-xl p-3 flex items-center justify-between">
        <label for="scene-toggle" class="flex items-center space-x-2 cursor-pointer">
          <input type="checkbox" id="scene-toggle" checked class="w-4 h-4 rounded bg-slate-800 border-slate-700 text-red-500 focus:ring-0 cursor-pointer">
          <span class="text-xs text-slate-300 font-medium">Scene Auto-Detect (YOLOv8)</span>
        </label>
        <span class="text-[11px] text-slate-500">Finds light or returns "no traffic light"</span>
      </div>

      <div class="bg-slate-900/40 border border-slate-800/80 rounded-xl p-4 space-y-3">
        <div class="flex items-center justify-between">
          <label class="text-xs font-semibold text-slate-300">Decision Threshold (<span id="th-val">0.160</span>)</label>
          <button id="reset-th" class="text-[11px] text-red-400 hover:text-red-300">Reset default (0.160)</button>
        </div>
        <input type="range" id="threshold-slider" min="0.01" max="0.99" step="0.01" value="0.16" class="w-full accent-red-500 cursor-pointer">
        <div class="flex justify-between text-[10px] text-slate-500 font-mono">
          <span>0.01 (High Recall)</span>
          <span>0.16 (Calibrated Val)</span>
          <span>0.99 (High Precision)</span>
        </div>
      </div>

      <div id="batch-card" class="hidden bg-slate-900/60 border border-slate-800 rounded-xl p-4">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Uploaded Queue</h3>
        <div id="batch-list" class="space-y-2 max-h-48 overflow-y-auto pr-1"></div>
      </div>
    </div>

    <div class="md:col-span-5 flex flex-col space-y-4">
      <div class="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 flex flex-col items-center">
        <div class="housing p-4 rounded-3xl border border-slate-700/80 shadow-2xl flex flex-col space-y-3 mb-6">
          <div id="lamp-red" class="w-14 h-14 rounded-full bg-red-950/40 border border-red-950 transition-all duration-300"></div>
          <div id="lamp-yellow" class="w-14 h-14 rounded-full bg-yellow-950/40 border border-yellow-950 transition-all duration-300"></div>
          <div id="lamp-green" class="w-14 h-14 rounded-full bg-green-950/40 border border-green-950 transition-all duration-300"></div>
        </div>

        <div id="result-badge" class="px-6 py-2 rounded-full text-sm font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700 mb-4 transition-all text-center">
          Awaiting Image
        </div>

        <div class="w-full space-y-2 text-xs">
          <div class="flex justify-between text-slate-400">
            <span>Red Probability (p_red)</span>
            <span id="prob-num" class="font-mono text-slate-200">-</span>
          </div>
          <div class="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
            <div id="prob-bar" class="h-full bg-slate-700 transition-all duration-300" style="width: 0%"></div>
          </div>
          <div class="flex justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
            <span>Latency</span>
            <span id="latency-num" class="font-mono text-slate-300">-</span>
          </div>
        </div>

        <div id="preview-box" class="w-full mt-4 aspect-square rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center overflow-hidden">
          <img id="preview-img" class="w-full h-full object-contain hidden" alt="Uploaded Preview">
          <span id="preview-placeholder" class="text-xs text-slate-600">No image loaded</span>
        </div>
      </div>
    </div>
  </main>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const slider = document.getElementById('threshold-slider');
    const thVal = document.getElementById('th-val');
    const hdrTh = document.getElementById('hdr-th');
    const resetTh = document.getElementById('reset-th');
    const sceneToggle = document.getElementById('scene-toggle');
    const lampRed = document.getElementById('lamp-red');
    const lampYellow = document.getElementById('lamp-yellow');
    const lampGreen = document.getElementById('lamp-green');
    const resultBadge = document.getElementById('result-badge');
    const probNum = document.getElementById('prob-num');
    const probBar = document.getElementById('prob-bar');
    const latencyNum = document.getElementById('latency-num');
    const previewImg = document.getElementById('preview-img');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const batchCard = document.getElementById('batch-card');
    const batchList = document.getElementById('batch-list');

    let currentFile = null;
    let currentData = null;

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('border-red-500'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('border-red-500'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-red-500');
      if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleFiles(e.target.files);
    });

    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value).toFixed(2);
      thVal.textContent = val;
      hdrTh.textContent = val;
      if (currentData !== null) renderResult(currentData, parseFloat(val));
    });

    resetTh.addEventListener('click', () => {
      slider.value = 0.16;
      thVal.textContent = "0.160";
      hdrTh.textContent = "0.160";
      if (currentData !== null) renderResult(currentData, 0.16);
    });

    sceneToggle.addEventListener('change', () => {
      if (currentFile !== null) uploadAndClassify(currentFile, false);
    });

    async function handleFiles(files) {
      if (!files || !files.length) return;
      currentFile = files[0];
      previewImg.src = URL.createObjectURL(currentFile);
      previewImg.classList.remove('hidden');
      previewPlaceholder.classList.add('hidden');

      if (files.length > 1) {
        batchCard.classList.remove('hidden');
        batchList.innerHTML = '';
        for (const f of files) {
          uploadAndClassify(f, true);
        }
      } else {
        uploadAndClassify(currentFile, false);
      }
    }

    async function uploadAndClassify(file, isBatch) {
      const buf = await file.arrayBuffer();
      const params = new URLSearchParams({
        filename: file.name,
        threshold: slider.value,
        scene: sceneToggle.checked ? 'true' : 'false'
      });

      try {
        const res = await fetch('/api/classify?' + params.toString(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: buf
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        if (!isBatch || file === currentFile) {
          currentData = data;
          latencyNum.textContent = `${data.latency_ms.toFixed(1)} ms`;
          renderResult(data, parseFloat(slider.value));
        }

        if (isBatch) {
          const item = document.createElement('div');
          item.className = 'flex items-center justify-between text-xs bg-slate-950/60 p-2 rounded-lg border border-slate-800';
          let tagClass = 'bg-slate-800 text-slate-400 border border-slate-700';
          let tagText = 'NO LIGHT';
          if (data.label === 'red') {
            tagClass = 'bg-red-500/20 text-red-400 border border-red-500/30';
            tagText = 'RED';
          } else if (data.label === 'not_red') {
            tagClass = 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
            tagText = 'NOT RED';
          }
          item.innerHTML = `
            <span class="truncate max-w-[180px] font-mono">${file.name}</span>
            <div class="flex items-center space-x-2">
              <span class="font-mono text-slate-400">${data.p_red !== null ? (data.p_red * 100).toFixed(1) + '%' : 'N/A'}</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-bold ${tagClass}">${tagText}</span>
            </div>
          `;
          batchList.appendChild(item);
        }
      } catch (err) {
        console.error(err);
      }
    }

    function renderResult(data, threshold) {
      const label = data.label;
      const pRed = data.p_red;

      if (label === 'no_traffic_light_found') {
        probNum.textContent = 'N/A (No light detected)';
        probBar.style.width = '0%';
        probBar.className = 'h-full bg-slate-700 transition-all duration-300';

        lampRed.className = 'w-14 h-14 rounded-full border bg-red-950/40 border-red-950 transition-all duration-300';
        lampYellow.className = 'w-14 h-14 rounded-full border bg-yellow-950/40 border-yellow-950 transition-all duration-300';
        lampGreen.className = 'w-14 h-14 rounded-full border bg-green-950/40 border-green-950 transition-all duration-300';

        resultBadge.className = 'px-6 py-2 rounded-full text-sm font-bold uppercase tracking-wider bg-slate-800/90 text-amber-300 border border-amber-500/40 mb-4 transition-all text-center';
        resultBadge.textContent = 'NO TRAFFIC LIGHT FOUND';
        return;
      }

      const isRed = pRed >= threshold;
      probNum.textContent = `${(pRed * 100).toFixed(1)}% (${pRed.toFixed(4)})`;
      probBar.style.width = `${Math.min(100, pRed * 100)}%`;

      lampRed.className = 'w-14 h-14 rounded-full border transition-all duration-300 ' + (isRed ? 'bg-red-500 glow-red border-red-300' : 'bg-red-950/40 border-red-950');
      lampYellow.className = 'w-14 h-14 rounded-full border transition-all duration-300 ' + (!isRed && pRed > 0.15 ? 'bg-yellow-400 glow-yellow border-yellow-200' : 'bg-yellow-950/40 border-yellow-950');
      lampGreen.className = 'w-14 h-14 rounded-full border transition-all duration-300 ' + (!isRed && pRed <= 0.15 ? 'bg-green-500 glow-green border-green-200' : 'bg-green-950/40 border-green-950');

      if (isRed) {
        resultBadge.className = 'px-6 py-2 rounded-full text-sm font-bold uppercase tracking-wider bg-red-500/20 text-red-400 border border-red-500/40 glow-red mb-4 transition-all text-center';
        resultBadge.textContent = 'RED LIGHT';
        probBar.className = 'h-full bg-red-500 transition-all duration-300';
      } else {
        resultBadge.className = 'px-6 py-2 rounded-full text-sm font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 glow-green mb-4 transition-all text-center';
        resultBadge.textContent = 'NOT RED';
        probBar.className = 'h-full bg-emerald-500 transition-all duration-300';
      }
    }
  </script>
</body>
</html>
"""


class TLCRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/metadata":
            meta = load_metadata(get_default_model_path())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(meta).encode("utf-8"))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/api/classify":
            self.handle_classify(parsed_url.query)
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def handle_classify(self, query_string: str) -> None:
        params = urllib.parse.parse_qs(query_string)
        filename = params.get("filename", ["uploaded_image.jpg"])[0]
        th_val = params.get("threshold", [None])[0]
        scene_mode = params.get("scene", ["true"])[0].lower() == "true"
        th_override = float(th_val) if th_val else None

        length = int(self.headers.get("Content-Length", 0))
        img_bytes = self.rfile.read(length)

        if not img_bytes:
            self.send_json_error("Empty image bytes received", HTTPStatus.BAD_REQUEST)
            return

        try:
            t0 = time.perf_counter()
            active_th = resolve_threshold(get_default_model_path(), th_override)

            if scene_mode:
                details = classify_scene_details(
                    img_bytes,
                    threshold=active_th,
                    raise_if_empty=False,
                )
                if not details:
                    label = "no_traffic_light_found"
                    p_red: float | None = None
                    is_red_val = False
                else:
                    is_red_val = any(d["label"] == "red" for d in details)
                    label = "red" if is_red_val else "not_red"
                    p_red = round(max(d["p_red"] for d in details), 4)
            else:
                p_red_calc = classify_proba(img_bytes)
                p_red = round(p_red_calc, 4)
                label = "red" if p_red_calc >= active_th else "not_red"
                is_red_val = bool(label == "red")

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            resp = {
                "filename": filename,
                "label": label,
                "p_red": p_red,
                "threshold": round(active_th, 4),
                "is_red": is_red_val,
                "latency_ms": round(latency_ms, 2),
            }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self.send_json_error(str(e), HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json_error(self, message: str, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    server = ThreadingHTTPServer((host, port), TLCRequestHandler)
    url = f"http://{host}:{port}"
    print(f"\n* TLC Web Interface running at: {url}")
    print("* Drag & drop images to classify in real-time.")
    print("* Press Ctrl+C to stop the server.\n")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down TLC Web Server...")
        server.server_close()
