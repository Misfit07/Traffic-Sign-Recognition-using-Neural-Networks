from __future__ import annotations

import base64
import io
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
from tlc.sign_engine import apply_clahe, classify_sign, load_image, load_sign_labels

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TLC - Traffic Light Classifier</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glow-red { box-shadow: 0 0 35px rgba(239, 68, 68, 0.9), inset 0 0 15px rgba(255, 255, 255, 0.6); }
    .glow-yellow { box-shadow: 0 0 35px rgba(234, 179, 8, 0.9), inset 0 0 15px rgba(255, 255, 255, 0.6); }
    .glow-green { box-shadow: 0 0 35px rgba(34, 197, 94, 0.9), inset 0 0 15px rgba(255, 255, 255, 0.6); }
    .signal-housing { background: linear-gradient(160deg, #1e293b 0%, #0f172a 100%); }
    .stripe-pattern {
      background-size: 30px 30px;
      background-image: radial-gradient(circle, #e2e8f0 1.5px, transparent 1.5px);
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen flex flex-col stripe-pattern antialiased">
  <header class="w-full bg-white/90 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-50 shadow-sm">
    <div class="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <a href="/" class="flex items-center space-x-2.5 group">
          <div class="flex items-center space-x-1 p-2 bg-slate-900 rounded-xl shadow-md group-hover:scale-105 transition-transform">
            <span class="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-yellow-400"></span>
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
          </div>
          <div class="flex flex-col">
            <div class="flex items-center space-x-1.5">
              <span class="text-xl font-extrabold tracking-tight text-slate-900">TLC</span>
            </div>
          </div>
        </a>
      </div>

      <div class="inline-flex p-1 bg-slate-100 rounded-2xl border border-slate-200 text-xs font-bold">
        <button id="btn-mode-light" onclick="setMode('light')" class="px-4 py-1.5 rounded-xl bg-white text-slate-900 shadow-sm transition-all flex items-center space-x-1.5">
          <span>Traffic Signals</span>
        </button>
        <button id="btn-mode-sign" onclick="setMode('sign')" class="px-4 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition-all flex items-center space-x-1.5">
          <span>Traffic Signs</span>
        </button>
      </div>

      <div class="flex items-center space-x-3">
        <div id="hdr-stat-box" class="hidden sm:flex items-center space-x-2 text-xs font-semibold px-3 py-1.5 rounded-xl bg-slate-100 border border-slate-200">
          <span id="hdr-stat-lbl" class="text-slate-500">Threshold:</span>
          <span id="hdr-stat-val" class="mono text-red-600 font-bold">0.160</span>
        </div>
        <a href="https://github.com/Misfit07/Traffic-Sign-Recognition-using-Neural-Networks" target="_blank" rel="noopener" class="text-xs font-bold px-3 py-1.5 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-colors flex items-center space-x-1.5 shadow-sm">
          <svg class="w-4 h-4 fill-current" viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
          <span class="hidden sm:inline">GitHub</span>
        </a>
      </div>
    </div>
  </header>

  <main class="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8 md:py-12 flex flex-col items-center">
    <div class="text-center max-w-2xl mx-auto mb-8">
      <h1 id="hero-title" class="text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-3">
        Traffic Light Classifier
      </h1>
      <p id="hero-subtitle" class="text-sm sm:text-base text-slate-600 font-medium leading-relaxed">
        Upload single crops or full street scene photos. Evaluates into
        <span class="font-bold text-red-600">"red"</span>,
        <span class="font-bold text-emerald-600">"not_red"</span>, or
        <span class="font-bold text-slate-700">"no_traffic_light_found"</span>.
      </p>
    </div>

    <div class="w-full bg-white rounded-3xl border border-slate-200/90 shadow-xl shadow-slate-200/50 overflow-hidden mb-8 transition-all">
      <div id="dropzone" class="p-8 sm:p-12 text-center cursor-pointer transition-all duration-300 hover:bg-slate-50/70 border-b border-slate-100 flex flex-col items-center justify-center min-h-[240px] group">
        <input type="file" id="file-input" multiple accept="image/*" class="hidden">

        <button type="button" class="px-8 py-4 bg-red-600 hover:bg-red-700 active:bg-red-800 text-white font-bold text-base sm:text-lg rounded-2xl shadow-lg shadow-red-600/30 group-hover:scale-105 transition-all flex items-center space-x-3 mb-4 pointer-events-none">
          <svg class="w-6 h-6 stroke-current stroke-2 fill-none" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
          <span id="btn-select-text">Select Traffic Light Images</span>
        </button>

        <p class="text-sm font-semibold text-slate-700 mb-1">or drag and drop images here</p>
        <p class="text-xs text-slate-400">Supports JPG, PNG, WEBP, PPM &bull; Single image or batch drop</p>
      </div>

      <div class="bg-slate-50/80 px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs border-t border-slate-100">
        <div id="ctrl-light-left" class="flex items-center space-x-3 w-full sm:w-auto">
          <label for="scene-toggle" class="flex items-center space-x-2.5 cursor-pointer select-none">
            <input type="checkbox" id="scene-toggle" checked class="w-4 h-4 rounded text-red-600 focus:ring-0 border-slate-300 accent-red-600 cursor-pointer">
            <span class="font-bold text-slate-700">Scene Auto-Detect (YOLOv8)</span>
          </label>
          <span class="text-slate-400 hidden sm:inline">&bull;</span>
          <span class="text-slate-500 hidden sm:inline">Crops fixture or reports "no signal"</span>
        </div>

        <div id="ctrl-sign-left" class="hidden items-center space-x-3 w-full sm:w-auto">
          <span class="font-bold text-slate-700">GTSRB Benchmark</span>
          <span class="text-slate-400 hidden sm:inline">&bull;</span>
          <span class="text-slate-500 hidden sm:inline">CLAHE Contrast Equalization Enabled</span>
        </div>

        <div class="flex items-center space-x-4 w-full sm:w-auto justify-between sm:justify-end">
          <div class="flex items-center space-x-2">
            <span id="slider-label" class="font-semibold text-slate-600">Threshold:</span>
            <input type="range" id="threshold-slider" min="0.01" max="0.99" step="0.01" value="0.16" class="w-28 sm:w-36 accent-red-600 cursor-pointer">
            <span id="th-val" class="mono font-bold text-slate-800 w-10 text-right">0.16</span>
          </div>
          <button id="reset-th" class="font-semibold text-red-600 hover:text-red-700 text-xs">Reset</button>
        </div>
      </div>
    </div>

    <div id="results-panel" class="w-full grid grid-cols-1 md:grid-cols-12 gap-6 transition-all duration-300">
      <div class="md:col-span-7 bg-white rounded-3xl border border-slate-200/90 shadow-xl shadow-slate-200/50 p-6 flex flex-col">
        <div class="flex items-center justify-between pb-4 mb-4 border-b border-slate-100">
          <h2 class="text-sm font-bold uppercase tracking-wider text-slate-500">Image Inspection</h2>
          <span id="active-filename" class="mono text-xs font-semibold text-slate-600 truncate max-w-[200px]">None loaded</span>
        </div>

        <div id="preview-box" class="w-full aspect-[4/3] rounded-2xl bg-slate-100 border border-slate-200/80 flex items-center justify-center overflow-hidden relative group">
          <img id="preview-img" class="w-full h-full object-contain hidden" alt="Uploaded Preview">
          <div id="preview-placeholder" class="text-center p-6 text-slate-400 flex flex-col items-center">
            <svg class="w-12 h-12 stroke-current stroke-1 mb-2 text-slate-300" viewBox="0 0 24 24" fill="none">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
            </svg>
            <span class="text-xs font-semibold">Drop or select an image to inspect</span>
          </div>
        </div>

        <div id="clahe-box" class="mt-4 p-3 bg-slate-50 rounded-2xl border border-slate-200/80 hidden">
          <div class="flex items-center justify-between mb-2">
            <span class="text-[11px] font-bold text-slate-500 uppercase tracking-wider">CLAHE Equalized Enhancement</span>
            <span class="text-[10px] text-slate-400">Adaptive LAB Contrast</span>
          </div>
          <div class="w-24 h-24 rounded-xl overflow-hidden border border-slate-300 bg-white shadow-inner">
            <img id="clahe-img" class="w-full h-full object-cover" alt="CLAHE Preview">
          </div>
        </div>

        <div id="batch-container" class="mt-6 pt-4 border-t border-slate-100 hidden">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Batch Queue (<span id="batch-count">0</span>)</h3>
          <div id="batch-list" class="space-y-2 max-h-48 overflow-y-auto pr-1"></div>
        </div>
      </div>

      <div class="md:col-span-5 bg-white rounded-3xl border border-slate-200/90 shadow-xl shadow-slate-200/50 p-6 sm:p-8 flex flex-col items-center justify-between">
        <div id="panel-light-mode" class="w-full flex flex-col items-center">
          <span class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-6">Signal Simulation</span>

          <div class="signal-housing p-4 rounded-3xl border border-slate-700 shadow-2xl flex flex-col space-y-3.5 mb-6">
            <div id="lamp-red" class="w-16 h-16 rounded-full bg-red-950/60 border border-red-900/40 transition-all duration-300"></div>
            <div id="lamp-yellow" class="w-16 h-16 rounded-full bg-yellow-950/60 border border-yellow-900/40 transition-all duration-300"></div>
            <div id="lamp-green" class="w-16 h-16 rounded-full bg-green-950/60 border border-green-900/40 transition-all duration-300"></div>
          </div>

          <div id="result-badge" class="px-6 py-2 rounded-full text-base font-extrabold tracking-wide bg-slate-100 text-slate-500 border border-slate-200 shadow-sm mb-6 transition-all text-center">
            Awaiting Image
          </div>

          <div class="w-full space-y-3 bg-slate-50 p-4 rounded-2xl border border-slate-200/70 text-xs">
            <div class="flex justify-between items-center text-slate-600 font-semibold">
              <span>Red Probability</span>
              <span id="prob-num" class="mono font-bold text-slate-900">-</span>
            </div>
            <div class="w-full h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div id="prob-bar" class="h-full bg-slate-400 transition-all duration-500 rounded-full" style="width: 0%"></div>
            </div>
            <div class="flex justify-between items-center text-[11px] text-slate-500 pt-2 border-t border-slate-200/70 font-medium">
              <span>Inference Latency</span>
              <span id="latency-num" class="mono font-bold text-slate-800">-</span>
            </div>
          </div>
        </div>

        <div id="panel-sign-mode" class="w-full flex flex-col items-center hidden">
          <span class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4">Traffic Sign Recognition</span>

          <div id="sign-symbol-card" class="w-24 h-24 rounded-3xl bg-slate-100 border-2 border-slate-200 shadow-md flex items-center justify-center text-slate-700 mb-4 transition-all">
            <svg class="w-10 h-10 stroke-current stroke-2 fill-none" viewBox="0 0 24 24">
              <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon>
            </svg>
          </div>

          <div id="sign-result-badge" class="px-5 py-2 rounded-full text-sm font-extrabold tracking-wide bg-slate-100 text-slate-600 border border-slate-200 shadow-sm mb-4 text-center max-w-full">
            Awaiting Sign
          </div>

          <div id="sign-warning" class="w-full p-2.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-semibold mb-4 text-center hidden">
            Low Confidence (&lt;<span id="warn-conf-limit">75%</span>): Sign may be out-of-distribution or degraded.
          </div>

          <div class="w-full space-y-2 bg-slate-50 p-4 rounded-2xl border border-slate-200/70 text-xs">
            <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">Top 3 Predictions</div>
            <div id="top3-list" class="space-y-2.5">
              <div class="text-slate-400 text-center py-2">No sign processed</div>
            </div>
            <div class="flex justify-between items-center text-[11px] text-slate-500 pt-3 border-t border-slate-200/70 font-medium">
              <span>ONNX Latency</span>
              <span id="sign-latency-num" class="mono font-bold text-slate-800">-</span>
            </div>
          </div>
        </div>

        <div class="w-full mt-6 text-center text-[11px] text-slate-400 font-medium">
          Zero-Torch ONNX &bull; MobileNetV3 + GTSRB CNN &bull; CPU / CUDA
        </div>
      </div>
    </div>
  </main>

  <footer class="w-full bg-white border-t border-slate-200/80 py-6 text-center text-xs text-slate-500 font-medium mt-auto">
    <div class="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div class="flex items-center space-x-2">
        <span class="font-bold text-slate-800">TLC Production</span>
        <span>&bull;</span>
        <span>Signals (S2TLD/TLoNY) & Signs (GTSRB)</span>
      </div>
      <div>
        &copy; 2026 TLC &bull; Embedded ONNX Runtime Engine
      </div>
    </div>
  </footer>

  <script>
    let activeMode = 'light';
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const slider = document.getElementById('threshold-slider');
    const thVal = document.getElementById('th-val');
    const hdrStatVal = document.getElementById('hdr-stat-val');
    const hdrStatLbl = document.getElementById('hdr-stat-lbl');
    const resetTh = document.getElementById('reset-th');
    const sceneToggle = document.getElementById('scene-toggle');
    const sliderLabel = document.getElementById('slider-label');
    const btnSelectText = document.getElementById('btn-select-text');
    const heroTitle = document.getElementById('hero-title');
    const heroSubtitle = document.getElementById('hero-subtitle');

    const btnModeLight = document.getElementById('btn-mode-light');
    const btnModeSign = document.getElementById('btn-mode-sign');
    const ctrlLightLeft = document.getElementById('ctrl-light-left');
    const ctrlSignLeft = document.getElementById('ctrl-sign-left');
    const panelLightMode = document.getElementById('panel-light-mode');
    const panelSignMode = document.getElementById('panel-sign-mode');

    const lampRed = document.getElementById('lamp-red');
    const lampYellow = document.getElementById('lamp-yellow');
    const lampGreen = document.getElementById('lamp-green');
    const resultBadge = document.getElementById('result-badge');
    const probNum = document.getElementById('prob-num');
    const probBar = document.getElementById('prob-bar');
    const latencyNum = document.getElementById('latency-num');

    const signSymbolCard = document.getElementById('sign-symbol-card');
    const signResultBadge = document.getElementById('sign-result-badge');
    const signWarning = document.getElementById('sign-warning');
    const warnConfLimit = document.getElementById('warn-conf-limit');
    const top3List = document.getElementById('top3-list');
    const signLatencyNum = document.getElementById('sign-latency-num');
    const claheBox = document.getElementById('clahe-box');
    const claheImg = document.getElementById('clahe-img');

    const previewImg = document.getElementById('preview-img');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const activeFilename = document.getElementById('active-filename');
    const batchContainer = document.getElementById('batch-container');
    const batchList = document.getElementById('batch-list');
    const batchCount = document.getElementById('batch-count');

    let currentFile = null;
    let currentData = null;

    function setMode(mode) {
      activeMode = mode;
      if (mode === 'light') {
        btnModeLight.className = 'px-4 py-1.5 rounded-xl bg-white text-slate-900 shadow-sm transition-all flex items-center space-x-1.5';
        btnModeSign.className = 'px-4 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition-all flex items-center space-x-1.5';
        ctrlLightLeft.classList.remove('hidden');
        ctrlSignLeft.classList.add('hidden');
        panelLightMode.classList.remove('hidden');
        panelSignMode.classList.add('hidden');
        claheBox.classList.add('hidden');

        heroTitle.textContent = 'Traffic Light Classifier';
        heroSubtitle.innerHTML = 'Upload single crops or full street scene photos. Evaluates into <span class=\"font-bold text-red-600\">\"red\"</span>, <span class=\"font-bold text-emerald-600\">\"not_red\"</span>, or <span class=\"font-bold text-slate-700\">\"no_traffic_light_found\"</span>.';
        btnSelectText.textContent = 'Select Traffic Light Images';
        sliderLabel.textContent = 'Threshold:';
        slider.min = '0.01';
        slider.max = '0.99';
        slider.step = '0.01';
        slider.value = '0.16';
        thVal.textContent = '0.16';
        hdrStatLbl.textContent = 'Threshold:';
        hdrStatVal.textContent = '0.160';
      } else {
        btnModeSign.className = 'px-4 py-1.5 rounded-xl bg-white text-slate-900 shadow-sm transition-all flex items-center space-x-1.5';
        btnModeLight.className = 'px-4 py-1.5 rounded-xl text-slate-600 hover:text-slate-900 transition-all flex items-center space-x-1.5';
        ctrlLightLeft.classList.add('hidden');
        ctrlSignLeft.classList.remove('hidden');
        panelLightMode.classList.add('hidden');
        panelSignMode.classList.remove('hidden');

        heroTitle.textContent = 'Traffic Sign Recognition';
        heroSubtitle.innerHTML = 'Classify European & German standard traffic signs into <span class=\"font-bold text-slate-800\">GTSRB benchmark categories</span> with CLAHE contrast enhancement.';
        btnSelectText.textContent = 'Select Traffic Sign Images';
        sliderLabel.textContent = 'Min Conf:';
        slider.min = '30';
        slider.max = '99';
        slider.step = '1';
        slider.value = '75';
        thVal.textContent = '75%';
        hdrStatLbl.textContent = 'Min Conf:';
        hdrStatVal.textContent = '75%';
      }

      if (currentFile !== null) {
        uploadAndClassify(currentFile, false);
      }
    }

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('bg-red-50/50', 'border-red-400');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('bg-red-50/50', 'border-red-400');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('bg-red-50/50', 'border-red-400');
      if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleFiles(e.target.files);
    });

    slider.addEventListener('input', (e) => {
      if (activeMode === 'light') {
        const val = parseFloat(e.target.value).toFixed(2);
        thVal.textContent = val;
        hdrStatVal.textContent = val;
        if (currentData !== null) renderResultLight(currentData, parseFloat(val));
      } else {
        const val = parseInt(e.target.value, 10);
        thVal.textContent = val + '%';
        hdrStatVal.textContent = val + '%';
        if (currentData !== null) renderResultSign(currentData, val);
      }
    });

    resetTh.addEventListener('click', () => {
      if (activeMode === 'light') {
        slider.value = 0.16;
        thVal.textContent = '0.16';
        hdrStatVal.textContent = '0.160';
        if (currentData !== null) renderResultLight(currentData, 0.16);
      } else {
        slider.value = 75;
        thVal.textContent = '75%';
        hdrStatVal.textContent = '75%';
        if (currentData !== null) renderResultSign(currentData, 75);
      }
    });

    sceneToggle.addEventListener('change', () => {
      if (currentFile !== null && activeMode === 'light') uploadAndClassify(currentFile, false);
    });

    async function handleFiles(files) {
      if (!files || !files.length) return;
      currentFile = files[0];
      activeFilename.textContent = currentFile.name;
      previewImg.src = URL.createObjectURL(currentFile);
      previewImg.classList.remove('hidden');
      previewPlaceholder.classList.add('hidden');

      if (files.length > 1) {
        batchContainer.classList.remove('hidden');
        batchCount.textContent = files.length;
        batchList.innerHTML = '';
        for (const f of files) {
          uploadAndClassify(f, true);
        }
      } else {
        batchContainer.classList.add('hidden');
        uploadAndClassify(currentFile, false);
      }
    }

    async function uploadAndClassify(file, isBatch) {
      const buf = await file.arrayBuffer();

      let endpoint = '';
      if (activeMode === 'light') {
        const params = new URLSearchParams({
          filename: file.name,
          threshold: slider.value,
          scene: sceneToggle.checked ? 'true' : 'false'
        });
        endpoint = '/api/classify?' + params.toString();
      } else {
        const params = new URLSearchParams({
          filename: file.name,
          confidence_threshold: slider.value
        });
        endpoint = '/api/classify_sign?' + params.toString();
      }

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: buf
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        if (!isBatch || file === currentFile) {
          currentData = data;
          if (activeMode === 'light') {
            latencyNum.textContent = `${data.latency_ms.toFixed(1)} ms`;
            renderResultLight(data, parseFloat(slider.value));
          } else {
            signLatencyNum.textContent = `${data.latency_ms.toFixed(1)} ms`;
            renderResultSign(data, parseFloat(slider.value));
          }
        }

        if (isBatch) {
          const item = document.createElement('div');
          item.className = 'flex items-center justify-between text-xs bg-slate-50 p-2.5 rounded-xl border border-slate-200 cursor-pointer hover:bg-slate-100 transition-colors';

          let tagClass = 'bg-slate-200 text-slate-700';
          let tagText = 'UNKNOWN';
          let metric = 'N/A';

          if (activeMode === 'light') {
            if (data.label === 'red') {
              tagClass = 'bg-red-100 text-red-700 border border-red-300';
              tagText = 'RED';
            } else if (data.label === 'not_red') {
              tagClass = 'bg-emerald-100 text-emerald-700 border border-emerald-300';
              tagText = 'NOT RED';
            } else {
              tagText = 'NO LIGHT';
            }
            metric = data.p_red !== null ? (data.p_red * 100).toFixed(1) + '%' : 'N/A';
          } else {
            tagClass = data.is_confident ? 'bg-red-100 text-red-700 border border-red-300' : 'bg-amber-100 text-amber-700 border border-amber-300';
            tagText = data.label;
            metric = `${data.confidence.toFixed(1)}%`;
          }

          item.innerHTML = `
            <span class="truncate max-w-[180px] mono font-medium text-slate-700">${file.name}</span>
            <div class="flex items-center space-x-2">
              <span class="mono text-slate-500 font-bold">${metric}</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-extrabold ${tagClass} truncate max-w-[120px]">${tagText}</span>
            </div>
          `;
          item.addEventListener('click', () => {
            currentFile = file;
            currentData = data;
            activeFilename.textContent = file.name;
            previewImg.src = URL.createObjectURL(file);
            if (activeMode === 'light') {
              latencyNum.textContent = `${data.latency_ms.toFixed(1)} ms`;
              renderResultLight(data, parseFloat(slider.value));
            } else {
              signLatencyNum.textContent = `${data.latency_ms.toFixed(1)} ms`;
              renderResultSign(data, parseFloat(slider.value));
            }
          });
          batchList.appendChild(item);
        }
      } catch (err) {
        console.error(err);
      }
    }

    function renderResultLight(data, threshold) {
      const label = data.label;
      const pRed = data.p_red;

      if (label === 'no_traffic_light_found') {
        probNum.textContent = 'N/A (No fixture)';
        probBar.style.width = '0%';
        probBar.className = 'h-full bg-slate-400 transition-all duration-300 rounded-full';

        lampRed.className = 'w-16 h-16 rounded-full border bg-red-950/60 border-red-950 transition-all duration-300';
        lampYellow.className = 'w-16 h-16 rounded-full border bg-yellow-950/60 border-yellow-950 transition-all duration-300';
        lampGreen.className = 'w-16 h-16 rounded-full border bg-green-950/60 border-green-950 transition-all duration-300';

        resultBadge.className = 'px-6 py-2 rounded-full text-sm font-extrabold tracking-wide bg-amber-50 text-amber-800 border-2 border-amber-300 shadow-sm mb-6 transition-all text-center';
        resultBadge.textContent = 'NO TRAFFIC LIGHT FOUND';
        return;
      }

      const isRed = pRed >= threshold;
      probNum.textContent = `${(pRed * 100).toFixed(1)}% (${pRed.toFixed(4)})`;
      probBar.style.width = `${Math.min(100, pRed * 100)}%`;

      lampRed.className = 'w-16 h-16 rounded-full border transition-all duration-300 ' + (isRed ? 'bg-red-500 glow-red border-red-300' : 'bg-red-950/60 border-red-950');
      lampYellow.className = 'w-16 h-16 rounded-full border transition-all duration-300 ' + (!isRed && pRed > 0.15 ? 'bg-yellow-400 glow-yellow border-yellow-200' : 'bg-yellow-950/60 border-yellow-950');
      lampGreen.className = 'w-16 h-16 rounded-full border transition-all duration-300 ' + (!isRed && pRed <= 0.15 ? 'bg-emerald-500 glow-green border-emerald-300' : 'bg-green-950/60 border-green-950');

      if (isRed) {
        resultBadge.className = 'px-6 py-2 rounded-full text-base font-extrabold tracking-wide bg-red-50 text-red-700 border-2 border-red-400 shadow-md shadow-red-500/10 mb-6 transition-all text-center';
        resultBadge.textContent = 'RED LIGHT';
        probBar.className = 'h-full bg-red-500 transition-all duration-300 rounded-full';
      } else {
        resultBadge.className = 'px-6 py-2 rounded-full text-base font-extrabold tracking-wide bg-emerald-50 text-emerald-700 border-2 border-emerald-400 shadow-md shadow-emerald-500/10 mb-6 transition-all text-center';
        resultBadge.textContent = 'NOT RED';
        probBar.className = 'h-full bg-emerald-500 transition-all duration-300 rounded-full';
      }
    }

    function renderResultSign(data, threshold) {
      if (data.clahe_b64) {
        claheImg.src = data.clahe_b64;
        claheBox.classList.remove('hidden');
      } else {
        claheBox.classList.add('hidden');
      }

      const isConfident = data.confidence >= threshold;
      if (isConfident) {
        signWarning.classList.add('hidden');
        signResultBadge.className = 'px-5 py-2 rounded-full text-sm font-extrabold tracking-wide bg-red-50 text-red-700 border-2 border-red-400 shadow-sm mb-4 text-center max-w-full';
        signResultBadge.textContent = `${data.label} (${data.confidence.toFixed(1)}%)`;
        signSymbolCard.className = 'w-24 h-24 rounded-3xl bg-red-50 border-2 border-red-300 shadow-lg shadow-red-500/10 flex items-center justify-center text-4xl mb-4 transition-all scale-105';
      } else {
        signWarning.classList.remove('hidden');
        warnConfLimit.textContent = `${threshold}%`;
        signResultBadge.className = 'px-5 py-2 rounded-full text-sm font-extrabold tracking-wide bg-amber-50 text-amber-800 border-2 border-amber-300 shadow-sm mb-4 text-center max-w-full';
        signResultBadge.textContent = `Uncertain: ${data.label} (${data.confidence.toFixed(1)}%)`;
        signSymbolCard.className = 'w-24 h-24 rounded-3xl bg-amber-50 border-2 border-amber-300 shadow-md flex items-center justify-center text-4xl mb-4 transition-all';
      }

      top3List.innerHTML = '';
      (data.top3 || []).forEach((item, idx) => {
        const row = document.createElement('div');
        row.className = 'space-y-1';
        row.innerHTML = `
          <div class="flex justify-between items-center text-slate-700 font-semibold">
            <span class="truncate max-w-[210px]">${idx + 1}. ${item.label}</span>
            <span class="mono font-bold text-slate-900">${item.confidence.toFixed(1)}%</span>
          </div>
          <div class="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
            <div class="h-full ${idx === 0 ? 'bg-red-500' : 'bg-slate-400'} rounded-full transition-all duration-500" style="width: ${item.confidence}%"></div>
          </div>
        `;
        top3List.appendChild(row);
      });
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
        elif self.path == "/api/sign_metadata":
            labels = load_sign_labels()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(labels).encode("utf-8"))
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path == "/api/classify":
            self.handle_classify(parsed_url.query)
        elif parsed_url.path == "/api/classify_sign":
            self.handle_classify_sign(parsed_url.query)
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

    def handle_classify_sign(self, query_string: str) -> None:
        params = urllib.parse.parse_qs(query_string)
        filename = params.get("filename", ["uploaded_sign.jpg"])[0]
        conf_thresh = float(params.get("confidence_threshold", [75.0])[0])

        length = int(self.headers.get("Content-Length", 0))
        img_bytes = self.rfile.read(length)

        if not img_bytes:
            self.send_json_error("Empty image bytes received", HTTPStatus.BAD_REQUEST)
            return

        try:
            result = classify_sign(img_bytes, confidence_threshold=conf_thresh)
            result["filename"] = filename

            img = load_image(img_bytes)
            clahe_img = apply_clahe(img)
            buf = io.BytesIO()
            clahe_img.save(buf, format="JPEG", quality=90)
            clahe_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            result["clahe_b64"] = clahe_b64

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
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
