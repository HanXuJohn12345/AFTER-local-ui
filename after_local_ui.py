import argparse
import cgi
import contextlib
import json
import mimetypes
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


ROOT = Path(__file__).resolve().parent
PRETRAINED = ROOT / "pretrained"
MODEL_PATH = PRETRAINED / "afterv2.audio.instr.ts"
MAP_PATH = PRETRAINED / "afterv2.audio.instr.png"
OUTPUT_DIR = PRETRAINED / "ui_outputs"
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096
MAX_SECONDS = 12
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
_infer_lock = threading.Lock()
_live_model = None
_live_device = DEFAULT_DEVICE
_live_fx_state = None
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AFTER Local Audio Test</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111315;
      --panel: #181b1f;
      --line: #2b3036;
      --text: #ecf0f3;
      --muted: #9aa6b2;
      --accent: #37c6a3;
      --accent2: #f0c25a;
      --danger: #ef6b73;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      margin-top: 4px;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(340px, 1fr) minmax(240px, 300px);
      gap: 18px;
      align-items: start;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 16px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .map-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #0c0e10;
      cursor: crosshair;
      touch-action: none;
      user-select: none;
    }
    #map {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      user-select: none;
      -webkit-user-drag: none;
    }
    #dot {
      position: absolute;
      width: 14px;
      height: 14px;
      border: 2px solid white;
      border-radius: 50%;
      background: var(--accent);
      transform: translate(-50%, -50%);
      left: 50%;
      top: 50%;
      box-shadow: 0 0 0 2px rgba(0,0,0,.45);
      pointer-events: auto;
      cursor: grab;
      touch-action: none;
    }
    #dot.dragging {
      cursor: grabbing;
    }
    label {
      display: grid;
      gap: 6px;
      margin: 12px 0;
      color: var(--muted);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    input[type="file"] {
      width: 100%;
      color: var(--muted);
    }
    .row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .control-block {
      display: grid;
      gap: 8px;
      margin: 12px 0;
    }
    .label-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
    }
    .label-line strong {
      color: var(--text);
      white-space: nowrap;
    }
    .segmented {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
    }
    .segmented button.active,
    .preset-row button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #06231d;
    }
    .preset-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr)) auto;
      gap: 6px;
    }
    .dim-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 2px;
    }
    button {
      border: 1px solid var(--line);
      background: #20252a;
      color: var(--text);
      border-radius: 6px;
      padding: 9px 12px;
      min-height: 38px;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #06231d;
    }
    button.warn {
      background: var(--accent2);
      border-color: var(--accent2);
      color: #251a03;
    }
    button.danger {
      background: var(--danger);
      border-color: var(--danger);
      color: #270306;
    }
    button:disabled {
      opacity: .45;
      cursor: not-allowed;
    }
    .readout {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 15px;
      word-break: break-word;
    }
    audio {
      width: 100%;
      margin-top: 8px;
    }
    .status {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 42px;
      color: var(--muted);
      background: #131619;
    }
    .status.busy { color: var(--accent2); }
    .status.ok { color: var(--accent); }
    .status.err { color: var(--danger); }
    @media (max-width: 800px) {
      main { padding: 16px; }
      header { align-items: start; flex-direction: column; }
      .grid { grid-template-columns: 1fr; }
      .readout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>AFTER Local Audio Test</h1>
        <div class="sub">Audio input to AFTER audio-to-audio transfer, using the exported model as-is.</div>
      </div>
      <div class="sub">Model: afterv2.audio.instr.ts</div>
    </header>

    <div class="grid">
      <section>
        <h2>Timbre Dimensions</h2>
        <div class="readout">
          <div class="metric"><span>Mode</span><strong>6D latent</strong></div>
          <div class="metric"><span>Range</span><strong>-2.00 to 2.00</strong></div>
          <div class="metric"><span>Chunk</span><strong>4096</strong></div>
        </div>
        <div class="dim-grid">
          <label>Timbre Dim 1 <strong id="timbreValue0">0.00</strong><input class="timbre-slider" data-index="0" type="range" min="-2" max="2" step="0.01" value="0" /></label>
          <label>Timbre Dim 2 <strong id="timbreValue1">0.00</strong><input class="timbre-slider" data-index="1" type="range" min="-2" max="2" step="0.01" value="0" /></label>
          <label>Timbre Dim 3 <strong id="timbreValue2">0.00</strong><input class="timbre-slider" data-index="2" type="range" min="-2" max="2" step="0.01" value="0" /></label>
          <label>Timbre Dim 4 <strong id="timbreValue3">0.00</strong><input class="timbre-slider" data-index="3" type="range" min="-2" max="2" step="0.01" value="0" /></label>
          <label>Timbre Dim 5 <strong id="timbreValue4">0.00</strong><input class="timbre-slider" data-index="4" type="range" min="-2" max="2" step="0.01" value="0" /></label>
          <label>Timbre Dim 6 <strong id="timbreValue5">0.00</strong><input class="timbre-slider" data-index="5" type="range" min="-2" max="2" step="0.01" value="0" /></label>
        </div>
        <label>
          nb_steps <strong id="stepsValue">1</strong>
          <input id="steps" type="range" min="1" max="6" step="1" value="1" />
        </label>
        <label>
          guidance_structure <strong id="guidanceValue">1.00</strong>
          <input id="guidance" type="range" min="0" max="2" step="0.05" value="1" />
        </label>
        <label>
          Input Gain <strong id="gainValue">0 dB</strong>
          <input id="inputGain" type="range" min="-24" max="24" step="1" value="0" />
        </label>
        <label>
          Wet / Dry <strong id="wetValue">100%</strong>
          <input id="wetMix" type="range" min="0" max="1" step="0.01" value="1" />
        </label>

        <label>
          Morph Speed <strong id="morphValue">0.25s</strong>
          <input id="morphSpeed" type="range" min="0" max="2" step="0.05" value="0.25" />
        </label>
        <div class="control-block">
          <div class="label-line"><span>Live Quality</span><strong id="qualityValue">Fast</strong></div>
          <div class="segmented">
            <button id="qualityFast" type="button" class="active">Fast</button>
            <button id="qualityRich" type="button">Rich</button>
          </div>
        </div>
        <div class="control-block">
          <div class="label-line"><span>Timbre Presets</span><strong id="presetValue">A</strong></div>
          <div class="preset-row">
            <button type="button" class="preset-btn active" data-preset="0">A</button>
            <button type="button" class="preset-btn" data-preset="1">B</button>
            <button type="button" class="preset-btn" data-preset="2">C</button>
            <button type="button" class="preset-btn" data-preset="3">D</button>
            <button id="savePresetBtn" type="button">Save</button>
          </div>
        </div>
      </section>

      <section>
        <h2>Input</h2>
        <div class="row">
          <button id="recordBtn" class="primary">Start Mic</button>
          <button id="stopBtn" class="danger" disabled>Stop</button>
          <button id="clearBtn">Clear</button>
          <span id="timer" class="sub">00:00.0</span>
        </div>
        <label>
          Upload audio instead
          <input id="fileInput" type="file" accept="audio/*,.wav,.flac,.mp3,.aac,.opus" />
        </label>
        <audio id="inputAudio" controls></audio>
        <div class="row" style="margin-top: 14px;">
          <button id="runBtn" class="warn" disabled>Run AFTER</button>
          <button id="liveStartBtn" class="primary">Start Live</button>
          <button id="liveStopBtn" class="danger" disabled>Stop Live</button>
        </div>
        <div id="status" class="status" style="margin-top: 14px;">Record from the microphone or choose an audio file.</div>
      </section>

      <section>
        <h2>后处理</h2>
        <label>
          Spring Mix <strong id="springValue">0%</strong>
          <input id="springMix" type="range" min="0" max="1" step="0.01" value="0" />
        </label>
        <label>
          Spring Decay <strong id="springDecayValue">0.85</strong>
          <input id="springDecay" type="range" min="0" max="1" step="0.01" value="0.85" />
        </label>
        <label>
          Reverb Boost <strong id="reverbBoostValue">1.0x</strong>
          <input id="reverbBoost" type="range" min="1" max="8" step="0.1" value="1" />
        </label>
        <label>
          Delay Mix <strong id="delayValue">0%</strong>
          <input id="delayMix" type="range" min="0" max="1" step="0.01" value="0" />
        </label>
        <label>
          Delay Time <strong id="delayTimeValue">320 ms</strong>
          <input id="delayTime" type="range" min="40" max="1200" step="10" value="320" />
        </label>
        <label>
          Delay Feedback <strong id="delayFeedbackValue">35%</strong>
          <input id="delayFeedback" type="range" min="0" max="0.92" step="0.01" value="0.35" />
        </label>
      </section>
    </div>

    <section>
      <h2>Output</h2>
      <audio id="outputAudio" controls></audio>
      <div class="readout">
        <div class="metric"><span>Source seconds</span><strong id="sourceSeconds">-</strong></div>
        <div class="metric"><span>Processed chunks</span><strong id="chunks">-</strong></div>
        <div class="metric"><span>Elapsed</span><strong id="elapsed">-</strong></div>
        <div class="metric"><span>Live latency</span><strong id="liveLatency">-</strong></div>
      </div>
    </section>
  </main>

  <script>
    const timbreSliders = Array.from(document.querySelectorAll(".timbre-slider"));
    const timbreValueEls = timbreSliders.map((_, index) => document.getElementById(`timbreValue${index}`));
    const steps = document.getElementById("steps");
    const guidance = document.getElementById("guidance");
    const inputGain = document.getElementById("inputGain");
    const wetMix = document.getElementById("wetMix");
    const springMix = document.getElementById("springMix");
    const springDecay = document.getElementById("springDecay");
    const reverbBoost = document.getElementById("reverbBoost");
    const delayMix = document.getElementById("delayMix");
    const delayTime = document.getElementById("delayTime");
    const delayFeedback = document.getElementById("delayFeedback");
    const morphSpeed = document.getElementById("morphSpeed");
    const qualityFast = document.getElementById("qualityFast");
    const qualityRich = document.getElementById("qualityRich");
    const savePresetBtn = document.getElementById("savePresetBtn");
    const presetButtons = Array.from(document.querySelectorAll(".preset-btn"));
    const stepsValue = document.getElementById("stepsValue");
    const guidanceValue = document.getElementById("guidanceValue");
    const gainValue = document.getElementById("gainValue");
    const wetValue = document.getElementById("wetValue");
    const springValue = document.getElementById("springValue");
    const springDecayValue = document.getElementById("springDecayValue");
    const reverbBoostValue = document.getElementById("reverbBoostValue");
    const delayValue = document.getElementById("delayValue");
    const delayTimeValue = document.getElementById("delayTimeValue");
    const delayFeedbackValue = document.getElementById("delayFeedbackValue");
    const morphValue = document.getElementById("morphValue");
    const qualityValue = document.getElementById("qualityValue");
    const presetValue = document.getElementById("presetValue");
    const recordBtn = document.getElementById("recordBtn");
    const stopBtn = document.getElementById("stopBtn");
    const clearBtn = document.getElementById("clearBtn");
    const runBtn = document.getElementById("runBtn");
    const liveStartBtn = document.getElementById("liveStartBtn");
    const liveStopBtn = document.getElementById("liveStopBtn");
    const fileInput = document.getElementById("fileInput");
    const inputAudio = document.getElementById("inputAudio");
    const outputAudio = document.getElementById("outputAudio");
    const statusBox = document.getElementById("status");
    const timer = document.getElementById("timer");
    const sourceSeconds = document.getElementById("sourceSeconds");
    const chunks = document.getElementById("chunks");
    const elapsed = document.getElementById("elapsed");
    const liveLatency = document.getElementById("liveLatency");

    let timbreValues = Array(6).fill(0);
    let targetTimbreValues = Array(6).fill(0);
    let audioBlob = null;
    let audioName = "mic.wav";
    let recording = false;
    let audioContext = null;
    let processor = null;
    let mediaStream = null;
    let recordedChunks = [];
    let recordStart = 0;
    let timerId = null;
    let liveMode = false;
    let liveStream = null;
    let liveProcessor = null;
    let liveContext = null;
    let liveQueue = [];
    let liveBusy = false;
    let liveScheduleTime = 0;
    let liveProcessedChunks = 0;
    let liveDroppedChunks = 0;
    let liveDevice = "unknown";
    let activePreset = 0;
    let timbrePresets = JSON.parse(localStorage.getItem("afterTimbrePresets6D") || localStorage.getItem("afterTimbrePresets") || "[null,null,null,null]");
    let lastMorphFrame = performance.now();

    function setStatus(text, mode) {
      statusBox.textContent = text;
      statusBox.className = "status" + (mode ? " " + mode : "");
    }

    function updateControls() {
      runBtn.disabled = !audioBlob || recording || liveMode;
      recordBtn.disabled = recording || liveMode;
      stopBtn.disabled = !recording;
      liveStartBtn.disabled = liveMode || recording;
      liveStopBtn.disabled = !liveMode;
    }

    function updateTimer() {
      const seconds = (performance.now() - recordStart) / 1000;
      const m = Math.floor(seconds / 60).toString().padStart(2, "0");
      const s = (seconds % 60).toFixed(1).padStart(4, "0");
      timer.textContent = `${m}:${s}`;
    }

    function clampTimbreValue(value) {
      return Math.min(Math.max(Number(value) || 0, -2), 2);
    }

    function setTimbreTarget(index, value, instant = false) {
      const v = clampTimbreValue(value);
      targetTimbreValues[index] = v;
      timbreSliders[index].value = v.toFixed(2);
      timbreValueEls[index].textContent = v.toFixed(2);
      if (instant || Number(morphSpeed.value) <= 0) {
        timbreValues[index] = v;
      }
    }

    function setTimbreTargetAll(values, instant = false) {
      const next = Array(6).fill(0).map((_, index) => clampTimbreValue(values[index] ?? 0));
      next.forEach((value, index) => setTimbreTarget(index, value, instant));
    }

    function currentTimbreString() {
      return timbreValues.map((value) => value.toFixed(4)).join(",");
    }

    function tickMorph(now) {
      const dt = Math.min(0.1, Math.max(0, (now - lastMorphFrame) / 1000));
      lastMorphFrame = now;
      const seconds = Number(morphSpeed.value);
      for (let i = 0; i < timbreValues.length; i++) {
        if (seconds <= 0) {
          timbreValues[i] = targetTimbreValues[i];
        } else {
          const alpha = Math.min(1, dt / seconds);
          timbreValues[i] += (targetTimbreValues[i] - timbreValues[i]) * alpha;
        }
      }
      window.requestAnimationFrame(tickMorph);
    }

    function updateQualityUi() {
      const value = Number(steps.value);
      qualityFast.classList.toggle("active", value === 1);
      qualityRich.classList.toggle("active", value === 2);
      qualityValue.textContent = value === 1 ? "Fast" : value === 2 ? "Rich" : `Steps ${value}`;
    }

    function setStepsValue(value) {
      steps.value = String(value);
      stepsValue.textContent = steps.value;
      updateQualityUi();
    }

    function updatePresetUi() {
      presetValue.textContent = String.fromCharCode(65 + activePreset);
      presetButtons.forEach((button, index) => {
        button.classList.toggle("active", index === activePreset);
        button.classList.toggle("warn", Boolean(timbrePresets[index]) && index !== activePreset);
      });
    }

    function saveActivePreset() {
      timbrePresets[activePreset] = targetTimbreValues.slice();
      localStorage.setItem("afterTimbrePresets6D", JSON.stringify(timbrePresets));
      updatePresetUi();
      setStatus(`Preset ${String.fromCharCode(65 + activePreset)} saved.`, "ok");
    }

    function loadPreset(index) {
      activePreset = index;
      const preset = timbrePresets[index];
      if (Array.isArray(preset)) {
        setTimbreTargetAll(preset);
      }
      updatePresetUi();
    }

    timbreSliders.forEach((slider, index) => {
      slider.addEventListener("input", () => setTimbreTarget(index, slider.value));
    });
    setTimbreTargetAll(targetTimbreValues, true);
    steps.addEventListener("input", () => setStepsValue(steps.value));
    guidance.addEventListener("input", () => guidanceValue.textContent = Number(guidance.value).toFixed(2));
    inputGain.addEventListener("input", () => gainValue.textContent = `${Number(inputGain.value)} dB`);
    wetMix.addEventListener("input", () => wetValue.textContent = `${Math.round(Number(wetMix.value) * 100)}%`);
    springMix.addEventListener("input", () => springValue.textContent = `${Math.round(Number(springMix.value) * 100)}%`);
    springDecay.addEventListener("input", () => springDecayValue.textContent = Number(springDecay.value).toFixed(2));
    reverbBoost.addEventListener("input", () => reverbBoostValue.textContent = `${Number(reverbBoost.value).toFixed(1)}x`);
    delayMix.addEventListener("input", () => delayValue.textContent = `${Math.round(Number(delayMix.value) * 100)}%`);
    delayTime.addEventListener("input", () => delayTimeValue.textContent = `${Math.round(Number(delayTime.value))} ms`);
    delayFeedback.addEventListener("input", () => delayFeedbackValue.textContent = `${Math.round(Number(delayFeedback.value) * 100)}%`);
    morphSpeed.addEventListener("input", () => morphValue.textContent = `${Number(morphSpeed.value).toFixed(2)}s`);
    qualityFast.addEventListener("click", () => setStepsValue(1));
    qualityRich.addEventListener("click", () => setStepsValue(2));
    savePresetBtn.addEventListener("click", saveActivePreset);
    presetButtons.forEach((button) => {
      button.addEventListener("click", () => loadPreset(Number(button.dataset.preset)));
    });
    updateQualityUi();
    updatePresetUi();
    window.requestAnimationFrame(tickMorph);

    function encodeWav(buffers, sampleRate) {
      const length = buffers.reduce((sum, b) => sum + b.length, 0);
      const pcm = new Float32Array(length);
      let offset = 0;
      for (const b of buffers) {
        pcm.set(b, offset);
        offset += b.length;
      }
      const bytes = new ArrayBuffer(44 + pcm.length * 2);
      const view = new DataView(bytes);
      const writeString = (pos, text) => {
        for (let i = 0; i < text.length; i++) view.setUint8(pos + i, text.charCodeAt(i));
      };
      writeString(0, "RIFF");
      view.setUint32(4, 36 + pcm.length * 2, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, "data");
      view.setUint32(40, pcm.length * 2, true);
      let p = 44;
      for (let i = 0; i < pcm.length; i++, p += 2) {
        const s = Math.max(-1, Math.min(1, pcm[i]));
        view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
      return new Blob([view], { type: "audio/wav" });
    }

    async function startRecording() {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
      audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(mediaStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);
      const mute = audioContext.createGain();
      mute.gain.value = 0;
      recordedChunks = [];
      processor.onaudioprocess = (event) => {
        if (!recording) return;
        recordedChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(mute);
      mute.connect(audioContext.destination);
      recording = true;
      recordStart = performance.now();
      timerId = window.setInterval(updateTimer, 100);
      setStatus("Recording microphone input.", "busy");
      updateControls();
    }

    async function stopRecording() {
      recording = false;
      window.clearInterval(timerId);
      if (processor) processor.disconnect();
      if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
      if (audioContext) await audioContext.close();
      audioBlob = encodeWav(recordedChunks, audioContext ? audioContext.sampleRate : 44100);
      audioName = "mic.wav";
      inputAudio.src = URL.createObjectURL(audioBlob);
      setStatus("Microphone take ready.", "ok");
      updateControls();
    }


    function scheduleLiveOutput(samples, sampleRate) {
      if (!liveContext || samples.length === 0) return;
      const buffer = liveContext.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const src = liveContext.createBufferSource();
      src.buffer = buffer;
      src.connect(liveContext.destination);
      const now = liveContext.currentTime;
      if (liveScheduleTime < now + 0.12) liveScheduleTime = now + 0.12;
      src.start(liveScheduleTime);
      liveScheduleTime += samples.length / sampleRate;
      liveProcessedChunks += 1;
      liveLatency.textContent = `${Math.max(0, liveScheduleTime - now).toFixed(2)}s`;
      chunks.textContent = `${liveProcessedChunks} live`;
    }

    async function pumpLiveQueue() {
      if (!liveMode || liveBusy || liveQueue.length === 0) return;
      liveBusy = true;
      const block = liveQueue.shift();
      const query = new URLSearchParams({
        zt: currentTimbreString(),
        nb_steps: steps.value,
        guidance_structure: guidance.value,
        input_gain_db: inputGain.value,
        wet_mix: wetMix.value,
        spring_mix: springMix.value,
        spring_decay: springDecay.value,
        reverb_boost: reverbBoost.value,
        delay_mix: delayMix.value,
        delay_time_ms: delayTime.value,
        delay_feedback: delayFeedback.value,
        sr: liveContext.sampleRate.toString()
      });
      try {
        const body = block.buffer.slice(block.byteOffset, block.byteOffset + block.byteLength);
        const response = await fetch(`/api/live_chunk?${query.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body
        });
        if (!response.ok) {
          let msg = "Live chunk failed.";
          try { msg = (await response.json()).error || msg; } catch (_) {}
          throw new Error(msg);
        }
        const sampleRate = Number(response.headers.get("X-Sample-Rate") || "44100");
        const arr = new Float32Array(await response.arrayBuffer());
        scheduleLiveOutput(arr, sampleRate);
        const elapsed = Number(response.headers.get("X-Elapsed-Seconds") || "0");
        setStatus(`Live running on ${liveDevice}. chunk ${liveProcessedChunks}, model ${elapsed.toFixed(2)}s, dropped ${liveDroppedChunks}.`, "busy");
      } catch (err) {
        setStatus(err.message, "err");
      } finally {
        liveBusy = false;
        if (liveMode) pumpLiveQueue();
      }
    }

    async function startLive() {
      liveStartBtn.disabled = true;
      setStatus("Starting microphone and loading live model.", "busy");
      liveStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
      liveContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      await liveContext.resume();
      const resetQuery = new URLSearchParams({
        nb_steps: steps.value,
        guidance_structure: guidance.value
      });
      const reset = await fetch(`/api/live_reset?${resetQuery.toString()}`, { method: "POST" });
      const resetData = await reset.json().catch(() => ({}));
      if (!reset.ok) throw new Error(resetData.error || "Could not reset live model.");
      liveDevice = resetData.device_name ? `${resetData.device} / ${resetData.device_name}` : (resetData.device || "unknown");

      const source = liveContext.createMediaStreamSource(liveStream);
      liveProcessor = liveContext.createScriptProcessor(4096, 1, 1);
      const mute = liveContext.createGain();
      mute.gain.value = 0;
      liveQueue = [];
      liveBusy = false;
      liveProcessedChunks = 0;
      liveDroppedChunks = 0;
      liveScheduleTime = liveContext.currentTime + 0.35;
      liveLatency.textContent = "warming";
      liveMode = true;
      liveProcessor.onaudioprocess = (event) => {
        if (!liveMode) return;
        liveQueue.push(new Float32Array(event.inputBuffer.getChannelData(0)));
        while (liveQueue.length > 6) {
          liveQueue.shift();
          liveDroppedChunks += 1;
        }
        pumpLiveQueue();
      };
      source.connect(liveProcessor);
      liveProcessor.connect(mute);
      mute.connect(liveContext.destination);
      setStatus(`Live running on ${liveDevice}. Move timbre dimension faders to affect upcoming chunks.`, "busy");
      updateControls();
    }

    async function stopLive() {
      liveMode = false;
      liveQueue = [];
      if (liveProcessor) liveProcessor.disconnect();
      if (liveStream) liveStream.getTracks().forEach(t => t.stop());
      if (liveContext) await liveContext.close();
      liveProcessor = null;
      liveStream = null;
      liveContext = null;
      liveBusy = false;
      liveLatency.textContent = "-";
      setStatus("Live stopped.", "ok");
      updateControls();
    }

    recordBtn.addEventListener("click", () => {
      startRecording().catch(err => setStatus(err.message, "err"));
    });
    stopBtn.addEventListener("click", () => {
      stopRecording().catch(err => setStatus(err.message, "err"));
    });
    clearBtn.addEventListener("click", () => {
      audioBlob = null;
      audioName = "mic.wav";
      inputAudio.removeAttribute("src");
      outputAudio.removeAttribute("src");
      fileInput.value = "";
      sourceSeconds.textContent = "-";
      chunks.textContent = "-";
      elapsed.textContent = "-";
      liveLatency.textContent = "-";
      timer.textContent = "00:00.0";
      setStatus("Record from the microphone or choose an audio file.");
      updateControls();
    });
    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      audioBlob = file;
      audioName = file.name || "input.wav";
      inputAudio.src = URL.createObjectURL(file);
      setStatus("Audio file ready.", "ok");
      updateControls();
    });

    runBtn.addEventListener("click", async () => {
      if (!audioBlob) return;
      const form = new FormData();
      form.append("audio", audioBlob, audioName);
      form.append("zt", targetTimbreValues.map((value) => value.toFixed(4)).join(","));
      form.append("nb_steps", steps.value);
      form.append("guidance_structure", guidance.value);
      form.append("input_gain_db", inputGain.value);
      form.append("wet_mix", wetMix.value);
      form.append("spring_mix", springMix.value);
      form.append("spring_decay", springDecay.value);
      form.append("reverb_boost", reverbBoost.value);
      form.append("delay_mix", delayMix.value);
      form.append("delay_time_ms", delayTime.value);
      form.append("delay_feedback", delayFeedback.value);
      runBtn.disabled = true;
      setStatus("Running AFTER inference. Short takes are friendlier for this exported model.", "busy");
      const started = performance.now();
      try {
        const response = await fetch("/api/generate", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Generation failed.");
        outputAudio.src = data.output_url + "?t=" + Date.now();
        sourceSeconds.textContent = data.source_seconds.toFixed(2);
        chunks.textContent = data.chunks;
        elapsed.textContent = data.elapsed_seconds.toFixed(2) + "s";
        setStatus(`Done on ${data.device || "unknown"}${data.device_name ? " / " + data.device_name : ""}.`, "ok");
      } catch (err) {
        setStatus(err.message, "err");
      } finally {
        runBtn.disabled = !audioBlob;
      }
    });

    liveStartBtn.addEventListener("click", () => {
      startLive().catch(err => {
        setStatus(err.message, "err");
        liveMode = false;
        updateControls();
      });
    });
    liveStopBtn.addEventListener("click", () => {
      stopLive().catch(err => setStatus(err.message, "err"));
    });
    updateControls();
  </script>
</body>
</html>
"""


def _json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _serve_file(handler, path: Path, content_type=None):
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _load_audio(path: Path):
    waveform, sr = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError("The input audio file is empty.")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    max_len = SAMPLE_RATE * MAX_SECONDS
    waveform = waveform[:, :max_len]
    waveform = waveform.clamp(-1.0, 1.0)
    return waveform



def _device_info(device: str):
    info = {"device": device, "cuda_available": torch.cuda.is_available()}
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(index)
    else:
        info["device_name"] = "CPU"
    return info


def _load_script_model(device: str):
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    if device != "cpu":
        model = model.to(device)
    return model


def _script_children(module):
    return list(module.children())


def _normalize_complex_py(time_transform, spec):
    mag = float(time_transform.beta_rescale) * (torch.abs(spec) + 1e-8) ** float(time_transform.alpha_rescale)
    phase = torch.angle(spec)
    unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    return mag.to(unit_phase.dtype) * unit_phase


def _denormalize_complex_py(time_transform, spec):
    scaled = spec / float(time_transform.beta_rescale)
    mag = torch.abs(scaled) ** (1.0 / float(time_transform.alpha_rescale))
    phase = torch.angle(scaled)
    unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    return mag.to(unit_phase.dtype) * unit_phase


def _time_forward_py(time_transform, audio):
    if bool(time_transform.stream):
        buffer_len = int(time_transform.nfft) - int(time_transform.hop_size)
        if time_transform.audio_buffer.shape[0] != audio.shape[0]:
            time_transform.audio_buffer = torch.zeros(
                (audio.shape[0], 1, buffer_len), device=audio.device, dtype=audio.dtype
            )
        audio = torch.cat([
            time_transform.audio_buffer.to(device=audio.device, dtype=audio.dtype),
            audio,
        ], dim=-1)
        time_transform.audio_buffer = audio[..., -buffer_len:].detach().clone()

    spec = time_transform.transform.forward(audio)
    if not bool(time_transform.stream):
        spec = spec[..., :-1]
    if bool(time_transform.normalize):
        spec = _normalize_complex_py(time_transform, spec)

    skip = int(time_transform.skip_features)
    if skip > 0:
        spec = spec[:, :, skip:]
    elif skip < 0:
        spec = spec[:, :, :skip]
    return torch.cat((torch.real(spec), torch.imag(spec)), -3)


def _time_inverse_stream_py(time_transform, spec):
    n = spec.shape[0]
    if not bool(time_transform.stream):
        spec = torch.cat((spec, torch.zeros_like(spec)[..., :1]), -1)
    else:
        spec = torch.cat((
            time_transform.spec_buffer[:n].to(device=spec.device, dtype=spec.dtype),
            spec,
        ), -1)
        time_transform.spec_buffer[:n] = spec[..., -int(time_transform.n_fade):].detach().clone()

    real, imag = torch.chunk(spec, 2, -3)
    spec_c = torch.complex(real.squeeze(-3), imag.squeeze(-3))
    if bool(time_transform.normalize):
        spec_c = _denormalize_complex_py(time_transform, spec_c)

    spec_c = spec_c.unsqueeze(1)
    skip = int(time_transform.skip_features)
    if skip > 0:
        spec_c = torch.cat((torch.zeros_like(spec_c)[:, :, :skip], spec_c), -2)
    elif skip < 0:
        spec_c = torch.cat((spec_c, torch.zeros_like(spec_c)[:, :, :abs(skip)]), -2)
    spec_c = spec_c.squeeze(1)

    audio = time_transform.inverse_transform.forward(spec_c).unsqueeze(1)
    if bool(time_transform.stream):
        fade_len = int(time_transform.hop_size) * int(time_transform.n_fade)
        alpha = torch.linspace(0, 1, fade_len, device=audio.device, dtype=audio.dtype)[None, None, :]
        audio[..., :fade_len] = (
            (1 - alpha) * time_transform.out_buffer[:n].to(device=audio.device, dtype=audio.dtype)
            + alpha * audio[..., :fade_len]
        )
        time_transform.out_buffer[:n] = audio[..., -fade_len:].detach().clone()
        audio = audio[..., :-fade_len]
    return audio


def _encode_stream_py(ae_model, audio):
    audio = _time_forward_py(ae_model.time_transform, audio)
    audio = ae_model.preconv.forward(audio)
    for layer in _script_children(ae_model.down_layers):
        audio = layer.forward(audio)
    audio = ae_model.rearrange_encode.forward(audio)
    audio = ae_model.middle_block_encode.forward(audio)
    return ae_model.bottleneck.forward_stream(audio)


def _decode_stream_py(ae_model, latent):
    audio = ae_model.middle_block_decode.forward(latent)
    audio = ae_model.rearrange_decode.forward(audio)
    for layer in _script_children(ae_model.up_layers):
        audio = layer.forward(audio)
    audio = ae_model.outconv.forward(audio)
    return _time_inverse_stream_py(ae_model.time_transform, audio)


def _sample_cuda(model, noise, cond, time_cond, nb_steps: int):
    dt = 1.0 / float(nb_steps)
    t_values = torch.linspace(0, 1, nb_steps + 1, device=noise.device, dtype=noise.dtype)[:-1]
    latent = noise
    for i, t_value in enumerate(t_values):
        t = t_value.repeat(latent.shape[0], 1, latent.shape[-1])
        latent = latent + dt * model.model_forward(latent, t, cond, time_cond, int(i))
        model.net.roll_cache(latent.shape[-1], int(i))
    return latent


def _prepare_audio_batch(waveform, device: str):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    return waveform.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)


def _db_to_gain(db_value: float):
    return float(10 ** (float(db_value) / 20.0))


def _mix_wet_dry(generated, dry, wet_mix: float):
    wet = _clamp_float(wet_mix, 0.0, 1.0)
    dry = dry.to(generated.device, dtype=generated.dtype)
    return generated * wet + dry * (1.0 - wet)


def _clamp_float(value, lo: float, hi: float):
    return max(lo, min(hi, float(value)))


def _make_fx_state(sample_rate: int):
    comb_ms = [21.0, 31.0, 43.0, 59.0, 73.0, 89.0, 109.0, 137.0]
    return {
        "sample_rate": int(sample_rate),
        "spring_buffers": [
            np.zeros(max(1, int(sample_rate * ms / 1000.0)), dtype=np.float32)
            for ms in comb_ms
        ],
        "spring_pos": [0 for _ in comb_ms],
        "spring_ap_buffer": np.zeros(max(1, int(sample_rate * 0.014)), dtype=np.float32),
        "spring_ap_pos": 0,
        "delay_buffer": np.zeros(max(1, int(sample_rate * 2.5)), dtype=np.float32),
        "delay_pos": 0,
    }


def _apply_spring_reverb_np(samples, state, mix: float, decay: float, reverb_boost: float = 1.0):
    mix = _clamp_float(mix, 0.0, 1.0)
    if mix <= 0.0:
        return samples

    decay = _clamp_float(decay, 0.0, 1.0)
    boost = _clamp_float(reverb_boost, 1.0, 8.0)
    feedback_base = min(0.98, 0.48 + decay * 0.47 + (boost - 1.0) * 0.015)
    wet_gain = (2.2 + decay * 4.8) * boost
    wet_level = 1.0 + (boost - 1.0) * 0.18
    buffers = state["spring_buffers"]
    positions = state["spring_pos"]
    ap_buffer = state["spring_ap_buffer"]
    ap_pos = int(state["spring_ap_pos"])
    out = np.empty_like(samples, dtype=np.float32)
    g = 0.62

    for n, value in enumerate(samples):
        dry_value = float(value)
        acc = 0.0
        for i, buffer in enumerate(buffers):
            pos = positions[i]
            delayed = float(buffer[pos])
            feedback = min(0.95, feedback_base * (0.95 - i * 0.025))
            buffer[pos] = np.float32(dry_value + delayed * feedback)
            positions[i] = (pos + 1) % len(buffer)
            acc += delayed

        wet = acc / max(1.0, float(len(buffers)) * 0.55)
        ap_delayed = float(ap_buffer[ap_pos])
        ap_out = -g * wet + ap_delayed
        ap_buffer[ap_pos] = np.float32(wet + g * ap_out)
        ap_pos = (ap_pos + 1) % len(ap_buffer)
        wet_value = np.tanh(ap_out * wet_gain) * wet_level
        out[n] = np.float32(dry_value * (1.0 - mix) + wet_value * mix)

    state["spring_ap_pos"] = ap_pos
    return out


def _apply_delay_np(samples, state, mix: float, time_ms: float, feedback: float):
    mix = _clamp_float(mix, 0.0, 1.0)
    if mix <= 0.0:
        return samples

    sample_rate = int(state["sample_rate"])
    buffer = state["delay_buffer"]
    write_pos = int(state["delay_pos"])
    delay_samples = int(sample_rate * _clamp_float(time_ms, 1.0, 2000.0) / 1000.0)
    delay_samples = max(1, min(delay_samples, len(buffer) - 1))
    feedback = _clamp_float(feedback, 0.0, 0.95)
    out = np.empty_like(samples, dtype=np.float32)

    for n, value in enumerate(samples):
        dry_value = float(value)
        read_pos = (write_pos - delay_samples) % len(buffer)
        delayed = float(buffer[read_pos])
        buffer[write_pos] = np.float32(dry_value + delayed * feedback)
        out[n] = np.float32(dry_value * (1.0 - mix) + delayed * mix)
        write_pos = (write_pos + 1) % len(buffer)

    state["delay_pos"] = write_pos
    return out


def _apply_post_fx(audio, sample_rate: int, spring_mix: float, spring_decay: float, reverb_boost: float, delay_mix: float, delay_time_ms: float, delay_feedback: float, fx_state=None):
    original_dim = audio.dim()
    original_device = audio.device
    original_dtype = audio.dtype
    samples = audio.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=True)

    if fx_state is None or int(fx_state.get("sample_rate", 0)) != int(sample_rate):
        fx_state = _make_fx_state(sample_rate)

    samples = _apply_spring_reverb_np(samples, fx_state, spring_mix, spring_decay, reverb_boost)
    samples = _apply_delay_np(samples, fx_state, delay_mix, delay_time_ms, delay_feedback)
    samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
    output = torch.from_numpy(samples).to(device=original_device, dtype=original_dtype)
    if original_dim == 2:
        output = output.unsqueeze(0)
    return output, fx_state


def _parse_timbre_values(raw):
    if raw is None:
        return [0.0] * 6
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    values = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(max(-4.0, min(4.0, float(part))))
    return (values + [0.0] * 6)[:6]


def _timbre_tensor(model, timbre_values, device: str, frames=None):
    channels = int(model.zt_channels)
    values = (_parse_timbre_values(timbre_values) + [0.0] * channels)[:channels]
    zt = torch.tensor([values], dtype=torch.float32, device=device)
    if frames is None:
        return zt
    return zt.unsqueeze(-1).repeat(1, 1, int(frames))


def _generate_timbre(model, waveform, timbre_values, nb_steps: int, guidance: float, device: str):
    model.set_nb_steps(int(nb_steps))
    model.set_guidance_structure(float(guidance))
    audio = _prepare_audio_batch(waveform, device)
    zt = _timbre_tensor(model, timbre_values, device)

    if device.startswith("cuda"):
        zsem = zt * model.latent_range
        z_structure = _encode_stream_py(model.emb_model_structure.model, audio)
        time_cond = model.encoder_time.forward_stream(z_structure).repeat(audio.shape[0], 1, 1)
        noise = torch.randn(
            audio.shape[0], model.ae_latents, time_cond.shape[-1],
            device=audio.device, dtype=audio.dtype
        )
        z = _sample_cuda(model, noise, zsem, time_cond, int(nb_steps))
        return _decode_stream_py(model.emb_model_structure.model, z).cpu().squeeze(0)

    latent = _timbre_tensor(model, timbre_values, device, frames=audio.shape[-1])
    return model.generate_timbre(torch.cat([audio, latent], dim=1)).cpu().squeeze(0)


def _run_after(input_path: Path, timbre_values, nb_steps: int, guidance: float, input_gain_db: float = 0.0, wet_mix: float = 1.0, spring_mix: float = 0.0, spring_decay: float = 0.7, reverb_boost: float = 1.0, delay_mix: float = 0.0, delay_time_ms: float = 320.0, delay_feedback: float = 0.35):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    waveform = _load_audio(input_path)
    source_len = waveform.shape[-1]
    if source_len == 0:
        raise ValueError("No usable audio samples found.")

    device = DEFAULT_DEVICE
    with _infer_lock, torch.inference_mode():
        model = _load_script_model(device)
        chunks = []
        for start in range(0, source_len, CHUNK_SIZE):
            chunk = waveform[:, start:start + CHUNK_SIZE]
            valid = chunk.shape[-1]
            if valid < CHUNK_SIZE:
                chunk = F.pad(chunk, (0, CHUNK_SIZE - valid))
            dry = chunk.clone()
            model_chunk = (chunk * _db_to_gain(input_gain_db)).clamp(-1.0, 1.0)
            generated = _generate_timbre(model, model_chunk, timbre_values, nb_steps, guidance, device)
            mixed = _mix_wet_dry(generated, dry, wet_mix)
            chunks.append(mixed[:, :valid])

    output = torch.cat(chunks, dim=-1).clamp(-1.0, 1.0)
    output, _ = _apply_post_fx(output, SAMPLE_RATE, spring_mix, spring_decay, reverb_boost, delay_mix, delay_time_ms, delay_feedback, None)
    output = output.clamp(-1.0, 1.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"after_{uuid.uuid4().hex}.wav"
    out_path = OUTPUT_DIR / out_name
    torchaudio.save(str(out_path), output, SAMPLE_RATE)
    return out_name, source_len / SAMPLE_RATE, len(chunks), _device_info(device)

def _reset_live_model(nb_steps: int = 1, guidance: float = 1.0):
    global _live_model, _live_device, _live_fx_state
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = DEFAULT_DEVICE
    with _infer_lock, torch.inference_mode():
        _live_device = device
        _live_model = _load_script_model(device)
        dummy_audio = torch.zeros(1, CHUNK_SIZE)
        _generate_timbre(_live_model, dummy_audio, [0.0] * 6, nb_steps, guidance, device)
        _generate_timbre(_live_model, dummy_audio, [0.0] * 6, nb_steps, guidance, device)
        _live_fx_state = _make_fx_state(SAMPLE_RATE)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return _device_info(device)


def _get_live_model_locked():
    global _live_model, _live_device
    if _live_model is None:
        _live_device = DEFAULT_DEVICE
        _live_model = _load_script_model(_live_device)
    return _live_model


def _process_live_chunk(raw: bytes, sr: int, timbre_values, nb_steps: int, guidance: float, input_gain_db: float = 0.0, wet_mix: float = 1.0, spring_mix: float = 0.0, spring_decay: float = 0.7, reverb_boost: float = 1.0, delay_mix: float = 0.0, delay_time_ms: float = 320.0, delay_feedback: float = 0.35):
    if not raw:
        raise ValueError("Empty live audio chunk.")
    samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if samples.size == 0:
        raise ValueError("No Float32 samples in live audio chunk.")

    waveform = torch.from_numpy(samples).unsqueeze(0).float().clamp(-1.0, 1.0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[-1] < CHUNK_SIZE:
        waveform = F.pad(waveform, (0, CHUNK_SIZE - waveform.shape[-1]))
    waveform = waveform[:, :CHUNK_SIZE]

    dry = waveform.clone().squeeze(0)
    model_waveform = (waveform * _db_to_gain(input_gain_db)).clamp(-1.0, 1.0)

    global _live_fx_state
    with _infer_lock, torch.inference_mode():
        model = _get_live_model_locked()
        generated = _generate_timbre(model, model_waveform, timbre_values, nb_steps, guidance, _live_device).squeeze(0)
        output = _mix_wet_dry(generated, dry, wet_mix)
        output, _live_fx_state = _apply_post_fx(output, SAMPLE_RATE, spring_mix, spring_decay, reverb_boost, delay_mix, delay_time_ms, delay_feedback, _live_fx_state)

    output = output.clamp(-1.0, 1.0).numpy().astype("<f4", copy=False)
    return output.tobytes(), int(output.shape[0]), _device_info(_live_device)

class Handler(BaseHTTPRequestHandler):
    server_version = "AFTERLocalUI/0.1"

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/map.png":
            _serve_file(self, MAP_PATH, "image/png")
            return
        if path.startswith("/outputs/"):
            name = Path(path.split("/outputs/", 1)[1]).name
            _serve_file(self, OUTPUT_DIR / name, "audio/wav")
            return
        if path == "/health":
            _json(self, 200, {"ok": True, "model": str(MODEL_PATH), "model_exists": MODEL_PATH.exists(), **_device_info(DEFAULT_DEVICE)})
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        started = time.perf_counter()
        try:
            if path == "/api/live_reset":
                params = parse_qs(parsed.query)
                nb_steps = max(1, min(6, int(params.get("nb_steps", ["1"])[0])))
                guidance = max(0.0, min(2.0, float(params.get("guidance_structure", ["1"])[0])))
                device_info = _reset_live_model(nb_steps, guidance)
                _json(self, 200, {"ok": True, "elapsed_seconds": time.perf_counter() - started, **device_info})
                return

            if path == "/api/live_chunk":
                params = parse_qs(parsed.query)
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                sr = int(float(params.get("sr", [str(SAMPLE_RATE)])[0]))
                timbre_values = _parse_timbre_values(params.get("zt", ["0,0,0,0,0,0"])[0])
                nb_steps = max(1, min(6, int(params.get("nb_steps", ["1"])[0])))
                guidance = max(0.0, min(2.0, float(params.get("guidance_structure", ["1"])[0])))
                input_gain_db = max(-48.0, min(48.0, float(params.get("input_gain_db", ["0"])[0])))
                wet_mix = max(0.0, min(1.0, float(params.get("wet_mix", ["1"])[0])))
                spring_mix = max(0.0, min(1.0, float(params.get("spring_mix", ["0"])[0])))
                spring_decay = max(0.0, min(1.0, float(params.get("spring_decay", ["0.7"])[0])))
                reverb_boost = max(1.0, min(8.0, float(params.get("reverb_boost", ["1"])[0])))
                delay_mix = max(0.0, min(1.0, float(params.get("delay_mix", ["0"])[0])))
                delay_time_ms = max(1.0, min(2000.0, float(params.get("delay_time_ms", ["320"])[0])))
                delay_feedback = max(0.0, min(0.95, float(params.get("delay_feedback", ["0.35"])[0])))
                body, sample_count, device_info = _process_live_chunk(raw, sr, timbre_values, nb_steps, guidance, input_gain_db, wet_mix, spring_mix, spring_decay, reverb_boost, delay_mix, delay_time_ms, delay_feedback)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Sample-Rate", str(SAMPLE_RATE))
                self.send_header("X-Samples", str(sample_count))
                self.send_header("X-Elapsed-Seconds", f"{time.perf_counter() - started:.6f}")
                self.send_header("X-Device", device_info.get("device", "unknown"))
                self.send_header("X-Device-Name", device_info.get("device_name", "unknown"))
                self.end_headers()
                self.wfile.write(body)
                return

            if path != "/api/generate":
                self.send_error(404)
                return

            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("Expected multipart/form-data.")
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            if "audio" not in form:
                raise ValueError("Missing audio field.")
            item = form["audio"]
            filename = Path(item.filename or "input.wav").name
            suffix = Path(filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(item.file.read())

            try:
                timbre_values = _parse_timbre_values(form.getfirst("zt", "0,0,0,0,0,0"))
                nb_steps = max(1, min(6, int(form.getfirst("nb_steps", "2"))))
                guidance = max(0.0, min(2.0, float(form.getfirst("guidance_structure", "1"))))
                input_gain_db = max(-48.0, min(48.0, float(form.getfirst("input_gain_db", "0"))))
                wet_mix = max(0.0, min(1.0, float(form.getfirst("wet_mix", "1"))))
                spring_mix = max(0.0, min(1.0, float(form.getfirst("spring_mix", "0"))))
                spring_decay = max(0.0, min(1.0, float(form.getfirst("spring_decay", "0.7"))))
                reverb_boost = max(1.0, min(8.0, float(form.getfirst("reverb_boost", "1"))))
                delay_mix = max(0.0, min(1.0, float(form.getfirst("delay_mix", "0"))))
                delay_time_ms = max(1.0, min(2000.0, float(form.getfirst("delay_time_ms", "320"))))
                delay_feedback = max(0.0, min(0.95, float(form.getfirst("delay_feedback", "0.35"))))
                out_name, seconds, chunk_count, device_info = _run_after(tmp_path, timbre_values, nb_steps, guidance, input_gain_db, wet_mix, spring_mix, spring_decay, reverb_boost, delay_mix, delay_time_ms, delay_feedback)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()

            _json(self, 200, {
                "output_url": f"/outputs/{out_name}",
                "source_seconds": seconds,
                "chunks": chunk_count,
                "elapsed_seconds": time.perf_counter() - started,
                **device_info,
            })
        except Exception as exc:
            _json(self, 500, {"error": str(exc)})
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AFTER Local UI running at http://{args.host}:{args.port}", flush=True)
    print(f"Model: {MODEL_PATH}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()






























