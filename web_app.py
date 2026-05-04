#!/usr/bin/env python3
"""
LocalScribe Web UI
==================
Small local-only web interface for uploading audio/documents and watching
LocalScribe jobs run with live logs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import cgi


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
UPLOAD_DIR = OUTPUT_DIR / "web_uploads"
LOCALSCRIBE = ROOT_DIR / "localscribe.py"

DOCUMENT_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf", ".html"}
LOG_LIMIT = 300_000

JOBS = {}
JOBS_LOCK = threading.Lock()


def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_filename(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return name or "upload"


def json_response(handler, status: int, data: dict):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, status: int, body: str, content_type: str = "text/html; charset=utf-8"):
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def append_log(job_id: str, text: str):
    if not text:
        return
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["logs"] = (job.get("logs", "") + text)[-LOG_LIMIT:]
        job["updated_at"] = now_iso()


def update_job(job_id: str, **updates):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = now_iso()


def infer_mode(file_path: Path, requested_mode: str) -> str:
    if requested_mode in {"audio", "document", "stream"}:
        return requested_mode
    return "document" if file_path.suffix.lower() in DOCUMENT_EXTENSIONS else "audio"


def build_command(file_path: Path, mode: str, chunk_seconds: int, speakers: str):
    if mode == "document":
        return [sys.executable, str(LOCALSCRIBE), "--document", str(file_path)]
    if mode == "stream":
        return [
            sys.executable,
            str(LOCALSCRIBE),
            "--simulate-stream",
            str(file_path),
            "--chunk-seconds",
            str(chunk_seconds),
        ]

    cmd = [sys.executable, str(LOCALSCRIBE), str(file_path)]
    if speakers:
        cmd.extend(["--speakers", str(speakers)])
    return cmd


def extract_result_paths(log_text: str):
    markdown_matches = re.findall(r"Markdown:\s*(.+?\.md)", log_text)
    json_matches = re.findall(r"JSON:\s*(.+?\.json)", log_text)
    return {
        "markdown": markdown_matches[-1].strip() if markdown_matches else None,
        "json": json_matches[-1].strip() if json_matches else None,
    }


def load_result(path: str):
    if not path:
        return None
    result_path = Path(path)
    if not result_path.exists():
        return None
    return result_path.read_text(encoding="utf-8", errors="replace")


def run_job(job_id: str):
    with JOBS_LOCK:
        job = dict(JOBS[job_id])

    file_path = Path(job["file_path"])
    mode = job["mode"]
    cmd = build_command(
        file_path,
        mode,
        int(job.get("chunk_seconds") or 120),
        job.get("speakers") or "",
    )

    update_job(
        job_id,
        status="running",
        started_at=now_iso(),
        command=" ".join(cmd),
    )
    append_log(job_id, f"$ {' '.join(cmd)}\n\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=0,
            env=env,
        )

        while True:
            chunk = process.stdout.read(1)
            if chunk:
                append_log(job_id, chunk)
            elif process.poll() is not None:
                rest = process.stdout.read()
                if rest:
                    append_log(job_id, rest)
                break

        return_code = process.wait()
        with JOBS_LOCK:
            final_log = JOBS[job_id].get("logs", "")

        paths = extract_result_paths(final_log)
        markdown_text = load_result(paths["markdown"])
        json_text = load_result(paths["json"])
        parsed_json = None
        if json_text:
            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError:
                parsed_json = None

        update_job(
            job_id,
            status="done" if return_code == 0 else "failed",
            return_code=return_code,
            finished_at=now_iso(),
            result_paths=paths,
            markdown=markdown_text,
            result_json=parsed_json,
        )
    except Exception as exc:
        append_log(job_id, f"\n[web_app error] {exc}\n")
        update_job(job_id, status="failed", error=str(exc), finished_at=now_iso())


def job_public_view(job: dict, include_result: bool = True):
    data = {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "mode": job["mode"],
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "updated_at": job.get("updated_at"),
        "return_code": job.get("return_code"),
        "command": job.get("command"),
        "result_paths": job.get("result_paths"),
        "logs": job.get("logs", ""),
    }
    if include_result:
        data["markdown"] = job.get("markdown")
        data["result_json"] = job.get("result_json")
    return data


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LocalScribe</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #f0f3f6;
      --text: #17202a;
      --muted: #667085;
      --line: #d7dde5;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --code: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .brand {
      display: flex;
      align-items: baseline;
      gap: 10px;
      font-weight: 700;
      font-size: 16px;
    }
    .brand span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 56px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      overflow: auto;
    }
    section {
      min-width: 0;
      padding: 16px;
      overflow: auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .panel h2, .panel h3 {
      margin: 0 0 12px;
      font-size: 14px;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    input, select, button {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    input[type=file] {
      padding: 7px;
      background: var(--panel-2);
    }
    button {
      margin-top: 14px;
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: #fff;
      color: var(--text);
      border-color: var(--line);
    }
    button.danger {
      background: #b42318;
      border-color: #b42318;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: .55;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .recorder {
      display: grid;
      gap: 10px;
    }
    .record-status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #98a2b3;
      flex: 0 0 auto;
    }
    .recording .dot {
      background: #d92d20;
      box-shadow: 0 0 0 4px rgba(217, 45, 32, .12);
    }
    audio {
      width: 100%;
      min-height: 36px;
    }
    .drop {
      border: 1px dashed #9aa7b7;
      border-radius: 8px;
      padding: 14px;
      background: #f8fafc;
    }
    .jobs {
      display: grid;
      gap: 8px;
    }
    .job {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      background: #fff;
    }
    .job.active {
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .job-title {
      font-weight: 700;
      word-break: break-word;
    }
    .job-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e6f4f1;
      color: #0f766e;
      font-size: 12px;
      font-weight: 700;
    }
    .status.failed {
      background: #fee4e2;
      color: var(--danger);
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .tab {
      width: auto;
      margin: 0;
      min-height: 32px;
      padding: 6px 10px;
      background: #fff;
      color: var(--text);
      border: 1px solid var(--line);
      font-weight: 600;
    }
    .tab.active {
      border-color: var(--accent);
      color: var(--accent-dark);
      background: #ecfdf3;
    }
    .result {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 420px;
    }
    .result h1, .result h2, .result h3 {
      margin: 1em 0 .45em;
      line-height: 1.25;
    }
    .result h1:first-child, .result h2:first-child, .result h3:first-child {
      margin-top: 0;
    }
    .result p {
      line-height: 1.65;
      margin: .55em 0;
    }
    .result ul, .result ol {
      padding-left: 22px;
      line-height: 1.6;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d1d5db;
      background: var(--code);
      border-radius: 8px;
      padding: 14px;
      min-height: 420px;
      font-size: 12px;
      line-height: 1.45;
    }
    .empty {
      color: var(--muted);
      padding: 30px;
      text-align: center;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
    }
    .metric strong {
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
      word-break: break-word;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">LocalScribe <span>local web console</span></div>
    <div id="serverStatus" class="status">ready</div>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>Record</h2>
        <div class="recorder" id="recorderPanel">
          <div class="record-status" id="recordStatus">
            <span class="dot"></span>
            <span id="recordLabel">Microphone ready</span>
            <strong id="recordTimer">00:00</strong>
          </div>
          <label for="recordModeInput">Process recording as</label>
          <select id="recordModeInput">
            <option value="audio">Audio full pass</option>
            <option value="stream">Streaming simulation</option>
          </select>
          <div class="row">
            <div>
              <label for="recordSpeakersInput">Speakers</label>
              <input id="recordSpeakersInput" type="number" min="1" max="20" placeholder="auto">
            </div>
            <div>
              <label for="recordChunkInput">Chunk seconds</label>
              <input id="recordChunkInput" type="number" min="5" value="120">
            </div>
          </div>
          <button id="startRecordButton" type="button">Start Recording</button>
          <button id="stopRecordButton" type="button" class="danger" disabled>Stop</button>
          <button id="processRecordButton" type="button" class="secondary" disabled>Process Recording</button>
          <audio id="recordPreview" controls hidden></audio>
        </div>
      </div>
      <div class="panel">
        <h2>Upload</h2>
        <form id="uploadForm">
          <div class="drop" id="dropZone">
            <label for="fileInput">Audio or document</label>
            <input id="fileInput" name="file" type="file" multiple required>
          </div>
          <label for="modeInput">Mode</label>
          <select id="modeInput" name="mode">
            <option value="auto">Auto</option>
            <option value="audio">Audio full pass</option>
            <option value="stream">Streaming simulation</option>
            <option value="document">Document</option>
          </select>
          <div class="row">
            <div>
              <label for="speakersInput">Speakers</label>
              <input id="speakersInput" name="speakers" type="number" min="1" max="20" placeholder="auto">
            </div>
            <div>
              <label for="chunkInput">Chunk seconds</label>
              <input id="chunkInput" name="chunk_seconds" type="number" min="5" value="120">
            </div>
          </div>
          <button type="submit">Upload & Process</button>
        </form>
      </div>
      <div class="panel">
        <h3>Jobs</h3>
        <div id="jobs" class="jobs"></div>
      </div>
    </aside>
    <section>
      <div class="metrics">
        <div class="metric"><strong>Status</strong><span id="mStatus">none</span></div>
        <div class="metric"><strong>Mode</strong><span id="mMode">none</span></div>
        <div class="metric"><strong>File</strong><span id="mFile">none</span></div>
        <div class="metric"><strong>Output</strong><span id="mOutput">none</span></div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="summary">Summary</button>
        <button class="tab" data-tab="transcript">Transcript</button>
        <button class="tab" data-tab="json">JSON</button>
        <button class="tab" data-tab="logs">Logs</button>
      </div>
      <div id="content" class="result">
        <div class="empty">No job selected.</div>
      </div>
    </section>
  </main>

  <script>
    const state = {
      jobs: [],
      selected: null,
      tab: 'summary',
      timer: null,
      recorder: null,
      recordStream: null,
      recordChunks: [],
      recordBlob: null,
      recordStartedAt: null,
      recordTimer: null,
      recordMime: '',
    };
    const $ = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function renderMarkdown(markdown) {
      if (!markdown) return '<div class="empty">No result yet.</div>';
      const lines = markdown.split(/\r?\n/);
      const html = [];
      let listOpen = false;
      const closeList = () => { if (listOpen) { html.push('</ul>'); listOpen = false; } };
      for (const raw of lines) {
        const line = raw.trimEnd();
        if (!line.trim()) { closeList(); continue; }
        if (line.startsWith('### ')) { closeList(); html.push(`<h3>${inline(line.slice(4))}</h3>`); continue; }
        if (line.startsWith('## ')) { closeList(); html.push(`<h2>${inline(line.slice(3))}</h2>`); continue; }
        if (line.startsWith('# ')) { closeList(); html.push(`<h1>${inline(line.slice(2))}</h1>`); continue; }
        if (/^[-*]\s+/.test(line)) {
          if (!listOpen) { html.push('<ul>'); listOpen = true; }
          html.push(`<li>${inline(line.replace(/^[-*]\s+/, ''))}</li>`);
          continue;
        }
        closeList();
        html.push(`<p>${inline(line)}</p>`);
      }
      closeList();
      return html.join('');
    }

    function inline(value) {
      return escapeHtml(value).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    }

    function getTranscript(job) {
      const data = job?.result_json;
      if (!data) return '';
      return data.polished_transcript || data.transcript || data.streaming_transcript || '';
    }

    function summaryFromMarkdown(markdown) {
      if (!markdown) return '';
      const transcriptIndex = markdown.search(/\n---\n\n## (Raw|Full|Streaming) Transcript/);
      if (transcriptIndex > 0) return markdown.slice(0, transcriptIndex);
      return markdown;
    }

    function renderContent() {
      const job = state.jobs.find(j => j.id === state.selected);
      const content = $('content');
      if (!job) {
        content.innerHTML = '<div class="empty">No job selected.</div>';
        setMetrics(null);
        return;
      }
      setMetrics(job);
      if (state.tab === 'logs') {
        content.innerHTML = `<pre>${escapeHtml(job.logs || '')}</pre>`;
        return;
      }
      if (state.tab === 'json') {
        content.innerHTML = `<pre>${escapeHtml(JSON.stringify(job.result_json || {}, null, 2))}</pre>`;
        return;
      }
      if (state.tab === 'transcript') {
        content.innerHTML = `<pre>${escapeHtml(getTranscript(job) || 'No transcript yet.')}</pre>`;
        return;
      }
      content.innerHTML = renderMarkdown(summaryFromMarkdown(job.markdown));
    }

    function setMetrics(job) {
      $('mStatus').textContent = job?.status || 'none';
      $('mMode').textContent = job?.mode || 'none';
      $('mFile').textContent = job?.filename || 'none';
      const output = job?.result_paths?.markdown || 'none';
      $('mOutput').textContent = output;
      $('serverStatus').textContent = job?.status || 'ready';
      $('serverStatus').className = job?.status === 'failed' ? 'status failed' : 'status';
    }

    function renderJobs() {
      const wrap = $('jobs');
      if (!state.jobs.length) {
        wrap.innerHTML = '<div class="empty">No jobs.</div>';
        return;
      }
      wrap.innerHTML = state.jobs.map(job => `
        <div class="job ${job.id === state.selected ? 'active' : ''}" data-job="${job.id}">
          <div class="job-title">${escapeHtml(job.filename)}</div>
          <div class="job-meta"><span>${escapeHtml(job.mode)}</span><span>${escapeHtml(job.status)}</span></div>
        </div>
      `).join('');
      wrap.querySelectorAll('.job').forEach(el => {
        el.addEventListener('click', () => {
          state.selected = el.dataset.job;
          renderJobs();
          renderContent();
        });
      });
    }

    async function refreshJobs() {
      const res = await fetch('/api/jobs');
      const list = await res.json();
      state.jobs = await Promise.all(list.jobs.map(async item => {
        const detail = await fetch(`/api/jobs/${item.id}`).then(r => r.json());
        return detail.job;
      }));
      if (!state.selected && state.jobs.length) state.selected = state.jobs[0].id;
      renderJobs();
      renderContent();
    }

    $('uploadForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      const res = await fetch('/api/jobs', { method: 'POST', body: form });
      const payload = await res.json();
      if (!res.ok) {
        alert(payload.error || 'Upload failed');
        return;
      }
      state.selected = payload.jobs[0].id;
      await refreshJobs();
    });

    function pickRecordMimeType() {
      if (!window.MediaRecorder) return '';
      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/ogg;codecs=opus',
      ];
      return candidates.find(type => MediaRecorder.isTypeSupported(type)) || '';
    }

    function extensionForMime(mime) {
      if (mime.includes('mp4')) return 'm4a';
      if (mime.includes('ogg')) return 'ogg';
      return 'webm';
    }

    function setRecordUi(status, isRecording=false) {
      $('recordLabel').textContent = status;
      $('recordStatus').classList.toggle('recording', isRecording);
      $('startRecordButton').disabled = isRecording;
      $('stopRecordButton').disabled = !isRecording;
      $('processRecordButton').disabled = isRecording || !state.recordBlob;
    }

    function updateRecordTimer() {
      if (!state.recordStartedAt) {
        $('recordTimer').textContent = '00:00';
        return;
      }
      const elapsed = Math.floor((Date.now() - state.recordStartedAt) / 1000);
      const minutes = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const seconds = String(elapsed % 60).padStart(2, '0');
      $('recordTimer').textContent = `${minutes}:${seconds}`;
    }

    async function startRecording() {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        alert('Browser recording is not supported in this browser.');
        return;
      }
      state.recordBlob = null;
      state.recordChunks = [];
      $('recordPreview').hidden = true;
      $('recordPreview').removeAttribute('src');

      try {
        state.recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.recordMime = pickRecordMimeType();
        const options = state.recordMime ? { mimeType: state.recordMime } : {};
        state.recorder = new MediaRecorder(state.recordStream, options);
        state.recorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) state.recordChunks.push(event.data);
        };
        state.recorder.onstop = () => {
          state.recordBlob = new Blob(state.recordChunks, { type: state.recordMime || 'audio/webm' });
          const url = URL.createObjectURL(state.recordBlob);
          $('recordPreview').src = url;
          $('recordPreview').hidden = false;
          state.recordStream?.getTracks().forEach(track => track.stop());
          state.recordStream = null;
          clearInterval(state.recordTimer);
          state.recordStartedAt = null;
          setRecordUi(`Recording ready (${Math.round(state.recordBlob.size / 1024)} KB)`, false);
        };
        state.recorder.start();
        state.recordStartedAt = Date.now();
        updateRecordTimer();
        state.recordTimer = setInterval(updateRecordTimer, 500);
        setRecordUi('Recording...', true);
      } catch (error) {
        alert(`Could not access microphone: ${error.message || error}`);
        setRecordUi('Microphone unavailable', false);
      }
    }

    function stopRecording() {
      if (state.recorder && state.recorder.state !== 'inactive') {
        state.recorder.stop();
      }
    }

    async function processRecording() {
      if (!state.recordBlob) {
        alert('No recording available.');
        return;
      }
      const form = new FormData();
      const ext = extensionForMime(state.recordMime || state.recordBlob.type || '');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      form.append('file', state.recordBlob, `browser_recording_${stamp}.${ext}`);
      form.append('mode', $('recordModeInput').value);
      form.append('chunk_seconds', $('recordChunkInput').value || '120');
      const speakers = $('recordSpeakersInput').value.trim();
      if (speakers) form.append('speakers', speakers);

      setRecordUi('Uploading recording...', false);
      const res = await fetch('/api/jobs', { method: 'POST', body: form });
      const payload = await res.json();
      if (!res.ok) {
        alert(payload.error || 'Recording upload failed');
        setRecordUi('Recording ready', false);
        return;
      }
      state.selected = payload.jobs[0].id;
      await refreshJobs();
      setRecordUi('Recording submitted', false);
    }

    $('startRecordButton').addEventListener('click', startRecording);
    $('stopRecordButton').addEventListener('click', stopRecording);
    $('processRecordButton').addEventListener('click', processRecording);

    document.querySelectorAll('.tab').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        state.tab = button.dataset.tab;
        renderContent();
      });
    });

    const dropZone = $('dropZone');
    dropZone.addEventListener('dragover', (event) => {
      event.preventDefault();
      dropZone.style.borderColor = '#0f766e';
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = '#9aa7b7';
    });
    dropZone.addEventListener('drop', (event) => {
      event.preventDefault();
      $('fileInput').files = event.dataTransfer.files;
      dropZone.style.borderColor = '#9aa7b7';
    });

    refreshJobs();
    state.timer = setInterval(refreshJobs, 1500);
  </script>
</body>
</html>
"""


class LocalScribeHandler(BaseHTTPRequestHandler):
    server_version = "LocalScribeWeb/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (now_iso(), fmt % args))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            text_response(self, 200, HTML_PAGE)
            return

        if parsed.path == "/api/jobs":
            with JOBS_LOCK:
                jobs = [
                    job_public_view(job, include_result=False)
                    for job in sorted(JOBS.values(), key=lambda item: item["created_at"], reverse=True)
                ]
            json_response(self, 200, {"jobs": jobs})
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", parsed.path)
        if match:
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    json_response(self, 404, {"error": "job not found"})
                    return
                payload = job_public_view(dict(job), include_result=True)
            json_response(self, 200, {"job": payload})
            return

        json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            json_response(self, 404, {"error": "not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            json_response(self, 400, {"error": "multipart/form-data required"})
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        file_fields = form["file"] if "file" in form else None
        if file_fields is None:
            json_response(self, 400, {"error": "file is required"})
            return
        if not isinstance(file_fields, list):
            file_fields = [file_fields]

        requested_mode = form.getfirst("mode", "auto")
        chunk_seconds = int(form.getfirst("chunk_seconds", "120") or "120")
        speakers = form.getfirst("speakers", "").strip()

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        created = []

        for field in file_fields:
            if not getattr(field, "filename", None):
                continue
            original_name = safe_filename(field.filename)
            job_id = uuid.uuid4().hex[:12]
            upload_path = UPLOAD_DIR / f"{job_id}_{original_name}"
            with upload_path.open("wb") as output:
                while True:
                    chunk = field.file.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            mode = infer_mode(upload_path, requested_mode)
            if mode == "stream" and upload_path.suffix.lower() in DOCUMENT_EXTENSIONS:
                mode = "document"

            job = {
                "id": job_id,
                "status": "queued",
                "filename": original_name,
                "file_path": str(upload_path),
                "mode": mode,
                "chunk_seconds": chunk_seconds,
                "speakers": speakers,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "logs": "",
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
            thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
            thread.start()
            created.append(job_public_view(job, include_result=False))

        if not created:
            json_response(self, 400, {"error": "no uploadable files found"})
            return

        json_response(self, 201, {"jobs": created})


def main():
    parser = argparse.ArgumentParser(description="Run the LocalScribe local web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), LocalScribeHandler)
    print(f"LocalScribe web UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
