#!/usr/bin/env python3
"""
LocalScribe — Full Pipeline: Transcription + Speaker Diarization + Summarization
=================================================================================
Transcribe Hebrew meetings with speaker identification and smart summarization —
100% local on your Mac. Also supports intelligent document summarization.

Audio Pipeline:
1. Speaker Diarization (pyannote.audio) → who spoke and when
2. Hebrew ASR (mlx-whisper + ivrit.ai Turbo) → accurate Hebrew transcription
3. Summarization (Ollama + Gemma) → summary + decisions + action items

Document Pipeline:
1. Read & Parse document (Markdown, TXT, PDF, DOCX)
2. Detect document type (medical, legal, meeting, report, etc.)
3. Summarization (Ollama + Gemma) → type-specific structured summary

Requirements:
- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Ollama (for summarization)
- HuggingFace token (for pyannote, free) — only for audio mode

Usage:
    # Audio mode
    python3 localscribe.py <audio_file>
    python3 localscribe.py --record
    python3 localscribe.py --speakers 3 meeting.mp3
    python3 localscribe.py --simulate-stream meeting.mp3
    python3 localscribe.py --live-stream --duration 600

    # Document mode
    python3 localscribe.py --document <file>
    python3 localscribe.py --document-dir <folder>

    # Interactive
    python3 localscribe.py
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import warnings
import re
import importlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
WHISPER_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"  # MLX-converted ivrit.ai Hebrew model
OLLAMA_MODEL = "gemma4:e4b"  # User-installed; strong Hebrew support
OUTPUT_DIR = Path(__file__).parent / "output"
HF_TOKEN_PATH = Path.home() / ".localscribe_hf_token"

# Diarization settings
MIN_SPEAKERS = 2
MAX_SPEAKERS = 10
MIN_SEGMENT_DURATION = 0.5  # seconds — ignore very short segments

# Document settings
SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf", ".html"}
MAX_DOC_CHARS = 50000  # Maximum characters to send to LLM
HEBREW_LETTER_RE = re.compile(r"[\u0590-\u05FF]")
HEBREW_WORD_RE = re.compile(r"[א-ת]{2,}")
HEBREW_FINAL_LETTERS = set("ךםןףץ")
RTL_MIRROR_TRANSLATION = str.maketrans("()[]{}<>", ")(][}{><")
COMMON_HEBREW_WORDS = {
    "של", "את", "על", "לא", "עם", "או", "אם", "כי", "הוא", "היא", "זה",
    "זו", "יש", "אין", "כל", "גם", "כדי", "לפי", "אשר", "בין", "ידי",
    "בנק", "הבנק", "אישור", "עקרוני", "מסמך", "הלוואה", "הלוואות",
    "ריבית", "מדד", "תשלום", "תשלומים", "מסלול", "מסלולים", "משכנתה",
    "דירה", "דיור", "נכס", "לקוח", "בקשה", "מספר", "תאריך", "סניף",
    "מסגרת", "הסבר", "מבקש", "נתונים", "מחיר", "תקופה", "חודשית",
    "חודשים", "לצרכן", "ישראל",
}

# Audio summarization settings
SUMMARY_CHUNK_SECONDS = 120  # Summarize long meetings in 2-minute windows
SUMMARY_REDUCE_MAX_CHARS = 24000  # Recursively reduce chunk summaries above this size
SUMMARY_REDUCE_GROUP_SIZE = 8
ENABLE_TRANSCRIPT_POLISH = True

# Streaming simulation settings
STREAM_CHUNK_SECONDS = 120  # Simulated live chunks for existing recordings
MIN_STREAM_RMS = 0.0005  # Treat near-silence as empty instead of asking Whisper to guess


# ============================================================
# Dependency Management
# ============================================================
def require_package(package_name: str, import_name: str = None):
    """Fail fast if a required Python package is missing."""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        print(f"  [ERROR] Missing Python package: {package_name}")
        print("          Activate the environment and run: pip install -r requirements.txt")
        sys.exit(1)


def require_command(command_name: str, install_hint: str):
    """Fail fast if a required system command is missing."""
    command_path = shutil.which(command_name)
    if not command_path:
        print(f"  [ERROR] Missing system command: {command_name}")
        print(f"          {install_hint}")
        sys.exit(1)

    real_path = str(Path(command_path).resolve())
    if command_name in {"ffmpeg", "ffprobe"} and (
        ".localscribe_env" in real_path or "static_ffmpeg" in real_path
    ):
        print(f"  [ERROR] {command_name} points to an old static-ffmpeg install:")
        print(f"          {command_path} -> {real_path}")
        print(f"          Remove it: rm -f {Path.home() / '.local' / 'bin' / command_name}")
        print("          Then install the system package: brew install ffmpeg")
        sys.exit(1)

    return command_path


def clean_llm_output(text: str) -> str:
    """Remove Ollama terminal control codes and hidden reasoning blocks."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)^thinking\.\.\..*?\.\.\.done thinking\.\s*", "", text).strip()
    return text


def sanitize_summary_language(summary: str) -> str:
    """Remove decision-like verbs when the summary itself says no decisions were made."""
    if "לא התקבלו החלטות" not in summary:
        return summary

    replacements = {
        "הוחלט כי": "נאמר כי",
        "הוחלט ש": "נאמר ש",
        "נקבע כי": "הוצג כי",
        "נקבע ש": "הוצג ש",
        "נקבעה כי": "הוצג כי",
        "נקבעה ש": "הוצג ש",
        "הוחלטו": "הוצגו",
    }

    for source, target in replacements.items():
        summary = summary.replace(source, target)

    return summary


def clean_polished_transcript_output(text: str) -> str:
    """Remove wrapper labels/models often add around corrected transcripts."""
    text = clean_llm_output(text)
    text = re.sub(r"(?im)^```(?:text|markdown)?\s*$", "", text).strip()
    text = re.sub(r"(?im)^```\s*$", "", text).strip()
    text = re.sub(
        r"(?im)^(corrected transcript|polished transcript|transcript מתוקן|תמלול מתוקן)\s*:?\s*$",
        "",
        text,
    ).strip()
    return text


def get_hf_token() -> Optional[str]:
    """Retrieve HuggingFace token from environment, saved file, or CLI login."""
    # Check environment variable first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Check saved token file
    if HF_TOKEN_PATH.exists():
        token = HF_TOKEN_PATH.read_text().strip()
        if token:
            return token

    # Check huggingface-cli login
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if hf_token_path.exists():
        token = hf_token_path.read_text().strip()
        if token:
            return token

    return None


def setup_hf_token():
    """Interactive setup for HuggingFace token."""
    print()
    print("+" + "=" * 62 + "+")
    print("|  HuggingFace Token Required (free) for Speaker Diarization   |")
    print("+" + "=" * 62 + "+")
    print()
    print("  Instructions:")
    print("  1. Go to: https://huggingface.co/settings/tokens")
    print("  2. Create a new token (Read access is sufficient)")
    print("  3. Accept the pyannote license terms:")
    print("     https://huggingface.co/pyannote/speaker-diarization-3.1")
    print()
    token = input("  Paste your token here: ").strip()
    if token:
        HF_TOKEN_PATH.write_text(token)
        HF_TOKEN_PATH.chmod(0o600)
        print("  Token saved!")
        return token
    return None


def ensure_dependencies(mode: str = "audio"):
    """Verify all dependencies are available."""
    print("Checking dependencies...")

    hf_token = None

    if mode in {"audio", "stream"}:
        # Core packages for audio and simulated streaming modes
        require_package("mlx-whisper", "mlx_whisper")
        print("  [OK] mlx-whisper (transcription)")

        require_package("soundfile", "soundfile")
        print("  [OK] soundfile (audio processing)")

        require_command("ffmpeg", "Install with: brew install ffmpeg")
        require_command("ffprobe", "Install with: brew install ffmpeg")
        print("  [OK] ffmpeg + ffprobe")

    if mode == "audio":
        require_package("torch", "torch")
        require_package("torchaudio", "torchaudio")
        require_package("pyannote.audio", "pyannote.audio")
        print("  [OK] pyannote.audio (speaker diarization)")

        # Check HuggingFace token
        hf_token = get_hf_token()
        if not hf_token:
            hf_token = setup_hf_token()
            if not hf_token:
                print("  [ERROR] Cannot continue without HuggingFace Token")
                sys.exit(1)
        print("  [OK] HuggingFace Token")

    if mode == "document":
        require_package("pdfplumber", "pdfplumber")
        require_package("python-docx", "docx")
        print("  [OK] Document parsing (pdfplumber + python-docx)")

    # Check Ollama (needed for both modes)
    require_command("ollama", "Install with: brew install ollama")
    print("  [OK] Ollama (summarization)")

    # Check Ollama model
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL.split(":")[0] not in result.stdout:
        print(f"  Downloading model {OLLAMA_MODEL}...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    print(f"  [OK] Model {OLLAMA_MODEL} ready")

    print()
    return hf_token


# ============================================================
# Stage 1: Speaker Diarization
# ============================================================
def run_diarization(audio_path: str, hf_token: str, num_speakers: int = None):
    """
    Run speaker diarization using pyannote.audio.
    Returns a list of segments: [{"start", "end", "speaker", "duration"}, ...]
    """
    from pyannote.audio import Pipeline
    import torch

    print("Stage 1: Speaker Diarization (pyannote.audio)...")
    print(f"   File: {Path(audio_path).name}")
    print("   (First run will take longer — downloading model)")
    print()

    start_time = time.time()

    # Load the diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Use MPS (Metal) if available on Apple Silicon
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("   Using Apple Metal GPU for acceleration")

    # Run diarization
    diarization_params = {}
    if num_speakers:
        diarization_params["num_speakers"] = num_speakers
    else:
        diarization_params["min_speakers"] = MIN_SPEAKERS
        diarization_params["max_speakers"] = MAX_SPEAKERS

    normalized_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", "-vn", normalized_wav],
            check=True,
        )
        diarization = pipeline(normalized_wav, **diarization_params)
    finally:
        if os.path.exists(normalized_wav):
            os.unlink(normalized_wav)

    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.duration >= MIN_SEGMENT_DURATION:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.duration,
            })

    elapsed = time.time() - start_time
    unique_speakers = set(seg["speaker"] for seg in segments)

    print(f"   Diarization complete in {elapsed:.1f}s")
    print(f"   Detected {len(unique_speakers)} speakers, {len(segments)} speech segments")
    print()

    return segments


# ============================================================
# Stage 2: Hebrew Transcription (per-segment)
# ============================================================
def transcribe_segments(audio_path: str, segments: list):
    """
    Transcribe each diarized segment using the ivrit.ai Whisper model.
    Returns segments enriched with transcription text.
    """
    import mlx_whisper
    import soundfile as sf
    import numpy as np

    print("Stage 2: Hebrew Transcription (ivrit.ai Turbo)...")
    print(f"   Model: {WHISPER_MODEL}")
    print(f"   Segments to transcribe: {len(segments)}")
    print()

    start_time = time.time()

    # Convert source to 16kHz mono WAV via ffmpeg (avoids pydub/ffprobe — Santa kills ffprobe from subprocess)
    full_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
         "-ac", "1", "-ar", "16000", "-vn", full_wav],
        check=True,
    )
    audio_data, sr = sf.read(full_wav)
    os.unlink(full_wav)

    transcribed_segments = []
    total = len(segments)

    # Group nearby segments from the same speaker for better context
    merged_segments = merge_adjacent_segments(segments, max_gap=1.5)

    for i, seg in enumerate(merged_segments):
        # Extract audio segment (slice numpy array directly)
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        segment_audio = audio_data[start_idx:end_idx]

        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment_audio, sr)
            tmp_path = tmp.name

        try:
            # Transcribe with mlx-whisper
            result = mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo=WHISPER_MODEL,
                language="he",
                task="transcribe",
            )
            text = result["text"].strip()
        except Exception as e:
            text = f"[Transcription error: {e}]"
        finally:
            os.unlink(tmp_path)

        if text:  # Only add non-empty segments
            transcribed_segments.append({
                **seg,
                "text": text,
            })

        # Progress indicator
        progress = (i + 1) / total * 100
        print(f"\r   [{i+1}/{total}] {progress:.0f}% ", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n   Transcription complete in {elapsed:.1f}s")
    print()

    return transcribed_segments


def merge_adjacent_segments(segments: list, max_gap: float = 1.5) -> list:
    """
    Merge adjacent segments from the same speaker if the gap between them is small.
    This improves transcription quality by giving more context to the model.
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start"] - last["end"]

        # Merge if same speaker and gap is small
        if seg["speaker"] == last["speaker"] and gap <= max_gap:
            last["end"] = seg["end"]
            last["duration"] = last["end"] - last["start"]
        else:
            merged.append(seg.copy())

    return merged


# ============================================================
# Stage 3: Summarization with Local LLM
# ============================================================
def build_speaker_map(segments: list) -> dict:
    """Create stable human-readable speaker labels for all segments."""
    speaker_map = {}
    speaker_counter = 1
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"
            speaker_counter += 1
    return speaker_map


def format_transcript_with_speakers(segments: list, speaker_map: dict = None) -> str:
    """Format the transcribed segments into a readable transcript with speaker labels."""
    speaker_map = speaker_map or build_speaker_map(segments)

    lines = []
    for seg in segments:
        speaker = seg["speaker"]
        friendly_name = speaker_map[speaker]
        timestamp = format_timestamp(seg["start"])
        lines.append(f"[{timestamp}] **{friendly_name}:** {seg['text']}")

    return "\n\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_time_range(start: float, end: float) -> str:
    """Format a start/end time range as MM:SS-MM:SS."""
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


def run_ollama_prompt(prompt: str, timeout: int = 180) -> str:
    """Run the configured local LLM and return cleaned text output."""
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, "--hidethinking", "--think", "false", "--nowordwrap", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return clean_llm_output(result.stdout)


def polish_transcript_chunk(transcript: str, start: float = None, end: float = None) -> str:
    """
    Use the local LLM to clean obvious ASR errors while preserving transcript structure.
    The raw transcript remains the source of truth and is always saved separately.
    """
    if not ENABLE_TRANSCRIPT_POLISH or not transcript.strip():
        return transcript

    time_context = ""
    if start is not None and end is not None:
        time_context = f"The transcript covers {format_time_range(start, end)}.\n"

    prompt = f"""/no_think
You are correcting a Hebrew ASR transcript from a meeting.
{time_context}The transcript may contain timestamps and speaker labels such as [03:20] **Speaker 2:**.

Goal: produce a cleaner readable transcript while preserving evidence.

Strict rules:
- Keep every timestamp and speaker label exactly as written.
- Keep the same order and the same line structure.
- Correct only obvious ASR mistakes, punctuation, spacing, and grammar when the intended meaning is clear from context.
- Smooth wording only lightly; do not rewrite into a summary.
- Do not add new facts, names, numbers, dates, decisions, or action items.
- Do not remove uncertain or unclear content. If unclear, keep it close to the original wording.
- Do not infer speaker identities from context.
- Return only the corrected transcript, no commentary.

---
Raw transcript:
{transcript}
---

Corrected transcript:"""

    polished = clean_polished_transcript_output(run_ollama_prompt(prompt, timeout=240))

    # If the model ignored the requested structure, keep the raw transcript.
    raw_markers = transcript.count("[")
    polished_markers = polished.count("[")
    if raw_markers and polished_markers < max(1, raw_markers // 2):
        return transcript

    return polished or transcript


def validate_audio_summary(summary: str, evidence: str, evidence_label: str) -> str:
    """Rewrite a summary conservatively, removing unsupported structured claims."""
    if len(evidence) > SUMMARY_REDUCE_MAX_CHARS:
        evidence = evidence[:SUMMARY_REDUCE_MAX_CHARS] + "\n\n[... Evidence truncated ...]"

    prompt = f"""/no_think
You are a strict verifier for Hebrew audio summaries.
You received source evidence ({evidence_label}) and a draft summary.

Rewrite the draft summary in Hebrew using exactly the same final sections:

## כותרת
## סיכום
## פריטי פעולה
## החלטות שהתקבלו
## נושאים פתוחים

Keep useful content, but remove anything that is not explicitly supported by the evidence.

Strict rules:
- Action items require an explicit accepted responsibility or assignment to a specific person/team/organization for future work.
- Do not list things that already happened during the meeting as action items.
- Do not list "present background", "explain", "clarify", "let someone speak", or "continue the current discussion" as action items.
- Do not convert criticism, rhetorical questions, public demands, policy recommendations, or calls for third parties to act into action items.
- Decisions require explicit approvals, votes, or clear stated decisions.
- Do not treat procedural discussion as a formal decision unless it is explicitly presented as a decision.
- Do not use decision verbs such as "הוחלט" or "נקבע" in the summary unless an explicit decision is supported.
  Prefer "נאמר", "הובהר", "נטען", or "הוצג" for non-decision content.
- Open topics require explicit unresolved questions or deferred items that the meeting says should be revisited, checked, or resolved.
- Do not convert criticism, allegations, public controversy, rhetorical questions, or broad policy disagreement into open topics.
- When unsure, prefer: לא הוגדרו פריטי פעולה / לא התקבלו החלטות / לא הוגדרו נושאים פתוחים.

---
Source evidence:
{evidence}
---

Draft summary:
{summary}
---

Return only the corrected Hebrew summary:"""

    return sanitize_summary_language(run_ollama_prompt(prompt, timeout=240))


def validate_audio_chunk_summary(summary: str, evidence: str, evidence_label: str) -> str:
    """Rewrite a chunk summary conservatively against its transcript evidence."""
    if len(evidence) > SUMMARY_REDUCE_MAX_CHARS:
        evidence = evidence[:SUMMARY_REDUCE_MAX_CHARS] + "\n\n[... Evidence truncated ...]"

    prompt = f"""/no_think
You are a strict verifier for one Hebrew audio chunk summary.
You received source evidence ({evidence_label}) and a draft chunk summary.

Rewrite the draft summary in Hebrew using exactly these sections:

## תמצית החלק
## פריטי פעולה
## החלטות
## נושאים פתוחים

Keep useful content, but remove anything that is not explicitly supported by the evidence.

Strict rules:
- Action items require an explicit accepted responsibility or assignment to a specific person/team/organization for future work.
- Requests to present background, explain context, give someone the floor, or continue the current discussion are not action items.
- Do not convert criticism, rhetorical questions, public demands, policy recommendations, or calls for third parties to act into action items.
- Decisions require explicit approvals, votes, or clear stated decisions.
- Do not use decision verbs such as "הוחלט" or "נקבע" in the chunk summary unless an explicit decision is supported.
  Prefer "נאמר", "הובהר", "נטען", or "הוצג" for non-decision content.
- Open topics require explicit unresolved questions or deferred items that the meeting says should be revisited, checked, or resolved.
- Do not convert criticism, allegations, public controversy, rhetorical questions, or broad policy disagreement into open topics.
- When unsure, prefer: לא הוגדרו פריטי פעולה / לא התקבלו החלטות / לא הוגדרו נושאים פתוחים.

---
Source evidence:
{evidence}
---

Draft chunk summary:
{summary}
---

Return only the corrected Hebrew chunk summary:"""

    return sanitize_summary_language(run_ollama_prompt(prompt, timeout=180))


def split_text_by_durations(text: str, durations: list) -> list:
    """Split text proportionally by time durations when word timestamps are unavailable."""
    if len(durations) <= 1:
        return [text]

    words = text.split()
    if len(words) < len(durations):
        return [text] + [""] * (len(durations) - 1)

    total_duration = sum(durations)
    total_words = len(words)
    parts = []
    cursor = 0

    for i, duration in enumerate(durations):
        remaining_windows = len(durations) - i
        remaining_words = total_words - cursor
        if i == len(durations) - 1:
            word_count = remaining_words
        else:
            proportional_count = round(total_words * (duration / total_duration))
            word_count = max(1, proportional_count)
            word_count = min(word_count, remaining_words - (remaining_windows - 1))

        parts.append(" ".join(words[cursor:cursor + word_count]))
        cursor += word_count

    return parts


def split_segments_for_summary_windows(segments: list, chunk_seconds: int) -> list:
    """
    Split long transcribed segments on summary-window boundaries.
    The text split is proportional because Whisper word timestamps are not stored.
    """
    split_segments = []

    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        if end <= start:
            split_segments.append(seg)
            continue

        first_window = int(start // chunk_seconds)
        last_window = int((end - 0.001) // chunk_seconds)
        if first_window == last_window:
            split_segments.append(seg)
            continue

        ranges = []
        for window in range(first_window, last_window + 1):
            part_start = max(start, window * chunk_seconds)
            part_end = min(end, (window + 1) * chunk_seconds)
            if part_end > part_start:
                ranges.append((part_start, part_end))

        text_parts = split_text_by_durations(
            seg.get("text", ""),
            [part_end - part_start for part_start, part_end in ranges],
        )

        for (part_start, part_end), text_part in zip(ranges, text_parts):
            if not text_part.strip():
                continue
            split_seg = seg.copy()
            split_seg["start"] = part_start
            split_seg["end"] = part_end
            split_seg["duration"] = part_end - part_start
            split_seg["text"] = text_part.strip()
            split_segments.append(split_seg)

    return split_segments


def summarize_with_speakers(transcript: str, num_speakers: int):
    """
    Summarize the audio transcript using a local LLM (Ollama).
    The prompt leverages speaker information for better summaries.
    """
    print(f"Stage 3: Smart Summarization ({OLLAMA_MODEL})...")
    print()

    prompt = f"""/no_think
You are a professional audio transcript summarizer. You received a Hebrew transcript with {num_speakers} labeled speakers.
Each speaker is labeled (Speaker 1, Speaker 2, etc.).

The audio may be a meeting, lesson, social exchange, interview, or language-practice recording.
First infer the content type from the transcript. Do not force a business-meeting frame when the transcript is not a business meeting.
Provide your summary in natural Hebrew.
Do not infer speaker gender from speaker labels. Refer to speakers as "דובר 1", "דובר 2", etc. when needed.

Provide the following:

## כותרת
A short, focused Hebrew title (one line)

## סיכום
3-5 sentences summarizing the actual content

## פריטי פעולה
A numbered list only if concrete action items were explicitly stated.
If none were stated, write: לא הוגדרו פריטי פעולה.

## החלטות שהתקבלו
A list only if concrete decisions were explicitly stated.
If none were stated, write: לא התקבלו החלטות.

## נושאים פתוחים
Topics raised but not resolved, only if explicit.
If none were stated, write: לא הוגדרו נושאים פתוחים.

Avoid inventing agenda, unresolved issues, decisions, owners, or deadlines.
Action items must be explicit future tasks assigned or requested in the transcript.
Only include action items when a participant accepts responsibility or the chair assigns a task
to a specific participant, team, or organization.
Do not list things that already happened, such as speakers presenting background.
Do not list "present background", "explain the issue", "clarify the topic",
or "let a speaker speak" as action items unless they are scheduled as future work.
Do not convert criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act into action items.
When unsure whether something is an action item, write: לא הוגדרו פריטי פעולה.
Decisions must be explicit approvals, votes, or clear stated decisions.
Do not turn explanations, claims, or recommendations into decisions.
Do not treat procedural conversation such as "we will discuss part of the list"
as a formal decision unless it is explicitly presented as a decision.
Open topics must be explicit unresolved questions or deferred items, not general background.
Only include open topics that the meeting explicitly says should be revisited,
continued later, checked, or resolved.
Do not convert criticism, allegations, public controversy, rhetorical questions,
or broad policy disagreement into open topics.
When unsure whether something is an open topic, write: לא הוגדרו נושאים פתוחים.
Use the exact Hebrew section headings above.
Do not add any extra sections or labels beyond the requested section headings.

---
Transcript:
{transcript}
---

Summarize professionally and clearly:"""

    start_time = time.time()

    summary = run_ollama_prompt(prompt, timeout=180)
    elapsed = time.time() - start_time

    print(f"   Summarization complete in {elapsed:.1f}s")
    print()

    return summary


def chunk_segments_by_time(
    segments: list,
    speaker_map: dict,
    chunk_seconds: int = SUMMARY_CHUNK_SECONDS,
) -> list:
    """Group transcribed segments into fixed time windows for long-meeting summaries."""
    if not segments:
        return []

    segments = split_segments_for_summary_windows(segments, chunk_seconds)

    chunks = []
    current_segments = []
    current_window = int(segments[0]["start"] // chunk_seconds)

    for seg in segments:
        window = int(seg["start"] // chunk_seconds)
        if current_segments and window != current_window:
            chunks.append({
                "start": current_segments[0]["start"],
                "end": current_segments[-1]["end"],
                "segments": current_segments,
                "transcript": format_transcript_with_speakers(current_segments, speaker_map),
            })
            current_segments = []
            current_window = window

        current_segments.append(seg)

    if current_segments:
        chunks.append({
            "start": current_segments[0]["start"],
            "end": current_segments[-1]["end"],
            "segments": current_segments,
            "transcript": format_transcript_with_speakers(current_segments, speaker_map),
        })

    return chunks


def summarize_transcript_chunk(chunk: dict, chunk_index: int, total_chunks: int, num_speakers: int) -> str:
    """Summarize one time-bounded transcript chunk."""
    time_range = format_time_range(chunk["start"], chunk["end"])
    transcript_for_summary = chunk.get("polished_transcript") or chunk["transcript"]
    prompt = f"""/no_think
You are summarizing one time-bounded part of a longer Hebrew audio transcript.
This is chunk {chunk_index} of {total_chunks}, covering {time_range}.
The full audio has {num_speakers} labeled speakers.

Provide a compact Hebrew chunk summary. Preserve speaker labels when useful.
Do not infer names, gender, decisions, tasks, or deadlines unless they are explicit.
Action items must be explicit future tasks assigned or requested in this chunk.
Only include action items when a participant accepts responsibility or the chair assigns a task
to a specific participant, team, or organization.
Do not list things that already happened, such as speakers presenting background.
Do not list "present background", "explain the issue", "clarify the topic",
or "let a speaker speak" as action items unless they are scheduled as future work.
Do not convert criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act into action items.
When unsure whether something is an action item, write: לא הוגדרו פריטי פעולה.
Decisions must be explicit approvals, votes, or clear stated decisions.
Do not turn explanations, claims, or recommendations into decisions.
Do not treat procedural conversation such as "we will discuss part of the list"
as a formal decision unless it is explicitly presented as a decision.
Open topics must be explicit unresolved questions or deferred items, not general background.
Only include open topics that the meeting explicitly says should be revisited,
continued later, checked, or resolved.
Do not convert criticism, allegations, public controversy, rhetorical questions,
or broad policy disagreement into open topics.
When unsure whether something is an open topic, write: לא הוגדרו נושאים פתוחים.

Use exactly these sections:

## תמצית החלק
2-4 sentences about what happened in this time window.

## נקודות לפי דוברים
Short bullets only for meaningful speaker-specific points.

## פריטי פעולה
Only explicit action items. If none, write: לא הוגדרו פריטי פעולה.

## החלטות
Only explicit decisions. If none, write: לא התקבלו החלטות.

## נושאים פתוחים
Only explicit unresolved topics. If none, write: לא הוגדרו נושאים פתוחים.

---
Transcript chunk:
{transcript_for_summary}
---

Summarize this chunk professionally and clearly:"""
    draft_summary = run_ollama_prompt(prompt, timeout=180)
    return validate_audio_chunk_summary(
        draft_summary,
        transcript_for_summary,
        "polished transcript chunk with speaker labels",
    )


def format_summary_units_for_prompt(summary_units: list) -> str:
    """Format chunk or reduced summaries for the next summarization level."""
    blocks = []
    for i, unit in enumerate(summary_units, start=1):
        time_range = format_time_range(unit["start"], unit["end"])
        blocks.append(f"### חלק {i} ({time_range})\n{unit['summary']}")
    return "\n\n".join(blocks)


def summary_units_char_count(summary_units: list) -> int:
    """Count summary text characters for recursive reduction decisions."""
    return sum(len(unit["summary"]) for unit in summary_units)


def reduce_summary_group(group: list, level: int, group_index: int, total_groups: int) -> dict:
    """Compress a group of chunk summaries into one intermediate summary."""
    time_range = format_time_range(group[0]["start"], group[-1]["end"])
    prompt = f"""/no_think
You are compressing intermediate Hebrew summaries from a long audio transcript.
This is reduction level {level}, group {group_index} of {total_groups}, covering {time_range}.

Merge the summaries below into one compact Hebrew intermediate summary.
Keep explicit decisions, action items, unresolved topics, and important speaker-specific points.
Do not invent details that are not present in the summaries.
Preserve action items only when they are explicit future tasks assigned or requested.
Drop vague or already-completed items such as presenting background or explaining a topic.
Drop criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act.
Preserve decisions only when they are explicit approvals, votes, or clear stated decisions.
Drop procedural discussion unless it is explicitly presented as a decision.

---
Intermediate summaries:
{format_summary_units_for_prompt(group)}
---

Return a compact structured summary in Hebrew:"""

    return {
        "start": group[0]["start"],
        "end": group[-1]["end"],
        "summary": run_ollama_prompt(prompt, timeout=240),
    }


def reduce_summaries_recursively(summary_units: list) -> list:
    """Recursively reduce summary units until the final prompt is small enough."""
    level = 1
    while (
        len(summary_units) > 1
        and summary_units_char_count(summary_units) > SUMMARY_REDUCE_MAX_CHARS
    ):
        groups = [
            summary_units[i:i + SUMMARY_REDUCE_GROUP_SIZE]
            for i in range(0, len(summary_units), SUMMARY_REDUCE_GROUP_SIZE)
        ]
        print(
            f"   Reducing summaries level {level}: "
            f"{len(summary_units)} -> {len(groups)} groups"
        )

        reduced = []
        for i, group in enumerate(groups, start=1):
            reduced.append(reduce_summary_group(group, level, i, len(groups)))
        summary_units = reduced
        level += 1

    return summary_units


def summarize_from_summary_units(summary_units: list, num_speakers: int) -> str:
    """Create the final meeting summary from chunk or reduced summaries."""
    evidence = format_summary_units_for_prompt(summary_units)
    prompt = f"""/no_think
You are a professional audio transcript summarizer.
You received structured summaries of a Hebrew audio transcript with {num_speakers} labeled speakers.
The summaries are ordered by time and may represent 2-minute chunks or recursively reduced groups.

Create the final Hebrew summary for the whole audio.
Preserve only explicit decisions, action items, unresolved topics, and meaningful speaker-specific points.
Do not invent agenda, owners, deadlines, decisions, or unresolved issues.
Action items must be explicit future tasks assigned or requested.
Only include action items when a participant accepts responsibility or the chair assigns a task
to a specific participant, team, or organization.
Do not list things that already happened, such as speakers presenting background.
Do not list "present background", "explain the issue", "clarify the topic",
or "let a speaker speak" as action items unless they are scheduled as future work.
Do not convert criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act into action items.
When unsure whether something is an action item, write: לא הוגדרו פריטי פעולה.
Decisions must be explicit approvals, votes, or clear stated decisions.
Do not turn explanations, claims, or recommendations into decisions.
Do not treat procedural conversation such as "we will discuss part of the list"
as a formal decision unless it is explicitly presented as a decision.
Open topics must be explicit unresolved questions or deferred items, not general background.
Only include open topics that the meeting explicitly says should be revisited,
continued later, checked, or resolved.
Do not convert criticism, allegations, public controversy, rhetorical questions,
or broad policy disagreement into open topics.
When unsure whether something is an open topic, write: לא הוגדרו נושאים פתוחים.
Do not infer speaker gender from speaker labels. Refer to speakers as "דובר 1", "דובר 2", etc. when needed.

Use exactly these sections:

## כותרת
A short, focused Hebrew title (one line)

## סיכום
3-6 sentences summarizing the whole audio

## פריטי פעולה
A numbered list only if concrete action items were explicitly stated.
If none were stated, write: לא הוגדרו פריטי פעולה.

## החלטות שהתקבלו
A list only if concrete decisions were explicitly stated.
If none were stated, write: לא התקבלו החלטות.

## נושאים פתוחים
Topics raised but not resolved, only if explicit.
If none were stated, write: לא הוגדרו נושאים פתוחים.

---
Time-ordered summaries:
{evidence}
---

Summarize professionally and clearly:"""
    draft_summary = run_ollama_prompt(prompt, timeout=240)
    return validate_audio_summary(draft_summary, evidence, "time-ordered chunk summaries")


def summarize_audio_hierarchically(segments: list, num_speakers: int, speaker_map: dict):
    """
    Summarize audio with time-window chunk summaries and recursive reduction.
    Returns the final summary, first-level chunk summaries, and polished transcript.
    """
    chunks = chunk_segments_by_time(segments, speaker_map)
    if len(chunks) <= 1:
        transcript = format_transcript_with_speakers(segments, speaker_map)
        polished_transcript = polish_transcript_chunk(
            transcript,
            segments[0]["start"] if segments else None,
            segments[-1]["end"] if segments else None,
        )
        return summarize_with_speakers(polished_transcript, num_speakers), [], polished_transcript

    print(f"Stage 3: Hierarchical Summarization ({OLLAMA_MODEL})...")
    print(
        f"   {len(chunks)} chunks, "
        f"{SUMMARY_CHUNK_SECONDS // 60}-minute target windows"
    )
    print()

    start_time = time.time()
    chunk_summaries = []
    polished_transcript_parts = []

    for i, chunk in enumerate(chunks, start=1):
        time_range = format_time_range(chunk["start"], chunk["end"])
        print(f"   Polishing transcript chunk {i}/{len(chunks)} ({time_range})...")
        polished_transcript = polish_transcript_chunk(
            chunk["transcript"],
            chunk["start"],
            chunk["end"],
        )
        chunk["polished_transcript"] = polished_transcript
        polished_transcript_parts.append(polished_transcript)

        print(f"   Summarizing chunk {i}/{len(chunks)} ({time_range})...")
        chunk_summaries.append({
            "index": i,
            "start": chunk["start"],
            "end": chunk["end"],
            "raw_transcript": chunk["transcript"],
            "polished_transcript": polished_transcript,
            "summary": summarize_transcript_chunk(chunk, i, len(chunks), num_speakers),
        })

    summary_units = reduce_summaries_recursively(chunk_summaries)
    print("   Creating final summary from chunk summaries...")
    final_summary = summarize_from_summary_units(summary_units, num_speakers)

    elapsed = time.time() - start_time
    print(f"   Hierarchical summarization complete in {elapsed:.1f}s")
    print()

    polished_transcript = "\n\n".join(part for part in polished_transcript_parts if part.strip())
    return final_summary, chunk_summaries, polished_transcript


# ============================================================
# Document Processing
# ============================================================
def detect_document_type(text: str, filename: str = "") -> str:
    """
    Detect the type of document based on content and filename.
    Returns one of: medical, legal, meeting, report, proposal, hr, technical, general
    """
    text_lower = text.lower()
    fname_lower = filename.lower()

    # Medical indicators (Hebrew + English)
    medical_keywords = [
        "מכתב שחרור", "הפניה רפואית", "אבחנה", "תרופות", "מטופל",
        "ניתוח", "אשפוז", "מרפאה", "בית חולים", "רופא", "מחלקה",
        "תופעות לוואי", "מינון", "בדיקות דם", "צילום רנטגן",
        "discharge", "medical", "clinical", "patient", "diagnosis",
        "medication", "prescription", "referral", "hospital",
    ]
    medical_score = sum(1 for kw in medical_keywords if kw in text_lower)

    # Legal indicators
    legal_keywords = [
        "חוזה", "הסכם", "שכירות", "משכיר", "שוכר", "ערבות",
        "סעיף", "תנאי", "עו\"ד", "חתימה", "פיצוי", "ביטול",
        "contract", "agreement", "legal", "clause", "liability",
        "tenant", "landlord", "lease", "indemnity",
    ]
    legal_score = sum(1 for kw in legal_keywords if kw in text_lower)

    # Meeting indicators
    meeting_keywords = [
        "פרוטוקול", "ישיבה", "סדר יום", "משתתפים", "נוכחים",
        "החלטות", "משימות", "דיון", "הפגישה הבאה", "דירקטוריון",
        "meeting", "minutes", "agenda", "attendees", "action items",
    ]
    meeting_score = sum(1 for kw in meeting_keywords if kw in text_lower)

    # Report indicators
    report_keywords = [
        "דוח", "רבעון", "הכנסות", "הוצאות", "תחזית", "ביצועים",
        "KPI", "ROI", "ARR", "צמיחה", "שיעור", "מדד",
        "report", "quarterly", "revenue", "performance",
    ]
    report_score = sum(1 for kw in report_keywords if kw in text_lower)

    # Proposal indicators
    proposal_keywords = [
        "הצעה", "פרויקט", "תקציב", "לוח זמנים", "סיכונים",
        "ROI", "יעדים", "מטרות", "שלבים", "צוות",
        "proposal", "project", "budget", "timeline", "objectives",
    ]
    proposal_score = sum(1 for kw in proposal_keywords if kw in text_lower)

    # HR indicators
    hr_keywords = [
        "מדיניות", "משאבי אנוש", "עובדים", "שעות עבודה", "חופשה",
        "גיוס", "הדרכה", "שכר", "תנאים סוציאליים",
        "HR", "policy", "employee", "hybrid", "remote work",
    ]
    hr_score = sum(1 for kw in hr_keywords if kw in text_lower)

    # Also check filename
    if "medical" in fname_lower or "רפואי" in fname_lower:
        medical_score += 3
    if "legal" in fname_lower or "חוזה" in fname_lower or "contract" in fname_lower:
        legal_score += 3
    if "meeting" in fname_lower or "פגישה" in fname_lower or "פרוטוקול" in fname_lower:
        meeting_score += 3
    if "report" in fname_lower or "דוח" in fname_lower:
        report_score += 3
    if "proposal" in fname_lower or "הצעה" in fname_lower:
        proposal_score += 3
    if "hr" in fname_lower or "policy" in fname_lower or "מדיניות" in fname_lower:
        hr_score += 3

    scores = {
        "medical": medical_score,
        "legal": legal_score,
        "meeting": meeting_score,
        "report": report_score,
        "proposal": proposal_score,
        "hr": hr_score,
    }

    best_type = max(scores, key=scores.get)
    if scores[best_type] >= 2:
        return best_type
    return "general"


def is_hebrew_text(text: str) -> bool:
    """Return True when the text is predominantly Hebrew."""
    hebrew_words = HEBREW_WORD_RE.findall(text)
    if len(hebrew_words) < 10:
        return False
    hebrew_chars = sum(1 for char in text if HEBREW_LETTER_RE.match(char))
    latin_chars = sum(1 for char in text if "A" <= char <= "Z" or "a" <= char <= "z")
    return hebrew_chars > latin_chars


def get_document_prompt(doc_type: str, text: str) -> str:
    """Generate a summarization prompt tailored to the document type."""

    type_labels = {
        "medical": "Medical",
        "legal": "Legal",
        "meeting": "Meeting Minutes",
        "report": "Report",
        "proposal": "Project Proposal",
        "hr": "HR / Policy",
        "general": "General",
    }

    type_label = type_labels.get(doc_type, "General")

    # Base prompt
    base = f"""/no_think
You are a professional document summarizer. You received a document of type: **{type_label}**.
The document may be in Hebrew or English. Provide your summary in the same language as the document.

"""

    # Type-specific instructions
    if doc_type == "medical":
        instructions = """Provide a structured medical summary including:

## Document Type
(Discharge letter / Referral / Test results / Other)

## Clinical Summary
3-5 sentences summarizing the medical situation, diagnosis, and treatment

## Diagnoses
List of diagnoses mentioned in the document

## Treatments & Medications
List of medications, dosages, and treatment duration

## Recommendations & Follow-up
Instructions for the patient, follow-up visits, and required tests

## Important Notes
When to seek medical attention, warning signs, restrictions"""

    elif doc_type == "legal":
        instructions = """Provide a structured legal summary including:

## Document Type
(Contract / Agreement / Power of Attorney / Other)

## Parties
The parties to the agreement and their details

## Executive Summary
3-5 sentences summarizing the key terms of the agreement

## Key Terms & Clauses
The main clauses and conditions

## Financial Obligations
Amounts, payment schedules, guarantees

## Important Dates
Agreement period, deadlines, termination conditions

## Notes & Risks
Special clauses, risks, points of attention"""

    elif doc_type == "meeting":
        instructions = """Provide a structured meeting summary including:

## Meeting Title
A short, focused name

## Summary
3-5 sentences summarizing the key points of the meeting

## Decisions Made
A numbered list of all decisions

## Action Items
For each item: responsible person, description, deadline

## Open Issues
Topics raised but not resolved

## Next Meeting
Date and planned topics (if mentioned)"""

    elif doc_type == "report":
        instructions = """Provide a structured report summary including:

## Report Title
A short, focused name

## Executive Summary
3-5 sentences summarizing the key findings

## Key Metrics
A list of the most important numbers and KPIs

## Trends & Insights
Analysis of the main trends emerging from the data

## Challenges & Risks
Issues identified and mitigation strategies

## Recommendations / Forecast
Recommended next steps or forecast for the next period"""

    elif doc_type == "proposal":
        instructions = """Provide a structured proposal summary including:

## Project Name
A short, focused name

## Summary
3-5 sentences summarizing the proposal

## Objectives & Goals
What the project aims to achieve

## Scope & Timeline
Key phases and target dates

## Budget
Summary of expected costs

## Risks
Key risks and mitigation strategies

## Expected ROI / Benefits
Expected return on investment"""

    elif doc_type == "hr":
        instructions = """Provide a structured policy summary including:

## Policy Name
A short, focused name

## Summary
3-5 sentences summarizing the key points of the policy

## Guiding Principles
The core principles underlying the policy

## Key Rules & Procedures
The main rules employees need to know

## Changes from Previous Policy
What is new or different (if relevant)

## Impact on Employees
How the policy affects employees in practice"""

    else:  # general
        instructions = """Provide a structured summary including:

## Title
A short, focused name for the document

## Summary
3-5 sentences summarizing the key points

## Key Points
The most important points in the document

## Important Details
Numbers, dates, names, or data worth noting

## Conclusions / Recommendations
Conclusions or recommendations from the document (if any)"""

    if is_hebrew_text(text):
        hebrew_type_labels = {
            "medical": "מסמך רפואי",
            "legal": "מסמך משפטי / פיננסי",
            "meeting": "פרוטוקול / סיכום פגישה",
            "report": "דוח",
            "proposal": "הצעה",
            "hr": "מדיניות משאבי אנוש",
            "general": "כללי",
        }
        base = f"""/no_think
אתה מסכם מסמכים מקצועי. קיבלת מסמך מסוג: **{hebrew_type_labels.get(doc_type, "כללי")}**.
המסמך בעברית. כתוב את כל הסיכום בעברית בלבד, כולל הכותרות. אל תתרגם לאנגלית.
אם פרט לא מופיע במסמך, כתוב שהוא לא צוין במקום להמציא.

"""
        hebrew_instructions = {
            "medical": """ספק סיכום רפואי מובנה הכולל:

## סוג המסמך
מכתב שחרור / הפניה / תוצאות בדיקה / אחר

## סיכום קליני
3-5 משפטים על המצב הרפואי, האבחנות והטיפול

## אבחנות
רשימת האבחנות שמופיעות במסמך

## טיפולים ותרופות
תרופות, מינונים ומשך טיפול

## המלצות ומעקב
הנחיות למטופל, ביקורי המשך ובדיקות נדרשות

## הערות חשובות
סימני אזהרה, מגבלות או מתי לפנות לטיפול""",
            "legal": """ספק סיכום משפטי / פיננסי מובנה הכולל:

## סוג המסמך
חוזה / הסכם / אישור עקרוני / ייפוי כוח / אחר

## צדדים
הצדדים למסמך והפרטים שלהם, אם צוינו

## תקציר מנהלים
3-5 משפטים שמסבירים את מהות המסמך והתנאים המרכזיים

## תנאים וסעיפים מרכזיים
הסעיפים, המגבלות והתנאים העיקריים

## התחייבויות כספיות
סכומים, לוחות תשלומים, ערבויות, ריביות או עמלות

## תאריכים חשובים
תקופות, מועדים, תוקף או תנאי סיום

## הערות וסיכונים
נקודות שדורשות תשומת לב, סיכונים או הסתייגויות""",
            "meeting": """ספק סיכום פגישה מובנה הכולל:

## כותרת הפגישה
שם קצר וממוקד

## סיכום
3-5 משפטים על עיקרי הפגישה

## החלטות שהתקבלו
רשימה ממוספרת של ההחלטות

## משימות להמשך
אחראי, תיאור ודדליין אם צוינו

## נושאים פתוחים
נושאים שעלו ולא נסגרו

## פגישה הבאה
מועד ונושאים מתוכננים, אם צוינו""",
            "report": """ספק סיכום דוח מובנה הכולל:

## כותרת הדוח
שם קצר וממוקד

## תקציר מנהלים
3-5 משפטים על הממצאים המרכזיים

## מדדים מרכזיים
המספרים והנתונים החשובים ביותר

## מגמות ותובנות
ניתוח המגמות המרכזיות

## אתגרים וסיכונים
בעיות, מגבלות או סיכונים שזוהו

## המלצות / תחזית
צעדים מומלצים או תחזית להמשך""",
            "proposal": """ספק סיכום הצעה מובנה הכולל:

## שם הפרויקט
שם קצר וממוקד

## סיכום
3-5 משפטים על ההצעה

## מטרות ויעדים
מה ההצעה מנסה להשיג

## היקף ולוחות זמנים
שלבים מרכזיים ותאריכי יעד

## תקציב
סיכום עלויות צפויות

## סיכונים
סיכונים מרכזיים ודרכי צמצום

## תועלת צפויה
ROI, ערך עסקי או יתרונות צפויים""",
            "hr": """ספק סיכום מדיניות מובנה הכולל:

## שם המדיניות
שם קצר וממוקד

## סיכום
3-5 משפטים על עיקרי המדיניות

## עקרונות מנחים
העקרונות המרכזיים של המדיניות

## כללים ונהלים מרכזיים
הכללים שהעובדים צריכים להכיר

## שינויים מהמדיניות הקודמת
מה חדש או שונה, אם רלוונטי

## השפעה על עובדים
איך המדיניות משפיעה בפועל""",
            "general": """ספק סיכום מובנה הכולל:

## כותרת
שם קצר וממוקד למסמך

## סיכום
3-5 משפטים על הנקודות המרכזיות

## נקודות מרכזיות
הפרטים החשובים ביותר במסמך

## פרטים חשובים
מספרים, תאריכים, שמות או נתונים שיש לשים לב אליהם

## מסקנות / המלצות
מסקנות או המלצות אם הן מופיעות במסמך""",
        }
        instructions = hebrew_instructions.get(doc_type, hebrew_instructions["general"])

    # Truncate text if too long
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n\n[... Document truncated due to length ...]"

    final_instruction = (
        "סכם בצורה מקצועית וברורה בעברית בלבד:"
        if is_hebrew_text(text)
        else "Summarize professionally and clearly:"
    )

    return f"""{base}{instructions}

---
Document:
{text}
---

{final_instruction}"""


def _repair_reversed_hebrew_pdf_token(token: str) -> str:
    """Repair a token from PDFs that expose Hebrew in visual LTR order."""
    if not HEBREW_LETTER_RE.search(token):
        return token

    chunks = re.findall(r"[A-Za-z0-9]+(?:[./:][A-Za-z0-9]+)*|[א-ת]+|.", token)
    repaired = []
    for chunk in reversed(chunks):
        repaired.append(chunk[::-1] if HEBREW_LETTER_RE.search(chunk) else chunk)
    return "".join(repaired).translate(RTL_MIRROR_TRANSLATION)


def _repair_reversed_hebrew_pdf_line(line: str) -> str:
    """Repair one visually ordered RTL line while preserving indentation."""
    if not HEBREW_LETTER_RE.search(line):
        return line

    leading_len = len(line) - len(line.lstrip())
    trailing_len = len(line) - len(line.rstrip())
    leading = line[:leading_len]
    trailing = line[len(line) - trailing_len:] if trailing_len else ""
    core = line.strip()
    if not core:
        return line

    tokens = core.split()
    return leading + " ".join(
        _repair_reversed_hebrew_pdf_token(token) for token in reversed(tokens)
    ) + trailing


def _hebrew_pdf_direction_score(text: str) -> int:
    """Score logical Hebrew order versus visually reversed extraction."""
    words = HEBREW_WORD_RE.findall(text)
    common_hits = sum(1 for word in words if word in COMMON_HEBREW_WORDS)
    bad_final_starts = sum(1 for word in words if word[0] in HEBREW_FINAL_LETTERS)
    return (common_hits * 3) - (bad_final_starts * 2)


def repair_hebrew_pdf_text(text: str) -> str:
    """Fix Hebrew PDFs whose text layer is extracted in visual left-to-right order."""
    if not text or not HEBREW_LETTER_RE.search(text):
        return text

    repaired = "\n".join(
        _repair_reversed_hebrew_pdf_line(line) for line in text.splitlines()
    )
    original_score = _hebrew_pdf_direction_score(text)
    repaired_score = _hebrew_pdf_direction_score(repaired)
    hebrew_words = len(HEBREW_WORD_RE.findall(text))
    threshold = max(8, int(hebrew_words * 0.03))

    if repaired_score > original_score + threshold:
        return repaired
    return text


def read_document(file_path: str) -> str:
    """Read a document file and return its text content."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in {".md", ".txt", ".rtf"}:
        return path.read_text(encoding="utf-8")

    elif ext == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
                    if not page_text:
                        page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return repair_hebrew_pdf_text("\n\n".join(text_parts))
        except ImportError as e:
            raise RuntimeError("Cannot read PDF: pdfplumber is not installed") from e
        except Exception as e:
            raise RuntimeError(f"Cannot read PDF: {e}") from e

    elif ext in {".docx", ".doc"}:
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except ImportError as e:
            raise RuntimeError("Cannot read DOCX: python-docx is not installed") from e
        except Exception as e:
            raise RuntimeError(f"Cannot read DOCX: {e}") from e

    elif ext == ".html":
        try:
            from html.parser import HTMLParser
            html_content = path.read_text(encoding="utf-8")

            class HTMLTextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.result = []
                def handle_data(self, data):
                    self.result.append(data)

            extractor = HTMLTextExtractor()
            extractor.feed(html_content)
            return " ".join(extractor.result)
        except Exception as e:
            raise RuntimeError(f"Cannot read HTML: {e}")

    else:
        # Try reading as plain text
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            raise RuntimeError(f"Unsupported format: {ext}")


def summarize_document(text: str, doc_type: str):
    """Summarize a document using a local LLM (Ollama)."""
    print(f"Summarizing document ({OLLAMA_MODEL})...")
    print()

    prompt = get_document_prompt(doc_type, text)

    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, "--hidethinking", "--think", "false", "--nowordwrap", prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.time() - start_time
    summary = clean_llm_output(result.stdout)

    print(f"   Summarization complete in {elapsed:.1f}s")
    print()

    return summary


def process_document(file_path: str):
    """
    Full document processing pipeline: Read -> Detect Type -> Summarize.
    """
    print()
    print("+" + "=" * 62 + "+")
    print("|           LocalScribe — Document Summarization               |")
    print("|   Read -> Detect Type -> Smart Summary                       |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"File: {file_path}")
    print()

    total_start = time.time()

    # Stage 1: Read document
    print("Stage 1: Reading document...")
    try:
        text = read_document(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return None

    if not text or len(text.strip()) < 50:
        print("[ERROR] Document is empty or too short")
        return None

    print(f"   Read {len(text):,} characters")
    print()

    # Stage 2: Detect document type
    print("Stage 2: Detecting document type...")
    doc_type = detect_document_type(text, file_path)
    type_labels = {
        "medical": "Medical",
        "legal": "Legal",
        "meeting": "Meeting Minutes",
        "report": "Report",
        "proposal": "Project Proposal",
        "hr": "HR / Policy",
        "general": "General",
    }
    print(f"   Detected type: {type_labels.get(doc_type, doc_type)}")
    print()

    # Stage 3: Summarize
    summary = summarize_document(text, doc_type)

    total_elapsed = time.time() - total_start

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(file_path).stem

    md_file = OUTPUT_DIR / f"doc_{base_name}_{timestamp}.md"
    md_content = f"""# Document Summary — LocalScribe

**Processed:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Source file:** {Path(file_path).name}
**Document type:** {type_labels.get(doc_type, doc_type)}
**Original length:** {len(text):,} characters

---

{summary}
"""
    md_file.write_text(md_content, encoding="utf-8")

    json_file = OUTPUT_DIR / f"doc_{base_name}_{timestamp}.json"
    data = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "source_file": str(file_path),
            "document_type": doc_type,
            "original_length": len(text),
            "model": OLLAMA_MODEL,
        },
        "original_text": text[:10000] + ("..." if len(text) > 10000 else ""),
        "summary": summary,
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total processing time: {total_elapsed:.1f}s")
    print()
    print(f"Results saved:")
    print(f"   Markdown: {md_file}")
    print(f"   JSON:     {json_file}")
    print()

    # Display summary
    print("=" * 60)
    print(f"Document Summary ({type_labels.get(doc_type, doc_type)}):")
    print("=" * 60)
    print()
    print(summary)
    print()
    print("=" * 60)

    return md_file


def process_document_dir(dir_path: str):
    """Process all documents in a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        print(f"[ERROR] Directory not found: {dir_path}")
        return

    files = [f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_DOC_EXTENSIONS]
    if not files:
        print(f"[ERROR] No supported documents found in: {dir_path}")
        return

    print(f"Found {len(files)} documents in directory")
    print()

    results = []
    for i, file in enumerate(sorted(files)):
        print(f"{'─' * 60}")
        print(f"  [{i+1}/{len(files)}] {file.name}")
        print(f"{'─' * 60}")
        result = process_document(str(file))
        if result:
            results.append(result)
        print()

    print(f"\n{'=' * 60}")
    print(f"Summarized {len(results)} of {len(files)} documents")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


# ============================================================
# Output & Results (Audio)
# ============================================================
def save_results(
    audio_path: str,
    segments: list,
    transcript: str,
    summary: str,
    summary_chunks: list = None,
    polished_transcript: str = None,
):
    """Save all results to organized output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem
    summary_chunks = summary_chunks or []

    # Count speakers
    unique_speakers = set(seg["speaker"] for seg in segments)
    num_speakers = len(unique_speakers)

    chunk_section = ""
    if summary_chunks:
        chunk_blocks = []
        for chunk in summary_chunks:
            time_range = format_time_range(chunk["start"], chunk["end"])
            chunk_blocks.append(f"### Chunk {chunk['index']} ({time_range})\n\n{chunk['summary']}")
        chunk_section = (
            "\n---\n\n"
            "## Intermediate Chunk Summaries\n\n"
            + "\n\n".join(chunk_blocks)
            + "\n"
        )

    polished_section = ""
    if polished_transcript and polished_transcript.strip() and polished_transcript.strip() != transcript.strip():
        polished_section = f"""
---

## Polished Transcript (LLM-Corrected Draft)

{polished_transcript}
"""

    # Markdown output
    md_file = OUTPUT_DIR / f"{base_name}_{timestamp}.md"
    md_content = f"""# Audio Summary — LocalScribe

**Processed:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Source file:** {Path(audio_path).name}
**Speakers:** {num_speakers}
**Duration:** {format_timestamp(segments[-1]['end'] if segments else 0)}

---

{summary}
{chunk_section}
{polished_section}

---

## Raw Transcript (with Speaker Identification)

{transcript}
"""
    md_file.write_text(md_content, encoding="utf-8")

    # JSON output (for future app integration)
    json_file = OUTPUT_DIR / f"{base_name}_{timestamp}.json"
    data = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "audio_file": str(audio_path),
            "duration_seconds": segments[-1]["end"] if segments else 0,
            "num_speakers": num_speakers,
            "models": {
                "diarization": "pyannote/speaker-diarization-3.1",
                "transcription": WHISPER_MODEL,
                "summarization": OLLAMA_MODEL,
            },
            "summary_mode": "hierarchical" if summary_chunks else "single_pass",
            "summary_chunk_seconds": SUMMARY_CHUNK_SECONDS if summary_chunks else None,
        },
        "segments": segments,
        "transcript": transcript,
        "polished_transcript": polished_transcript,
        "summary": summary,
        "summary_chunks": summary_chunks,
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Results saved:")
    print(f"   Markdown: {md_file}")
    print(f"   JSON:     {json_file}")
    print()

    return md_file, json_file


def display_results(transcript: str, summary: str):
    """Display results in the terminal."""
    print()
    print("=" * 60)
    print("Audio Summary:")
    print("=" * 60)
    print()
    print(summary)
    print()
    print("=" * 60)
    print()
    print("-" * 60)
    print("Full Transcript (with speakers):")
    print("-" * 60)
    print()
    # Show first 2000 chars of transcript
    if len(transcript) > 2000:
        print(transcript[:2000])
        print(f"\n   ... ({len(transcript) - 2000} more characters in saved file)")
    else:
        print(transcript)
    print()
    print("-" * 60)


# ============================================================
# Simulated Streaming (from an existing recording)
# ============================================================
def summarize_stream_chunk(
    transcript: str,
    start: float,
    end: float,
    chunk_index: int,
    total_chunks: int = None,
) -> str:
    """Summarize one simulated streaming chunk."""
    if not transcript.strip():
        return "לא זוהה דיבור ברור בחלק זה."

    time_range = format_time_range(start, end)
    chunk_context = (
        f"This is chunk {chunk_index} of {total_chunks}, covering {time_range}."
        if total_chunks
        else f"This is live chunk {chunk_index}, covering {time_range}."
    )
    prompt = f"""/no_think
You are summarizing a live-style Hebrew audio chunk.
{chunk_context}
The transcript was produced without full-file speaker diarization, so do not claim reliable speaker identities.

Provide a compact Hebrew summary for this chunk.
Do not invent decisions, action items, owners, deadlines, or unresolved issues.
Action items require an explicit accepted responsibility or assignment to a specific person/team/organization.
Do not convert criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act into action items.
Decisions require explicit approvals, votes, or clear stated decisions.
Open topics require explicit unresolved questions or deferred items that should be revisited.

Use exactly these sections:

## תמצית החלק
2-4 sentences about what happened in this chunk.

## פריטי פעולה
Only explicit action items. If none, write: לא הוגדרו פריטי פעולה.

## החלטות
Only explicit decisions. If none, write: לא התקבלו החלטות.

## נושאים פתוחים
Only explicit unresolved topics. If none, write: לא הוגדרו נושאים פתוחים.

---
Transcript chunk:
[טווח זמן {time_range}]
{transcript}
---

Summarize this streaming chunk professionally and clearly:"""

    draft_summary = run_ollama_prompt(prompt, timeout=180)
    return validate_audio_chunk_summary(
        draft_summary,
        f"[טווח זמן {time_range}]\n{transcript}",
        "streaming transcript chunk",
    )


def summarize_stream_final(stream_chunks: list) -> str:
    """Create a final summary from simulated streaming chunk summaries."""
    summary_units = [
        {
            "start": chunk["start"],
            "end": chunk["end"],
            "summary": chunk["summary"],
        }
        for chunk in stream_chunks
        if chunk.get("summary")
    ]

    if not summary_units:
        return "לא זוהה דיבור ברור לסיכום."

    summary_units = reduce_summaries_recursively(summary_units)
    evidence = format_summary_units_for_prompt(summary_units)

    prompt = f"""/no_think
You are creating a final Hebrew summary from live-style chunk summaries.
The original audio was processed in chronological streaming chunks, without reliable full-file speaker diarization.

Create the final summary for the whole audio.
Preserve only explicit decisions, action items, unresolved topics, and important content.
Do not invent agenda, owners, deadlines, speaker identities, decisions, or unresolved issues.
Action items require an explicit accepted responsibility or assignment to a specific person/team/organization.
Do not convert criticism, rhetorical questions, public demands, policy recommendations,
or calls for third parties to act into action items.
Decisions require explicit approvals, votes, or clear stated decisions.
Open topics require explicit unresolved questions or deferred items that should be revisited.

Use exactly these sections:

## כותרת
A short, focused Hebrew title (one line)

## סיכום
3-6 sentences summarizing the whole audio

## פריטי פעולה
A numbered list only if concrete action items were explicitly stated.
If none were stated, write: לא הוגדרו פריטי פעולה.

## החלטות שהתקבלו
A list only if concrete decisions were explicitly stated.
If none were stated, write: לא התקבלו החלטות.

## נושאים פתוחים
Topics raised but not resolved, only if explicit.
If none were stated, write: לא הוגדרו נושאים פתוחים.

---
Time-ordered streaming chunk summaries:
{evidence}
---

Summarize professionally and clearly:"""

    draft_summary = run_ollama_prompt(prompt, timeout=240)
    return validate_audio_summary(draft_summary, evidence, "streaming chunk summaries")


def save_streaming_results(
    audio_path: str,
    stream_chunks: list,
    final_summary: str,
    chunk_seconds: int,
    mode_name: str = "Streaming Simulation",
    mode_description: str = "Simulated streaming from existing recording",
    speaker_tracking: str = "Not reliable in streaming simulation",
    extra_metadata: dict = None,
):
    """Save streaming transcript chunks and summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem
    extra_metadata = extra_metadata or {}

    chunk_blocks = []
    transcript_lines = []
    for chunk in stream_chunks:
        time_range = format_time_range(chunk["start"], chunk["end"])
        transcript = chunk.get("transcript", "").strip() or "[לא זוהה דיבור ברור]"
        transcript_lines.append(f"[{time_range}] {transcript}")
        chunk_blocks.append(
            f"### Chunk {chunk['index']} ({time_range})\n\n"
            f"**Transcript:**\n\n{transcript}\n\n"
            f"**Chunk Summary:**\n\n{chunk['summary']}"
        )

    duration = stream_chunks[-1]["end"] if stream_chunks else 0

    md_file = OUTPUT_DIR / f"stream_{base_name}_{timestamp}.md"
    md_content = f"""# {mode_name} — LocalScribe

**Processed:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Source file:** {Path(audio_path).name}
**Mode:** {mode_description}
**Chunk size:** {chunk_seconds}s
**Chunks:** {len(stream_chunks)}
**Duration:** {format_timestamp(duration)}
**Speaker tracking:** {speaker_tracking}

---

## Final Streaming Summary

{final_summary}

---

## Streaming Chunks

{chr(10).join(chunk_blocks)}

---

## Streaming Transcript

{chr(10).join(transcript_lines)}
"""
    md_file.write_text(md_content, encoding="utf-8")

    json_file = OUTPUT_DIR / f"stream_{base_name}_{timestamp}.json"
    data = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "audio_file": str(audio_path),
            "mode": mode_description,
            "chunk_seconds": chunk_seconds,
            "duration_seconds": duration,
            "speaker_tracking": speaker_tracking,
            "models": {
                "transcription": WHISPER_MODEL,
                "summarization": OLLAMA_MODEL,
            },
            **extra_metadata,
        },
        "chunks": stream_chunks,
        "final_summary": final_summary,
        "streaming_transcript": "\n".join(transcript_lines),
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Streaming simulation results saved:")
    print(f"   Markdown: {md_file}")
    print(f"   JSON:     {json_file}")
    print()

    return md_file, json_file


def process_stream_simulation(audio_path: str, chunk_seconds: int = STREAM_CHUNK_SECONDS):
    """
    Simulate live streaming from an existing recording.
    This intentionally avoids full-file diarization so the quality reflects a live draft.
    """
    import mlx_whisper
    import numpy as np
    import soundfile as sf

    print()
    print("+" + "=" * 62 + "+")
    print("|           LocalScribe — Streaming Simulation                 |")
    print("|   Existing recording -> chunk transcript -> rolling summary  |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"File: {audio_path}")
    print(f"Chunk size: {chunk_seconds}s")
    print("Speaker tracking: disabled for live-style draft")
    print()

    total_start = time.time()

    full_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", "-vn", full_wav],
            check=True,
        )
        audio_data, sr = sf.read(full_wav)
    finally:
        if os.path.exists(full_wav):
            os.unlink(full_wav)

    if len(audio_data) == 0:
        print("[ERROR] Audio file is empty")
        return None

    chunk_samples = int(chunk_seconds * sr)
    total_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
    duration = len(audio_data) / sr
    print(f"Duration: {format_timestamp(duration)}")
    print(f"Chunks to process: {total_chunks}")
    print()

    stream_chunks = []

    for chunk_index, start_sample in enumerate(range(0, len(audio_data), chunk_samples), start=1):
        end_sample = min(start_sample + chunk_samples, len(audio_data))
        chunk_audio = audio_data[start_sample:end_sample]
        start_sec = start_sample / sr
        end_sec = end_sample / sr
        time_range = format_time_range(start_sec, end_sec)

        print(f"Chunk {chunk_index}/{total_chunks} ({time_range})")

        rms = float(np.sqrt(np.mean(np.square(chunk_audio)))) if len(chunk_audio) else 0.0
        transcript = ""
        transcribe_elapsed = 0.0

        if rms >= MIN_STREAM_RMS:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk_audio, sr)
                tmp_path = tmp.name

            try:
                transcribe_start = time.time()
                result = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo=WHISPER_MODEL,
                    language="he",
                    task="transcribe",
                )
                transcribe_elapsed = time.time() - transcribe_start
                transcript = result["text"].strip()
            finally:
                os.unlink(tmp_path)
        else:
            print("   Near-silence detected; skipping transcription")

        summarize_start = time.time()
        chunk_summary = summarize_stream_chunk(
            transcript,
            start_sec,
            end_sec,
            chunk_index,
            total_chunks,
        )
        summarize_elapsed = time.time() - summarize_start

        stream_chunks.append({
            "index": chunk_index,
            "start": start_sec,
            "end": end_sec,
            "rms": rms,
            "transcript": transcript,
            "summary": chunk_summary,
            "transcription_seconds": transcribe_elapsed,
            "summary_seconds": summarize_elapsed,
        })

        print(f"   Transcript chars: {len(transcript)}")
        print(f"   Transcription: {transcribe_elapsed:.1f}s | Summary: {summarize_elapsed:.1f}s")
        print()

    print("Creating final streaming summary from chunk summaries...")
    final_start = time.time()
    final_summary = summarize_stream_final(stream_chunks)
    final_elapsed = time.time() - final_start
    print(f"   Final summary complete in {final_elapsed:.1f}s")
    print()

    total_elapsed = time.time() - total_start
    print(f"Total streaming simulation time: {total_elapsed:.1f}s")
    print()

    md_file, json_file = save_streaming_results(
        audio_path,
        stream_chunks,
        final_summary,
        chunk_seconds,
    )

    print("=" * 60)
    print("Final Streaming Summary:")
    print("=" * 60)
    print()
    print(final_summary)
    print()
    print("=" * 60)

    return md_file


def transcribe_stream_chunk_file(chunk_file: str):
    """Transcribe one 16k mono WAV chunk and return transcript plus basic metrics."""
    import mlx_whisper
    import numpy as np
    import soundfile as sf

    audio_data, sr = sf.read(str(chunk_file))
    duration = len(audio_data) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio_data)))) if len(audio_data) else 0.0

    if rms < MIN_STREAM_RMS:
        return {
            "transcript": "",
            "duration": duration,
            "rms": rms,
            "transcription_seconds": 0.0,
        }

    transcribe_start = time.time()
    result = mlx_whisper.transcribe(
        str(chunk_file),
        path_or_hf_repo=WHISPER_MODEL,
        language="he",
        task="transcribe",
    )

    return {
        "transcript": result["text"].strip(),
        "duration": duration,
        "rms": rms,
        "transcription_seconds": time.time() - transcribe_start,
    }


def get_ready_recorded_chunk(recording_dir: Path, processed_files: set, recorder: subprocess.Popen):
    """
    Return the next chunk file that ffmpeg has finished writing.
    While recording is active, the newest segment is treated as still open.
    """
    files = sorted(recording_dir.glob("chunk_*.wav"))
    if recorder.poll() is None:
        files = files[:-1]

    for file in files:
        if file not in processed_files and file.exists() and file.stat().st_size > 0:
            return file

    return None


def wait_for_recorder_to_stop(recorder: subprocess.Popen, timeout: float = 5.0):
    """Terminate ffmpeg cleanly, then force-kill if needed."""
    if recorder.poll() is not None:
        return

    recorder.terminate()
    try:
        recorder.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        recorder.kill()
        recorder.wait()


def concat_audio_chunks(chunk_files: list, output_path: Path) -> Optional[Path]:
    """Concatenate recorded WAV chunks into one WAV for the final full pass."""
    if not chunk_files:
        return None

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
        list_path = Path(list_file.name)
        for chunk_file in chunk_files:
            escaped = str(Path(chunk_file).resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-ar", "16000", "-ac", "1",
                str(output_path),
            ],
            check=True,
        )
    finally:
        if list_path.exists():
            list_path.unlink()

    return output_path if output_path.exists() and output_path.stat().st_size > 0 else None


def process_live_recorded_chunk(
    chunk_file: Path,
    chunk_index: int,
    start_sec: float,
    stream_chunks: list,
):
    """Transcribe and summarize one live-recorded chunk."""
    metrics = transcribe_stream_chunk_file(str(chunk_file))
    end_sec = start_sec + metrics["duration"]
    time_range = format_time_range(start_sec, end_sec)

    print(f"Processing live chunk {chunk_index} ({time_range})")
    if not metrics["transcript"]:
        print("   No clear speech detected")

    summarize_start = time.time()
    chunk_summary = summarize_stream_chunk(
        metrics["transcript"],
        start_sec,
        end_sec,
        chunk_index,
        total_chunks=None,
    )
    summarize_elapsed = time.time() - summarize_start

    stream_chunks.append({
        "index": chunk_index,
        "start": start_sec,
        "end": end_sec,
        "source_chunk_file": str(chunk_file),
        "rms": metrics["rms"],
        "transcript": metrics["transcript"],
        "summary": chunk_summary,
        "transcription_seconds": metrics["transcription_seconds"],
        "summary_seconds": summarize_elapsed,
    })

    print(f"   Transcript chars: {len(metrics['transcript'])}")
    print(
        f"   Transcription: {metrics['transcription_seconds']:.1f}s "
        f"| Summary: {summarize_elapsed:.1f}s"
    )
    print()

    return end_sec


def process_live_stream(
    duration_seconds: int = None,
    chunk_seconds: int = STREAM_CHUNK_SECONDS,
    final_pass: bool = True,
    num_speakers: int = None,
):
    """
    Record microphone audio continuously into chunks and process them as live drafts.
    After recording, optionally run the full diarization pipeline on the combined audio.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_dir = OUTPUT_DIR / f"live_chunks_{timestamp}"
    recording_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = recording_dir / "chunk_%05d.wav"
    final_recording = OUTPUT_DIR / f"live_recording_{timestamp}.wav"

    print()
    print("+" + "=" * 62 + "+")
    print("|           LocalScribe — Live Streaming                       |")
    print("|   Mic -> live chunks -> draft summaries -> final full pass   |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"Chunk size: {chunk_seconds}s")
    print(f"Duration: {duration_seconds}s" if duration_seconds else "Duration: until Ctrl+C")
    print("Speaker tracking during live draft: disabled")
    if final_pass:
        print("Final pass: enabled (diarization + corrected final summary after recording)")
    else:
        print("Final pass: disabled")
    print()

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation", "-i", ":0",
    ]
    if duration_seconds:
        cmd.extend(["-t", str(duration_seconds)])
    cmd.extend([
        "-ar", "16000", "-ac", "1",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        str(chunk_pattern),
    ])

    print("Starting microphone recorder...")
    recorder = subprocess.Popen(cmd)
    print("Recording. Press Ctrl+C to stop.")
    print()

    processed_files = set()
    processed_order = []
    stream_chunks = []
    chunk_index = 1
    next_start = 0.0
    interrupted = False

    try:
        while True:
            ready_chunk = get_ready_recorded_chunk(recording_dir, processed_files, recorder)
            if ready_chunk is None:
                if recorder.poll() is not None:
                    break
                time.sleep(0.5)
                continue

            processed_files.add(ready_chunk)
            processed_order.append(ready_chunk)
            next_start = process_live_recorded_chunk(
                ready_chunk,
                chunk_index,
                next_start,
                stream_chunks,
            )
            chunk_index += 1

    except KeyboardInterrupt:
        interrupted = True
        print()
        print("Stopping live recording...")
        wait_for_recorder_to_stop(recorder)

    finally:
        wait_for_recorder_to_stop(recorder)

    # Process any final segment ffmpeg closed after stopping.
    while True:
        ready_chunk = get_ready_recorded_chunk(recording_dir, processed_files, recorder)
        if ready_chunk is None:
            break
        processed_files.add(ready_chunk)
        processed_order.append(ready_chunk)
        next_start = process_live_recorded_chunk(
            ready_chunk,
            chunk_index,
            next_start,
            stream_chunks,
        )
        chunk_index += 1

    if not stream_chunks:
        print("[ERROR] No chunks were recorded or processed")
        return None

    print("Creating final live-stream summary from chunk summaries...")
    final_start = time.time()
    final_summary = summarize_stream_final(stream_chunks)
    print(f"   Final live-stream summary complete in {time.time() - final_start:.1f}s")
    print()

    combined = concat_audio_chunks(processed_order, final_recording)
    if combined:
        print(f"Combined recording saved: {combined}")
    else:
        print("[WARN] Could not create combined recording for final pass")
    print(f"Raw chunks saved in: {recording_dir}")
    print()

    md_file, json_file = save_streaming_results(
        str(combined or final_recording),
        stream_chunks,
        final_summary,
        chunk_seconds,
        mode_name="Live Streaming",
        mode_description="Live microphone streaming draft",
        speaker_tracking="Not reliable during live draft; use final full pass for speaker labels",
        extra_metadata={
            "recording_chunks_dir": str(recording_dir),
            "interrupted": interrupted,
            "final_pass_enabled": final_pass,
        },
    )

    print("=" * 60)
    print("Final Live Draft Summary:")
    print("=" * 60)
    print()
    print(final_summary)
    print()
    print("=" * 60)

    if final_pass and combined:
        print()
        print("Running final full pass with diarization on the combined recording...")
        hf_token = ensure_dependencies(mode="audio")
        process_audio(str(combined), hf_token, num_speakers)

    return md_file


# ============================================================
# Recording
# ============================================================
def record_audio(duration_seconds: int = None) -> Optional[str]:
    """Record audio from microphone."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"recording_{timestamp}.wav"

    print("Starting recording...")
    if duration_seconds:
        print(f"   (Recording for {duration_seconds} seconds)")
    else:
        print("   (Press Ctrl+C to stop)")
    print()

    try:
        cmd = ["ffmpeg", "-y"]
        if duration_seconds:
            cmd.extend(["-t", str(duration_seconds)])
        cmd.extend([
            "-f", "avfoundation", "-i", ":0",
            "-ar", "16000", "-ac", "1", str(output_file),
        ])
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nRecording stopped")

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Recording saved: {output_file}")
        return str(output_file)
    return None


# ============================================================
# Main Pipeline
# ============================================================
def process_audio(audio_path: str, hf_token: str, num_speakers: int = None):
    """
    Full audio pipeline: Diarization -> Transcription -> Summarization.
    """
    print()
    print("+" + "=" * 62 + "+")
    print("|           LocalScribe — Full Audio Processing                |")
    print("|   Diarization -> Transcription -> Summarization              |")
    print("+" + "=" * 62 + "+")
    print()
    print(f"File: {audio_path}")
    print()

    total_start = time.time()

    # Stage 1: Diarization
    segments = run_diarization(audio_path, hf_token, num_speakers)

    if not segments:
        print("[ERROR] No speech detected in file")
        return

    # Stage 2: Transcription
    transcribed_segments = transcribe_segments(audio_path, segments)

    if not transcribed_segments:
        print("[ERROR] Transcription failed")
        return

    # Format transcript with stable speaker labels
    speaker_map = build_speaker_map(transcribed_segments)
    transcript = format_transcript_with_speakers(transcribed_segments, speaker_map)
    unique_speakers = set(seg["speaker"] for seg in transcribed_segments)

    # Stage 3: Summarization
    summary, summary_chunks, polished_transcript = summarize_audio_hierarchically(
        transcribed_segments,
        len(unique_speakers),
        speaker_map,
    )

    # Save and display
    total_elapsed = time.time() - total_start

    print(f"Total processing time: {total_elapsed:.1f}s")
    print()

    md_file, json_file = save_results(
        audio_path,
        transcribed_segments,
        transcript,
        summary,
        summary_chunks,
        polished_transcript=polished_transcript,
    )
    display_results(transcript, summary)

    return md_file


# ============================================================
# CLI Interface
# ============================================================
def main():
    """Main entry point."""
    print()
    print("+" + "=" * 62 + "+")
    print("|   LocalScribe v2.0 — Transcription + Diarization + Summary  |")
    print("|   100% Local  |  Hebrew  |  Speaker ID  |  Documents        |")
    print("+" + "=" * 62 + "+")
    print()

    # Check for document mode first (doesn't need HF token)
    if "--document" in sys.argv:
        idx = sys.argv.index("--document")
        if idx + 1 < len(sys.argv):
            doc_path = sys.argv[idx + 1]
            ensure_dependencies(mode="document")
            process_document(doc_path)
            return
        else:
            print("[ERROR] File path required after --document")
            return

    if "--document-dir" in sys.argv:
        idx = sys.argv.index("--document-dir")
        if idx + 1 < len(sys.argv):
            dir_path = sys.argv[idx + 1]
            ensure_dependencies(mode="document")
            process_document_dir(dir_path)
            return
        else:
            print("[ERROR] Directory path required after --document-dir")
            return

    if "--simulate-stream" in sys.argv:
        idx = sys.argv.index("--simulate-stream")
        if idx + 1 >= len(sys.argv):
            print("[ERROR] File path required after --simulate-stream")
            return

        audio_path = sys.argv[idx + 1]
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}")
            sys.exit(1)

        chunk_seconds = STREAM_CHUNK_SECONDS
        if "--chunk-seconds" in sys.argv:
            chunk_idx = sys.argv.index("--chunk-seconds")
            if chunk_idx + 1 >= len(sys.argv):
                print("[ERROR] Seconds value required after --chunk-seconds")
                return
            chunk_seconds = int(sys.argv[chunk_idx + 1])
            if chunk_seconds <= 0:
                print("[ERROR] --chunk-seconds must be positive")
                return

        ensure_dependencies(mode="stream")
        process_stream_simulation(audio_path, chunk_seconds)
        return

    if "--live-stream" in sys.argv:
        chunk_seconds = STREAM_CHUNK_SECONDS
        if "--chunk-seconds" in sys.argv:
            chunk_idx = sys.argv.index("--chunk-seconds")
            if chunk_idx + 1 >= len(sys.argv):
                print("[ERROR] Seconds value required after --chunk-seconds")
                return
            chunk_seconds = int(sys.argv[chunk_idx + 1])
            if chunk_seconds <= 0:
                print("[ERROR] --chunk-seconds must be positive")
                return

        duration_seconds = None
        if "--duration" in sys.argv:
            duration_idx = sys.argv.index("--duration")
            if duration_idx + 1 >= len(sys.argv):
                print("[ERROR] Seconds value required after --duration")
                return
            duration_seconds = int(sys.argv[duration_idx + 1])
            if duration_seconds <= 0:
                print("[ERROR] --duration must be positive")
                return

        num_speakers = None
        if "--speakers" in sys.argv:
            speakers_idx = sys.argv.index("--speakers")
            if speakers_idx + 1 >= len(sys.argv):
                print("[ERROR] Speaker count required after --speakers")
                return
            num_speakers = int(sys.argv[speakers_idx + 1])

        final_pass = "--no-final-pass" not in sys.argv
        ensure_dependencies(mode="stream")
        process_live_stream(
            duration_seconds=duration_seconds,
            chunk_seconds=chunk_seconds,
            final_pass=final_pass,
            num_speakers=num_speakers,
        )
        return

    # Audio mode
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--record":
            hf_token = ensure_dependencies(mode="audio")
            audio_path = record_audio()
            if audio_path:
                process_audio(audio_path, hf_token)
            return

        if arg == "--help" or arg == "-h":
            print("Usage:")
            print()
            print("  Audio mode (transcription + diarization + summarization):")
            print("  python3 localscribe.py <audio_file>            # Process a file")
            print("  python3 localscribe.py --speakers 3 file.mp3   # Specify speaker count")
            print("  python3 localscribe.py --record                # Record and process")
            print("  python3 localscribe.py --simulate-stream file.mp3 --chunk-seconds 120")
            print("  python3 localscribe.py --live-stream --duration 600 --chunk-seconds 120")
            print("  python3 localscribe.py --live-stream --no-final-pass")
            print()
            print("  Document mode (smart summarization):")
            print("  python3 localscribe.py --document <file>       # Summarize a single document")
            print("  python3 localscribe.py --document-dir <folder> # Summarize all docs in folder")
            print()
            print("  Supported document formats: .md .txt .pdf .docx .doc .rtf .html")
            print("  Auto-detected types: medical, legal, meeting, report, proposal, HR, general")
            print()
            return

        # Check for --speakers flag
        num_speakers = None
        args = sys.argv[1:]
        if "--speakers" in args:
            idx = args.index("--speakers")
            if idx + 1 < len(args):
                num_speakers = int(args[idx + 1])
                args.pop(idx)
                args.pop(idx)

        audio_path = args[0] if args else None
        if not audio_path or not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}")
            sys.exit(1)

        hf_token = ensure_dependencies(mode="audio")
        process_audio(audio_path, hf_token, num_speakers)
        return

    # Interactive menu
    print("What would you like to do?")
    print()
    print("  1.  Process an audio file (transcription + diarization + summary)")
    print("  2.  Record a new meeting and process it")
    print("  3.  Transcribe only (no speaker diarization)")
    print("  4.  Summarize a document")
    print("  5.  Summarize all documents in a folder")
    print("  6.  Simulate streaming from an audio file")
    print("  7.  Live stream from microphone")
    print()

    choice = input("Choose (1-7): ").strip()

    if choice == "1":
        hf_token = ensure_dependencies(mode="audio")
        print()
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}")
            return

        print()
        speakers_input = input("Number of speakers (Enter for auto-detect): ").strip()
        num_speakers = int(speakers_input) if speakers_input else None

        process_audio(audio_path, hf_token, num_speakers)

    elif choice == "2":
        hf_token = ensure_dependencies(mode="audio")
        print()
        audio_path = record_audio()
        if audio_path:
            process_audio(audio_path, hf_token)

    elif choice == "3":
        # Simple transcription without diarization (legacy mode)
        hf_token = ensure_dependencies(mode="audio")
        print()
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}")
            return

        import mlx_whisper

        print("Transcribing (without speaker diarization)...")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=WHISPER_MODEL,
            language="he",
            task="transcribe",
        )
        print()
        print("-" * 50)
        print(result["text"])
        print("-" * 50)

    elif choice == "4":
        ensure_dependencies(mode="document")
        print()
        doc_path = input("Enter path to document: ").strip()
        if not os.path.exists(doc_path):
            print(f"[ERROR] File not found: {doc_path}")
            return
        process_document(doc_path)

    elif choice == "5":
        ensure_dependencies(mode="document")
        print()
        dir_path = input("Enter path to folder: ").strip()
        process_document_dir(dir_path)

    elif choice == "6":
        ensure_dependencies(mode="stream")
        print()
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_path):
            print(f"[ERROR] File not found: {audio_path}")
            return

        chunk_input = input(f"Chunk seconds (Enter for {STREAM_CHUNK_SECONDS}): ").strip()
        chunk_seconds = int(chunk_input) if chunk_input else STREAM_CHUNK_SECONDS
        process_stream_simulation(audio_path, chunk_seconds)

    elif choice == "7":
        ensure_dependencies(mode="stream")
        print()
        duration_input = input("Duration seconds (Enter to stop with Ctrl+C): ").strip()
        duration_seconds = int(duration_input) if duration_input else None
        chunk_input = input(f"Chunk seconds (Enter for {STREAM_CHUNK_SECONDS}): ").strip()
        chunk_seconds = int(chunk_input) if chunk_input else STREAM_CHUNK_SECONDS
        final_input = input("Run final full pass with speaker diarization? [Y/n]: ").strip().lower()
        final_pass = final_input not in {"n", "no"}
        num_speakers = None
        if final_pass:
            speakers_input = input("Number of speakers for final pass (Enter for auto-detect): ").strip()
            num_speakers = int(speakers_input) if speakers_input else None

        process_live_stream(
            duration_seconds=duration_seconds,
            chunk_seconds=chunk_seconds,
            final_pass=final_pass,
            num_speakers=num_speakers,
        )

    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
