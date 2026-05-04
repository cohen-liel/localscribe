#!/usr/bin/env python3
"""
LocalScribe v1.0 (Legacy) — Transcription + Summarization
===========================================================
Simple script that transcribes a Hebrew meeting and summarizes it locally.
NOTE: This is the v1.0 legacy script. For the full pipeline with speaker
diarization, use localscribe.py instead.

Requirements:
- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Ollama installed (for summarization model)
- mlx-whisper (fast transcription on Apple Silicon)

Installation:
    pip install mlx-whisper
    brew install ffmpeg ollama
    ollama pull gemma4:e4b   # or any other Ollama model (qwen3:4b, gemma3:4b, ...)
"""

import subprocess
import sys
import os
import json
import time
import importlib
import shutil
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
WHISPER_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"  # Hebrew-tuned MLX Whisper
OLLAMA_MODEL = "gemma4:e4b"  # Summarization — change to any Ollama model you have
OUTPUT_DIR = Path(__file__).parent / "output"  # Avoids macOS TCC restrictions on ~/Documents


def require_package(package_name, import_name=None):
    """Fail fast if a required Python package is missing."""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"  [ERROR] Missing Python package: {package_name}")
        print("          Activate the environment and run: pip install -r requirements.txt")
        sys.exit(1)


def require_command(command_name, install_hint):
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


def clean_llm_output(text):
    """Remove Ollama terminal control codes and hidden reasoning blocks."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)^thinking\.\.\..*?\.\.\.done thinking\.\s*", "", text).strip()
    return text


def ensure_dependencies():
    """Verify that all dependencies are installed."""
    print("Checking dependencies...")

    require_package("mlx-whisper", "mlx_whisper")
    print("  [OK] mlx-whisper installed")

    require_command("ffmpeg", "Install with: brew install ffmpeg")
    require_command("ffprobe", "Install with: brew install ffmpeg")
    print("  [OK] ffmpeg + ffprobe")

    # Check Ollama
    require_command("ollama", "Install with: brew install ollama")
    print("  [OK] Ollama installed")

    # Check that the model exists
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL.split(":")[0] not in result.stdout:
        print(f"  Downloading model {OLLAMA_MODEL}...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    print(f"  [OK] Model {OLLAMA_MODEL} ready")

    print()


def record_audio(duration_seconds=None):
    """
    Record audio from the microphone.
    If duration_seconds=None, recording continues until Ctrl+C is pressed.
    """
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
            "-ar", "16000", "-ac", "1",
            str(output_file),
        ])
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\nRecording stopped")

    if output_file.exists():
        print(f"Recording saved: {output_file}")
        return str(output_file)
    return None


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using MLX Whisper (very fast on Apple Silicon).
    Supports Hebrew.
    """
    import mlx_whisper

    print(f"Transcribing file: {Path(audio_path).name}")
    print(f"   (Using model: {WHISPER_MODEL})")
    print("   First run may take a few minutes (downloading model)...")
    print()

    start_time = time.time()

    # Transcribe with MLX Whisper — optimized for Apple Silicon
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="he",  # Hebrew
        task="transcribe",
        word_timestamps=True,
    )

    elapsed = time.time() - start_time
    transcript = result["text"]

    print(f"Transcription complete in {elapsed:.1f}s!")
    print(f"   ({len(transcript.split())} words)")
    print()
    print("-" * 50)
    print("Transcript:")
    print("-" * 50)
    print(transcript)
    print("-" * 50)
    print()

    return transcript


def summarize_text(transcript):
    """
    Summarize the text using a local language model (Ollama).
    """
    print(f"Summarizing transcript with {OLLAMA_MODEL}...")
    print()

    prompt = f"""/no_think
You are a professional audio transcript summarizer. You received a Hebrew transcript.
The audio may be a meeting, lesson, social exchange, interview, or language-practice recording.
Do not force a business-meeting frame when the transcript is not a business meeting.
Do not infer speaker gender from speaker labels.

Provide in natural Hebrew:

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

Avoid inventing agenda, unresolved issues, decisions, owners, or deadlines.
Use the exact Hebrew section headings above.
Do not add any extra sections or labels beyond the requested section headings.

The transcript:
---
{transcript}
---

Summarize in Hebrew:"""

    # Call Ollama
    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, "--hidethinking", "--think", "false", "--nowordwrap", prompt],
        capture_output=True,
        text=True,
        timeout=120
    )

    elapsed = time.time() - start_time
    summary = clean_llm_output(result.stdout)

    print(f"Summarization complete in {elapsed:.1f}s!")
    print()
    print("=" * 50)
    print("Transcript Summary:")
    print("=" * 50)
    print(summary)
    print("=" * 50)
    print()

    return summary


def save_results(audio_path, transcript, summary):
    """Save results to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as Markdown
    md_file = OUTPUT_DIR / f"summary_{timestamp}.md"

    content = f"""# Transcript Summary
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Source file:** {Path(audio_path).name}

---

## Summary

{summary}

---

## Full Transcript

{transcript}
"""

    md_file.write_text(content, encoding="utf-8")
    print(f"Results saved to: {md_file}")

    # Also save as JSON (for future app integration)
    json_file = OUTPUT_DIR / f"meeting_{timestamp}.json"
    data = {
        "date": datetime.now().isoformat(),
        "audio_file": str(audio_path),
        "transcript": transcript,
        "summary": summary,
        "model_transcription": WHISPER_MODEL,
        "model_summary": OLLAMA_MODEL,
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return md_file


def process_existing_file(audio_path):
    """Process an existing audio file."""
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        sys.exit(1)

    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript)
    result_file = save_results(audio_path, transcript, summary)

    return result_file


def main():
    """Main menu."""
    print()
    print("+" + "=" * 56 + "+")
    print("|   LocalScribe v1.0 (Legacy) — Transcribe + Summarize  |")
    print("|   100% Offline  |  Hebrew  |  Apple Silicon            |")
    print("+" + "=" * 56 + "+")
    print()
    print("NOTE: For full pipeline with speaker diarization, use localscribe.py")
    print()

    ensure_dependencies()

    # Check if a file was passed as argument
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Processing file: {audio_path}")
        process_existing_file(audio_path)
        return

    # Interactive menu
    print("What would you like to do?")
    print("  1.  Record a new meeting and summarize")
    print("  2.  Transcribe and summarize an existing audio file")
    print("  3.  Summarize already-transcribed text")
    print()

    choice = input("Choose (1/2/3): ").strip()

    if choice == "1":
        print()
        audio_path = record_audio()
        if audio_path:
            transcript = transcribe_audio(audio_path)
            summary = summarize_text(transcript)
            save_results(audio_path, transcript, summary)

    elif choice == "2":
        print()
        audio_path = input("Enter path to audio file: ").strip()
        process_existing_file(audio_path)

    elif choice == "3":
        print()
        print("Paste the text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
            lines.append(line)
        transcript = "\n".join(lines[:-1])  # Remove trailing empty line
        summary = summarize_text(transcript)
        save_results("manual_input", transcript, summary)

    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
