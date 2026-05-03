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
    brew install ollama
    ollama pull qwen3:1.7b
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # Transcription model — excellent Hebrew support
OLLAMA_MODEL = "qwen3:1.7b"  # Summarization model — small, fast, works great on M4
OUTPUT_DIR = Path.home() / "LocalScribe_Output"


def ensure_dependencies():
    """Verify that all dependencies are installed."""
    print("Checking dependencies...")

    # Check mlx-whisper
    try:
        import mlx_whisper
        print("  [OK] mlx-whisper installed")
    except ImportError:
        print("  [MISSING] mlx-whisper not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-whisper"], check=True)
        print("  [OK] mlx-whisper installed successfully")

    # Check Ollama
    result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  [ERROR] Ollama not installed!")
        print("          Install with: brew install ollama")
        print("          Or download from: https://ollama.com/download")
        sys.exit(1)
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
        cmd = ["rec", "-r", "16000", "-c", "1", "-b", "16", str(output_file)]
        if duration_seconds:
            cmd.extend(["trim", "0", str(duration_seconds)])

        # Try sox/rec first; if not available, fall back to ffmpeg
        try:
            process = subprocess.run(cmd, check=True)
        except FileNotFoundError:
            # Fallback: use ffmpeg
            cmd = [
                "ffmpeg", "-f", "avfoundation", "-i", ":0",
                "-ar", "16000", "-ac", "1",
                str(output_file)
            ]
            if duration_seconds:
                cmd.insert(1, "-t")
                cmd.insert(2, str(duration_seconds))
            process = subprocess.run(cmd, check=True)

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
    print(f"Summarizing meeting with {OLLAMA_MODEL}...")
    print()

    prompt = f"""You are a meeting summarizer. You received a transcript of a meeting in Hebrew.
Provide:
1. **Title** — a short name for the meeting (one line)
2. **Summary** — 3-5 sentences summarizing the key points
3. **Action Items** — a numbered list of things to do
4. **Decisions Made** — if any

The transcript:
---
{transcript}
---

Summarize in Hebrew:"""

    # Call Ollama
    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=120
    )

    elapsed = time.time() - start_time
    summary = result.stdout.strip()

    print(f"Summarization complete in {elapsed:.1f}s!")
    print()
    print("=" * 50)
    print("Meeting Summary:")
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
    md_file = OUTPUT_DIR / f"meeting_{timestamp}.md"

    content = f"""# Meeting Summary
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
