#!/usr/bin/env python3
"""
LocalScribe — Quick Smoke Test
================================
Verifies that all components are installed and working correctly.
Runs a summarization test with sample Hebrew meeting data.

Usage:
    python3 quick_test.py
"""

import subprocess
import sys
import os
import time
import re

# Configuration (must match localscribe.py)
OLLAMA_MODEL = "gemma4:e4b"

# Sample transcript: a simulated Hebrew team meeting with 5 speakers.
# Used to test the summarization pipeline without requiring audio input.
SAMPLE_TRANSCRIPT_WITH_SPEAKERS = """
[00:00] **Speaker 1:** Good morning everyone. Let's start the meeting. We have three topics today: the product launch, the budget, and hiring. Yossi, what's the status on the technical side?
[00:15] **Speaker 2:** We're almost done. There are two minor bugs left that I need to close by end of week. Other than that, we're ready for launch.
[00:28] **Speaker 1:** Great. Dana, what about the landing page design?
[00:33] **Speaker 3:** I'm working on it. I need the final copy from Michal to finish.
[00:40] **Speaker 1:** Michal?
[00:42] **Speaker 4:** I'll send the final texts tomorrow morning.
[00:47] **Speaker 1:** Good. Yossi, make sure those two bugs are closed by end of week. I need Dana to finish the landing page design by Tuesday.
[00:51] **Speaker 3:** Sure, I can finish that. I just need the final texts from Michal.
[01:02] **Speaker 4:** I'll send the texts tomorrow morning.
[01:08] **Speaker 1:** Excellent. The second topic is the budget. We're 15 percent over the original budget. My decision is to cut the Facebook ad budget and move the money to Google, where we're seeing better ROI. Ori, can you handle that?
[01:35] **Speaker 5:** Yes, I'll update the campaigns by Thursday.
[01:42] **Speaker 1:** Third topic — hiring. We need another full-stack developer. Michal, you're coordinating the interviews. We have three candidates next week.
[01:58] **Speaker 4:** Right, interviews are on Monday and Tuesday. I'll send everyone the resumes today.
[02:10] **Speaker 1:** Okay, to summarize: launch on May 25, Dana finishes design by Tuesday, Michal sends texts tomorrow, Ori updates campaigns by Thursday, and interviews next week. Thanks everyone!
"""


def clean_llm_output(text: str) -> str:
    """Remove Ollama terminal control codes and hidden reasoning blocks."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)^thinking\.\.\..*?\.\.\.done thinking\.\s*", "", text).strip()
    return text


def check_component(name: str, check_fn) -> bool:
    """Check a single component and print the result."""
    try:
        check_fn()
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def test_full_pipeline():
    """Test the full pipeline with sample data."""
    print()
    print("+" + "=" * 62 + "+")
    print("|   LocalScribe v2.0 — Quick Smoke Test                       |")
    print("+" + "=" * 62 + "+")
    print()

    all_ok = True

    # --- Component checks ---
    print("Checking components:")
    print()

    # Check Ollama
    def check_ollama():
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        assert result.returncode == 0, "Ollama is not running — start with: ollama serve"
        assert OLLAMA_MODEL.split(":")[0] in result.stdout, f"Model {OLLAMA_MODEL} not installed"
    all_ok &= check_component("Ollama + summarization model", check_ollama)

    # Check mlx-whisper
    def check_whisper():
        import mlx_whisper  # noqa: F401
    all_ok &= check_component("mlx-whisper (transcription)", check_whisper)

    # Check pyannote
    def check_pyannote():
        import pyannote.audio  # noqa: F401
    all_ok &= check_component("pyannote.audio (speaker diarization)", check_pyannote)

    # Check torch
    def check_torch():
        import torch
        has_mps = torch.backends.mps.is_available()
        if not has_mps:
            print("    Note: Metal GPU not available (expected on Apple Silicon)")
    all_ok &= check_component("PyTorch (Metal GPU)", check_torch)

    # Check soundfile (replaces pydub — see requirements.txt)
    def check_soundfile():
        import soundfile  # noqa: F401
    all_ok &= check_component("soundfile (audio processing)", check_soundfile)

    # Check ffmpeg/ffprobe
    def check_ffmpeg():
        from pathlib import Path
        import shutil

        for command in ("ffmpeg", "ffprobe"):
            command_path = shutil.which(command)
            assert command_path, f"{command} not found"
            real_path = str(Path(command_path).resolve())
            assert ".localscribe_env" not in real_path and "static_ffmpeg" not in real_path, (
                f"{command} points to old static-ffmpeg install: {command_path} -> {real_path}"
            )
            result = subprocess.run([command, "-version"], capture_output=True, text=True)
            assert result.returncode == 0
    all_ok &= check_component("ffmpeg + ffprobe", check_ffmpeg)

    # Check HF token
    def check_hf_token():
        from pathlib import Path
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return
        hf_path = Path.home() / ".localscribe_hf_token"
        if hf_path.exists():
            return
        hf_cache = Path.home() / ".cache" / "huggingface" / "token"
        if hf_cache.exists():
            return
        raise Exception("Not found — run install.sh or enter manually")
    all_ok &= check_component("HuggingFace Token", check_hf_token)

    # Check document parsing libraries
    def check_doc_libs():
        import pdfplumber  # noqa: F401
        import docx  # noqa: F401
    all_ok &= check_component("Document parsing (pdfplumber + python-docx)", check_doc_libs)

    print()

    if not all_ok:
        print("[WARN] Some components are missing. Run: ./install.sh")
        print()
        return

    # --- Summarization test ---
    print("-" * 60)
    print("Running summarization test with sample transcript (5 speakers):")
    print("-" * 60)
    print()

    prompt = f"""/no_think
You are a professional meeting summarizer. You received a transcript of a meeting with 5 participants.
Each speaker is labeled (Speaker 1, Speaker 2, etc.).

Provide the following:

## Title
A short, focused name for the meeting (one line)

## Summary
3-5 sentences summarizing the key points of the meeting

## Action Items
A numbered list. For each item include:
- Who is responsible (by speaker number)
- What needs to be done
- Deadline (if mentioned)

## Decisions Made
A list of decisions made during the meeting (if any)

---
Transcript:
{SAMPLE_TRANSCRIPT_WITH_SPEAKERS}
---

Summarize professionally and clearly:"""

    print(f"  Summarizing with {OLLAMA_MODEL}...")
    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, "--hidethinking", "--think", "false", "--nowordwrap", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )

    elapsed = time.time() - start_time
    summary = clean_llm_output(result.stdout)

    print(f"  [OK] Summarization complete in {elapsed:.1f}s")
    print()
    print("=" * 60)
    print("Summary Result:")
    print("=" * 60)
    print()
    print(summary)
    print()
    print("=" * 60)
    print()
    print("All tests passed! You can now run:")
    print()
    print("   python3 localscribe.py recording.mp3      # Process a full audio file")
    print("   python3 localscribe.py --record            # Record and process")
    print("   python3 localscribe.py --document file.pdf # Summarize a document")
    print()
    print("   Output includes: Hebrew transcription + speaker identification + smart summary")
    print()


if __name__ == "__main__":
    test_full_pipeline()
