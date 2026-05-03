#!/usr/bin/env python3
"""
LocalScribe - Full Pipeline: Transcription + Speaker Diarization + Summarization
=================================================================================
תמלול פגישות בעברית עם זיהוי דוברים וסיכום חכם - הכל מקומי על המאק שלך.

Pipeline:
1. Speaker Diarization (pyannote.audio) → מי דיבר ומתי
2. Hebrew ASR (mlx-whisper + ivrit.ai Turbo) → תמלול עברית מדויק
3. Summarization (Ollama + Qwen3) → סיכום + החלטות + Action Items

Requirements:
- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Ollama (for summarization)
- HuggingFace token (for pyannote, free)

Usage:
    python3 localscribe.py <audio_file>
    python3 localscribe.py                  # interactive menu
    python3 localscribe.py --record         # record and process
"""

import subprocess
import sys
import os
import json
import time
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-d4"  # Best Hebrew ASR model (94-95% accuracy)
OLLAMA_MODEL = "qwen3:1.7b"  # Fast, good Hebrew support
OUTPUT_DIR = Path.home() / "LocalScribe_Output"
HF_TOKEN_PATH = Path.home() / ".localscribe_hf_token"

# Diarization settings
MIN_SPEAKERS = 2
MAX_SPEAKERS = 10
MIN_SEGMENT_DURATION = 0.5  # seconds - ignore very short segments


# ============================================================
# Dependency Management
# ============================================================
def check_and_install(package_name: str, import_name: str = None):
    """Check if a package is installed, install if not."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"  ⬇️  מתקין {package_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            check=True,
            capture_output=True,
        )
        return True


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token for pyannote access."""
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
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  נדרש HuggingFace Token (חינמי) עבור מודל זיהוי הדוברים    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  📋 הוראות:")
    print("  1. היכנס ל: https://huggingface.co/settings/tokens")
    print("  2. צור token חדש (Read access מספיק)")
    print("  3. קבל את תנאי השימוש של pyannote:")
    print("     https://huggingface.co/pyannote/speaker-diarization-3.1")
    print()
    token = input("  הדבק את ה-Token כאן: ").strip()
    if token:
        HF_TOKEN_PATH.write_text(token)
        HF_TOKEN_PATH.chmod(0o600)
        print("  ✅ Token נשמר!")
        return token
    return None


def ensure_dependencies():
    """Verify all dependencies are available."""
    print("🔍 בודק תלויות...")

    # Core packages
    check_and_install("mlx-whisper", "mlx_whisper")
    print("  ✅ mlx-whisper (תמלול)")

    check_and_install("torch", "torch")
    check_and_install("torchaudio", "torchaudio")
    check_and_install("pyannote.audio", "pyannote")
    print("  ✅ pyannote.audio (זיהוי דוברים)")

    check_and_install("pydub", "pydub")
    print("  ✅ pydub (עיבוד אודיו)")

    # Check Ollama
    result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  ❌ Ollama לא מותקן!")
        print("     התקן עם: brew install ollama")
        sys.exit(1)
    print("  ✅ Ollama (סיכום)")

    # Check Ollama model
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL.split(":")[0] not in result.stdout:
        print(f"  ⬇️  מוריד מודל {OLLAMA_MODEL}...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    print(f"  ✅ מודל {OLLAMA_MODEL} מוכן")

    # Check HuggingFace token
    token = get_hf_token()
    if not token:
        token = setup_hf_token()
        if not token:
            print("  ❌ לא ניתן להמשיך ללא HuggingFace Token")
            sys.exit(1)
    print("  ✅ HuggingFace Token")

    # Check ffmpeg
    result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  ⚠️  ffmpeg לא מותקן - התקן עם: brew install ffmpeg")
    else:
        print("  ✅ ffmpeg")

    print()
    return token


# ============================================================
# Stage 1: Speaker Diarization
# ============================================================
def run_diarization(audio_path: str, hf_token: str, num_speakers: int = None):
    """
    Run speaker diarization using pyannote.audio.
    Returns a list of segments: [(start, end, speaker_label), ...]
    """
    from pyannote.audio import Pipeline
    import torch

    print("🎭 שלב 1: זיהוי דוברים (Speaker Diarization)...")
    print(f"   קובץ: {Path(audio_path).name}")
    print("   (הפעם הראשונה תיקח יותר זמן - הורדת המודל)")
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
        print("   🚀 משתמש ב-Apple Metal GPU")
    
    # Run diarization
    diarization_params = {}
    if num_speakers:
        diarization_params["num_speakers"] = num_speakers
    else:
        diarization_params["min_speakers"] = MIN_SPEAKERS
        diarization_params["max_speakers"] = MAX_SPEAKERS

    diarization = pipeline(audio_path, **diarization_params)

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

    print(f"   ✅ זיהוי דוברים הושלם ב-{elapsed:.1f} שניות")
    print(f"   📊 זוהו {len(unique_speakers)} דוברים, {len(segments)} קטעי דיבור")
    print()

    return segments


# ============================================================
# Stage 2: Hebrew Transcription (per-segment)
# ============================================================
def transcribe_segments(audio_path: str, segments: list):
    """
    Transcribe each diarized segment using ivrit.ai Whisper model.
    Returns segments enriched with transcription text.
    """
    import mlx_whisper
    from pydub import AudioSegment

    print("📝 שלב 2: תמלול עברית (ivrit.ai Turbo)...")
    print(f"   מודל: {WHISPER_MODEL}")
    print(f"   {len(segments)} קטעים לתמלול...")
    print()

    start_time = time.time()

    # Load the full audio
    audio = AudioSegment.from_file(audio_path)

    transcribed_segments = []
    total = len(segments)

    # Group nearby segments from the same speaker for better context
    merged_segments = merge_adjacent_segments(segments, max_gap=1.5)

    for i, seg in enumerate(merged_segments):
        # Extract audio segment
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Save to temp file for whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            segment_audio.export(tmp.name, format="wav")
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
            text = f"[שגיאת תמלול: {e}]"
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
    print(f"\n   ✅ תמלול הושלם ב-{elapsed:.1f} שניות")
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
def format_transcript_with_speakers(segments: list) -> str:
    """Format the transcribed segments into a readable transcript with speaker labels."""
    # Create friendly speaker names
    speaker_map = {}
    speaker_counter = 1

    lines = []
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_map:
            speaker_map[speaker] = f"דובר {speaker_counter}"
            speaker_counter += 1

        friendly_name = speaker_map[speaker]
        timestamp = format_timestamp(seg["start"])
        lines.append(f"[{timestamp}] **{friendly_name}:** {seg['text']}")

    return "\n\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def summarize_with_speakers(transcript: str, num_speakers: int):
    """
    Summarize the meeting transcript using local LLM (Ollama).
    The prompt is designed to leverage speaker information.
    """
    print(f"🤖 שלב 3: סיכום חכם ({OLLAMA_MODEL})...")
    print()

    prompt = f"""/no_think
אתה עוזר מקצועי לסיכום פגישות בעברית. קיבלת תמלול של פגישה עם {num_speakers} משתתפים.
כל דובר מסומן (דובר 1, דובר 2, וכו').

עליך לספק:

## כותרת
שם קצר וממוקד לפגישה (שורה אחת)

## סיכום
3-5 משפטים שמסכמים את עיקרי הפגישה

## משימות לביצוע (Action Items)
רשימה ממוספרת. לכל משימה ציין:
- מי אחראי (לפי מספר הדובר)
- מה צריך לעשות
- עד מתי (אם צוין)

## החלטות שהתקבלו
רשימה של החלטות שהתקבלו בפגישה (אם יש)

## נקודות פתוחות
נושאים שעלו אך לא הוכרעו (אם יש)

---
התמלול:
{transcript}
---

סכם בעברית בצורה מקצועית ומסודרת:"""

    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=180,
    )

    elapsed = time.time() - start_time
    summary = result.stdout.strip()

    # Remove thinking tags if present (Qwen3 sometimes outputs them)
    if "<think>" in summary:
        import re
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    print(f"   ✅ סיכום הושלם ב-{elapsed:.1f} שניות")
    print()

    return summary


# ============================================================
# Output & Results
# ============================================================
def save_results(audio_path: str, segments: list, transcript: str, summary: str):
    """Save all results to organized output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem

    # Count speakers
    unique_speakers = set(seg["speaker"] for seg in segments)
    num_speakers = len(unique_speakers)

    # --- Markdown output ---
    md_file = OUTPUT_DIR / f"{base_name}_{timestamp}.md"
    md_content = f"""# סיכום פגישה - LocalScribe

**תאריך עיבוד:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
**קובץ מקור:** {Path(audio_path).name}
**מספר דוברים:** {num_speakers}
**משך:** {format_timestamp(segments[-1]['end'] if segments else 0)}

---

{summary}

---

## תמלול מלא (עם זיהוי דוברים)

{transcript}
"""
    md_file.write_text(md_content, encoding="utf-8")

    # --- JSON output (for future app integration) ---
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
        },
        "segments": segments,
        "transcript": transcript,
        "summary": summary,
    }
    json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"💾 תוצאות נשמרו:")
    print(f"   📄 Markdown: {md_file}")
    print(f"   📊 JSON:     {json_file}")
    print()

    return md_file, json_file


def display_results(transcript: str, summary: str):
    """Display results in the terminal."""
    print()
    print("═" * 60)
    print("📋 סיכום הפגישה:")
    print("═" * 60)
    print()
    print(summary)
    print()
    print("═" * 60)
    print()
    print("─" * 60)
    print("📄 תמלול מלא (עם דוברים):")
    print("─" * 60)
    print()
    # Show first 2000 chars of transcript
    if len(transcript) > 2000:
        print(transcript[:2000])
        print(f"\n   ... ({len(transcript) - 2000} תווים נוספים בקובץ השמור)")
    else:
        print(transcript)
    print()
    print("─" * 60)


# ============================================================
# Recording
# ============================================================
def record_audio(duration_seconds: int = None) -> Optional[str]:
    """Record audio from microphone."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"recording_{timestamp}.wav"

    print("🎙️  מתחיל הקלטה...")
    if duration_seconds:
        print(f"   (הקלטה ל-{duration_seconds} שניות)")
    else:
        print("   (לחץ Ctrl+C כדי לעצור)")
    print()

    try:
        # Try sox/rec first, then ffmpeg
        try:
            cmd = ["rec", "-r", "16000", "-c", "1", "-b", "16", str(output_file)]
            if duration_seconds:
                cmd.extend(["trim", "0", str(duration_seconds)])
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            cmd = [
                "ffmpeg", "-f", "avfoundation", "-i", ":0",
                "-ar", "16000", "-ac", "1", str(output_file),
            ]
            if duration_seconds:
                cmd.insert(1, "-t")
                cmd.insert(2, str(duration_seconds))
            subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  הקלטה הופסקה")

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"✅ הקלטה נשמרה: {output_file}")
        return str(output_file)
    return None


# ============================================================
# Main Pipeline
# ============================================================
def process_audio(audio_path: str, hf_token: str, num_speakers: int = None):
    """
    Full pipeline: Diarization → Transcription → Summarization.
    """
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           LocalScribe - עיבוד מלא                           ║")
    print("║   🎭 זיהוי דוברים → 📝 תמלול עברית → 🤖 סיכום חכם          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"📂 קובץ: {audio_path}")
    print()

    total_start = time.time()

    # Stage 1: Diarization
    segments = run_diarization(audio_path, hf_token, num_speakers)

    if not segments:
        print("❌ לא זוהה דיבור בקובץ")
        return

    # Stage 2: Transcription
    transcribed_segments = transcribe_segments(audio_path, segments)

    if not transcribed_segments:
        print("❌ התמלול נכשל")
        return

    # Format transcript with speaker labels
    transcript = format_transcript_with_speakers(transcribed_segments)
    unique_speakers = set(seg["speaker"] for seg in transcribed_segments)

    # Stage 3: Summarization
    summary = summarize_with_speakers(transcript, len(unique_speakers))

    # Save and display
    total_elapsed = time.time() - total_start

    print(f"⏱️  זמן עיבוד כולל: {total_elapsed:.1f} שניות")
    print()

    md_file, json_file = save_results(
        audio_path, transcribed_segments, transcript, summary
    )
    display_results(transcript, summary)

    return md_file


# ============================================================
# CLI Interface
# ============================================================
def main():
    """Main entry point."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   LocalScribe v2.0 - תמלול + דוברים + סיכום                 ║")
    print("║   🔒 100% מקומי | 🇮🇱 עברית | 🎭 זיהוי דוברים              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    hf_token = ensure_dependencies()

    # Check if audio file was passed as argument
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--record":
            audio_path = record_audio()
            if audio_path:
                process_audio(audio_path, hf_token)
            return

        if arg == "--help" or arg == "-h":
            print("שימוש:")
            print("  python3 localscribe.py <audio_file>    # עיבוד קובץ")
            print("  python3 localscribe.py --record        # הקלטה ועיבוד")
            print("  python3 localscribe.py                 # תפריט אינטראקטיבי")
            print()
            print("אפשרויות נוספות:")
            print("  --speakers N    # ציון מספר דוברים ידוע")
            print()
            return

        # Check for --speakers flag
        num_speakers = None
        if "--speakers" in sys.argv:
            idx = sys.argv.index("--speakers")
            if idx + 1 < len(sys.argv):
                num_speakers = int(sys.argv[idx + 1])
                sys.argv.pop(idx)
                sys.argv.pop(idx)

        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"❌ הקובץ לא נמצא: {audio_path}")
            sys.exit(1)

        process_audio(audio_path, hf_token, num_speakers)
        return

    # Interactive menu
    print("מה תרצה לעשות?")
    print()
    print("  1. 📂  לעבד קובץ אודיו קיים (תמלול + דוברים + סיכום)")
    print("  2. 🎙️  להקליט פגישה חדשה ולעבד")
    print("  3. 📝  לתמלל בלבד (בלי זיהוי דוברים)")
    print()

    choice = input("בחר (1/2/3): ").strip()

    if choice == "1":
        print()
        audio_path = input("הכנס נתיב לקובץ אודיו: ").strip()
        if not os.path.exists(audio_path):
            print(f"❌ הקובץ לא נמצא: {audio_path}")
            return

        # Ask about number of speakers
        print()
        speakers_input = input("מספר דוברים (Enter לזיהוי אוטומטי): ").strip()
        num_speakers = int(speakers_input) if speakers_input else None

        process_audio(audio_path, hf_token, num_speakers)

    elif choice == "2":
        print()
        audio_path = record_audio()
        if audio_path:
            process_audio(audio_path, hf_token)

    elif choice == "3":
        # Simple transcription without diarization (legacy mode)
        print()
        audio_path = input("הכנס נתיב לקובץ אודיו: ").strip()
        if not os.path.exists(audio_path):
            print(f"❌ הקובץ לא נמצא: {audio_path}")
            return

        import mlx_whisper

        print(f"📝 מתמלל (בלי זיהוי דוברים)...")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=WHISPER_MODEL,
            language="he",
            task="transcribe",
        )
        print()
        print("─" * 50)
        print(result["text"])
        print("─" * 50)

    else:
        print("❌ בחירה לא חוקית")


if __name__ == "__main__":
    main()
