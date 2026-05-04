#!/usr/bin/env python3
"""
LocalScribe — Full Pipeline: Transcription + Speaker Diarization + Summarization
=================================================================================
Transcribe Hebrew meetings with speaker identification and smart summarization —
100% local on your Mac. Also supports intelligent document summarization.

Audio Pipeline:
1. Speaker Diarization (pyannote.audio) → who spoke and when
2. Hebrew ASR (mlx-whisper + ivrit.ai Turbo) → accurate Hebrew transcription
3. Summarization (Ollama + Qwen3) → summary + decisions + action items

Document Pipeline:
1. Read & Parse document (Markdown, TXT, PDF, DOCX)
2. Detect document type (medical, legal, meeting, report, etc.)
3. Summarization (Ollama + Qwen3) → type-specific structured summary

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


# ============================================================
# Dependency Management
# ============================================================
def check_and_install(package_name: str, import_name: str = None):
    """Check if a package is installed; install it if missing."""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        print(f"  Installing {package_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            check=True,
            capture_output=True,
        )
        return True


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

    if mode == "audio":
        # Core packages for audio mode
        check_and_install("mlx-whisper", "mlx_whisper")
        print("  [OK] mlx-whisper (transcription)")

        check_and_install("torch", "torch")
        check_and_install("torchaudio", "torchaudio")
        check_and_install("pyannote.audio", "pyannote")
        print("  [OK] pyannote.audio (speaker diarization)")

        check_and_install("soundfile", "soundfile")
        print("  [OK] soundfile (audio processing)")

        # Check HuggingFace token
        hf_token = get_hf_token()
        if not hf_token:
            hf_token = setup_hf_token()
            if not hf_token:
                print("  [ERROR] Cannot continue without HuggingFace Token")
                sys.exit(1)
        print("  [OK] HuggingFace Token")

        # Check ffmpeg
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if result.returncode != 0:
            print("  [WARN] ffmpeg not installed — install with: brew install ffmpeg")
        else:
            print("  [OK] ffmpeg")

    # Check Ollama (needed for both modes)
    result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  [ERROR] Ollama not installed!")
        print("          Install with: brew install ollama")
        sys.exit(1)
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
def format_transcript_with_speakers(segments: list) -> str:
    """Format the transcribed segments into a readable transcript with speaker labels."""
    speaker_map = {}
    speaker_counter = 1

    lines = []
    for seg in segments:
        speaker = seg["speaker"]
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        friendly_name = speaker_map[speaker]
        timestamp = format_timestamp(seg["start"])
        lines.append(f"[{timestamp}] **{friendly_name}:** {seg['text']}")

    return "\n\n".join(lines)


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def summarize_with_speakers(transcript: str, num_speakers: int):
    """
    Summarize the meeting transcript using a local LLM (Ollama).
    The prompt leverages speaker information for better summaries.
    """
    print(f"Stage 3: Smart Summarization ({OLLAMA_MODEL})...")
    print()

    prompt = f"""/no_think
You are a professional meeting summarizer. You received a transcript of a meeting in Hebrew with {num_speakers} participants.
Each speaker is labeled (Speaker 1, Speaker 2, etc.).

The transcript is in Hebrew. Provide your summary in Hebrew.

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

## Open Issues
Topics raised but not resolved (if any)

---
Transcript:
{transcript}
---

Summarize professionally and clearly:"""

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
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    print(f"   Summarization complete in {elapsed:.1f}s")
    print()

    return summary


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

    # Truncate text if too long
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n\n[... Document truncated due to length ...]"

    return f"""{base}{instructions}

---
Document:
{text}
---

Summarize professionally and clearly:"""


def read_document(file_path: str) -> str:
    """Read a document file and return its text content."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in {".md", ".txt", ".rtf"}:
        return path.read_text(encoding="utf-8")

    elif ext == ".pdf":
        try:
            check_and_install("pdfplumber")
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            # Fallback to pdftotext CLI
            try:
                result = subprocess.run(
                    ["pdftotext", str(path), "-"],
                    capture_output=True, text=True, timeout=30
                )
                return result.stdout
            except Exception as e:
                raise RuntimeError(f"Cannot read PDF: {e}")

    elif ext in {".docx", ".doc"}:
        try:
            check_and_install("python-docx", "docx")
            import docx
            doc = docx.Document(str(path))
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            raise RuntimeError(f"Cannot read DOCX: {e}")

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
        ["ollama", "run", OLLAMA_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = time.time() - start_time
    summary = result.stdout.strip()

    # Remove thinking tags if present
    if "<think>" in summary:
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

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
def save_results(audio_path: str, segments: list, transcript: str, summary: str):
    """Save all results to organized output files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_path).stem

    # Count speakers
    unique_speakers = set(seg["speaker"] for seg in segments)
    num_speakers = len(unique_speakers)

    # Markdown output
    md_file = OUTPUT_DIR / f"{base_name}_{timestamp}.md"
    md_content = f"""# Meeting Summary — LocalScribe

**Processed:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Source file:** {Path(audio_path).name}
**Speakers:** {num_speakers}
**Duration:** {format_timestamp(segments[-1]['end'] if segments else 0)}

---

{summary}

---

## Full Transcript (with Speaker Identification)

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
        },
        "segments": segments,
        "transcript": transcript,
        "summary": summary,
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
    print("Meeting Summary:")
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

    # Format transcript with speaker labels
    transcript = format_transcript_with_speakers(transcribed_segments)
    unique_speakers = set(seg["speaker"] for seg in transcribed_segments)

    # Stage 3: Summarization
    summary = summarize_with_speakers(transcript, len(unique_speakers))

    # Save and display
    total_elapsed = time.time() - total_start

    print(f"Total processing time: {total_elapsed:.1f}s")
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
    print()

    choice = input("Choose (1-5): ").strip()

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

    else:
        print("[ERROR] Invalid choice")


if __name__ == "__main__":
    main()
