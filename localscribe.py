#!/usr/bin/env python3
"""
LocalScribe - Full Pipeline: Transcription + Speaker Diarization + Summarization
=================================================================================
תמלול פגישות בעברית עם זיהוי דוברים וסיכום חכם - הכל מקומי על המאק שלך.
כולל גם יכולת סיכום מסמכים טקסטואליים (רפואיים, משפטיים, עסקיים ועוד).

Pipeline (Audio):
1. Speaker Diarization (pyannote.audio) → מי דיבר ומתי
2. Hebrew ASR (mlx-whisper + ivrit.ai Turbo) → תמלול עברית מדויק
3. Summarization (Ollama + Qwen3) → סיכום + החלטות + Action Items

Pipeline (Document):
1. Read & Parse document (Markdown, TXT, PDF, DOCX)
2. Detect document type (medical, legal, meeting, report, etc.)
3. Summarization (Ollama + Qwen3) → סיכום מותאם לסוג המסמך

Requirements:
- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Ollama (for summarization)
- HuggingFace token (for pyannote, free) - only for audio mode

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
WHISPER_MODEL = "ivrit-ai/whisper-large-v3-turbo-d4"  # Best Hebrew ASR model (94-95% accuracy)
OLLAMA_MODEL = "qwen3:1.7b"  # Fast, good Hebrew support
OUTPUT_DIR = Path.home() / "LocalScribe_Output"
HF_TOKEN_PATH = Path.home() / ".localscribe_hf_token"

# Diarization settings
MIN_SPEAKERS = 2
MAX_SPEAKERS = 10
MIN_SEGMENT_DURATION = 0.5  # seconds - ignore very short segments

# Document settings
SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf", ".html"}
MAX_DOC_CHARS = 50000  # Maximum characters to send to LLM


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


def ensure_dependencies(mode: str = "audio"):
    """Verify all dependencies are available."""
    print("🔍 בודק תלויות...")

    hf_token = None

    if mode == "audio":
        # Core packages for audio mode
        check_and_install("mlx-whisper", "mlx_whisper")
        print("  ✅ mlx-whisper (תמלול)")

        check_and_install("torch", "torch")
        check_and_install("torchaudio", "torchaudio")
        check_and_install("pyannote.audio", "pyannote")
        print("  ✅ pyannote.audio (זיהוי דוברים)")

        check_and_install("pydub", "pydub")
        print("  ✅ pydub (עיבוד אודיו)")

        # Check HuggingFace token
        hf_token = get_hf_token()
        if not hf_token:
            hf_token = setup_hf_token()
            if not hf_token:
                print("  ❌ לא ניתן להמשיך ללא HuggingFace Token")
                sys.exit(1)
        print("  ✅ HuggingFace Token")

        # Check ffmpeg
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if result.returncode != 0:
            print("  ⚠️  ffmpeg לא מותקן - התקן עם: brew install ffmpeg")
        else:
            print("  ✅ ffmpeg")

    # Check Ollama (needed for both modes)
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

    print()
    return hf_token


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
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    print(f"   ✅ סיכום הושלם ב-{elapsed:.1f} שניות")
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

    # Medical indicators
    medical_keywords = [
        "מכתב שחרור", "הפניה רפואית", "אבחנה", "תרופות", "מטופל",
        "ניתוח", "אשפוז", "מרפאה", "בית חולים", "רופא", "מחלקה",
        "תופעות לוואי", "מינון", "בדיקות דם", "צילום רנטגן",
        "discharge", "medical", "clinical", "patient", "diagnosis",
    ]
    medical_score = sum(1 for kw in medical_keywords if kw in text_lower)

    # Legal indicators
    legal_keywords = [
        "חוזה", "הסכם", "שכירות", "משכיר", "שוכר", "ערבות",
        "סעיף", "תנאי", "עו\"ד", "חתימה", "פיצוי", "ביטול",
        "contract", "agreement", "legal", "clause", "liability",
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
        "medical": "רפואי",
        "legal": "משפטי",
        "meeting": "פרוטוקול פגישה",
        "report": "דוח",
        "proposal": "הצעת פרויקט",
        "hr": "מדיניות / משאבי אנוש",
        "general": "כללי",
    }

    type_label = type_labels.get(doc_type, "כללי")

    # Base prompt structure
    base = f"""/no_think
אתה עוזר מקצועי לסיכום מסמכים בעברית. קיבלת מסמך מסוג: **{type_label}**.

"""

    # Type-specific instructions
    if doc_type == "medical":
        instructions = """עליך לספק סיכום רפואי מובנה הכולל:

## סוג המסמך
(מכתב שחרור / הפניה / תוצאות בדיקה / אחר)

## סיכום קליני
3-5 משפטים שמסכמים את המצב הרפואי, האבחנה, והטיפול

## אבחנות
רשימת האבחנות שצוינו במסמך

## טיפולים ותרופות
רשימת התרופות, המינונים, ומשך הטיפול

## המלצות ומעקב
הנחיות למטופל, ביקורי מעקב, ובדיקות נדרשות

## דגשים חשובים
מתי לפנות לרופא, סימני אזהרה, הגבלות"""

    elif doc_type == "legal":
        instructions = """עליך לספק סיכום משפטי מובנה הכולל:

## סוג המסמך
(חוזה / הסכם / ייפוי כוח / אחר)

## צדדים
הצדדים להסכם ופרטיהם

## סיכום עיקרי
3-5 משפטים שמסכמים את עיקרי ההסכם

## תנאים עיקריים
הסעיפים והתנאים המרכזיים

## התחייבויות כספיות
סכומים, מועדי תשלום, ערבויות

## תאריכים חשובים
תקופת ההסכם, דדליינים, תנאי ביטול

## הערות ודגשים
סעיפים מיוחדים, סיכונים, נקודות לתשומת לב"""

    elif doc_type == "meeting":
        instructions = """עליך לספק סיכום פגישה מובנה הכולל:

## כותרת הפגישה
שם קצר וממוקד

## סיכום
3-5 משפטים שמסכמים את עיקרי הפגישה

## החלטות שהתקבלו
רשימה ממוספרת של כל ההחלטות

## משימות לביצוע (Action Items)
לכל משימה: אחראי, תיאור, דדליין

## נקודות פתוחות
נושאים שעלו אך לא הוכרעו

## הפגישה הבאה
תאריך ונושאים מתוכננים (אם צוין)"""

    elif doc_type == "report":
        instructions = """עליך לספק סיכום דוח מובנה הכולל:

## כותרת הדוח
שם קצר וממוקד

## תקציר מנהלים
3-5 משפטים שמסכמים את עיקרי הדוח

## נתונים מרכזיים
טבלה או רשימה של המספרים והמדדים החשובים

## מגמות ותובנות
ניתוח המגמות העיקריות שעולות מהנתונים

## אתגרים וסיכונים
בעיות שזוהו ודרכי התמודדות

## המלצות / תחזית
צעדים מומלצים או תחזית לתקופה הבאה"""

    elif doc_type == "proposal":
        instructions = """עליך לספק סיכום הצעה מובנה הכולל:

## שם הפרויקט
שם קצר וממוקד

## סיכום
3-5 משפטים שמסכמים את ההצעה

## מטרות ויעדים
מה הפרויקט אמור להשיג

## היקף ולוח זמנים
שלבים עיקריים ותאריכי יעד

## תקציב
סיכום העלויות הצפויות

## סיכונים
סיכונים עיקריים ודרכי התמודדות

## ROI / תועלות צפויות
ההחזר הצפוי על ההשקעה"""

    elif doc_type == "hr":
        instructions = """עליך לספק סיכום מדיניות מובנה הכולל:

## שם המדיניות
שם קצר וממוקד

## סיכום
3-5 משפטים שמסכמים את עיקרי המדיניות

## עקרונות מנחים
העקרונות המרכזיים שעליהם מבוססת המדיניות

## כללים ונהלים עיקריים
הכללים המרכזיים שהעובדים צריכים לדעת

## שינויים מהמדיניות הקודמת
מה חדש או שונה (אם רלוונטי)

## השפעה על העובדים
איך המדיניות משפיעה על העובדים בפועל"""

    else:  # general
        instructions = """עליך לספק סיכום מובנה הכולל:

## כותרת
שם קצר וממוקד למסמך

## סיכום
3-5 משפטים שמסכמים את עיקרי המסמך

## נקודות מרכזיות
הנקודות החשובות ביותר במסמך

## פרטים חשובים
מספרים, תאריכים, שמות, או נתונים חשובים

## מסקנות / המלצות
מסקנות או המלצות שעולות מהמסמך (אם יש)"""

    # Truncate text if too long
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS] + "\n\n[... המסמך קוצר בשל אורכו ...]"

    return f"""{base}{instructions}

---
המסמך:
{text}
---

סכם בעברית בצורה מקצועית ומסודרת:"""


def read_document(file_path: str) -> str:
    """Read a document file and return its text content."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in {".md", ".txt", ".rtf"}:
        return path.read_text(encoding="utf-8")

    elif ext == ".pdf":
        try:
            # Try pdfplumber first
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
            # Fallback to subprocess (pdftotext)
            try:
                result = subprocess.run(
                    ["pdftotext", str(path), "-"],
                    capture_output=True, text=True, timeout=30
                )
                return result.stdout
            except Exception as e:
                raise RuntimeError(f"לא ניתן לקרוא PDF: {e}")

    elif ext in {".docx", ".doc"}:
        try:
            check_and_install("python-docx", "docx")
            import docx
            doc = docx.Document(str(path))
            return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            raise RuntimeError(f"לא ניתן לקרוא DOCX: {e}")

    elif ext == ".html":
        try:
            from html.parser import HTMLParser
            html_content = path.read_text(encoding="utf-8")
            # Simple HTML to text
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
            raise RuntimeError(f"לא ניתן לקרוא HTML: {e}")

    else:
        # Try reading as plain text
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            raise RuntimeError(f"פורמט לא נתמך: {ext}")


def summarize_document(text: str, doc_type: str):
    """Summarize a document using local LLM (Ollama)."""
    print(f"🤖 מסכם מסמך ({OLLAMA_MODEL})...")
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

    print(f"   ✅ סיכום הושלם ב-{elapsed:.1f} שניות")
    print()

    return summary


def process_document(file_path: str):
    """
    Full document processing pipeline: Read → Detect Type → Summarize.
    """
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           LocalScribe - סיכום מסמך                          ║")
    print("║   📄 קריאה → 🔍 זיהוי סוג → 🤖 סיכום חכם                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"📂 קובץ: {file_path}")
    print()

    total_start = time.time()

    # Read document
    print("📄 שלב 1: קריאת המסמך...")
    try:
        text = read_document(file_path)
    except Exception as e:
        print(f"❌ שגיאה בקריאת הקובץ: {e}")
        return None

    if not text or len(text.strip()) < 50:
        print("❌ המסמך ריק או קצר מדי")
        return None

    print(f"   ✅ נקראו {len(text):,} תווים")
    print()

    # Detect document type
    print("🔍 שלב 2: זיהוי סוג המסמך...")
    doc_type = detect_document_type(text, file_path)
    type_labels = {
        "medical": "🏥 רפואי",
        "legal": "⚖️ משפטי",
        "meeting": "📋 פרוטוקול פגישה",
        "report": "📊 דוח",
        "proposal": "💡 הצעת פרויקט",
        "hr": "👥 מדיניות / משאבי אנוש",
        "general": "📄 כללי",
    }
    print(f"   ✅ סוג: {type_labels.get(doc_type, doc_type)}")
    print()

    # Summarize
    summary = summarize_document(text, doc_type)

    total_elapsed = time.time() - total_start

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(file_path).stem

    md_file = OUTPUT_DIR / f"doc_{base_name}_{timestamp}.md"
    md_content = f"""# סיכום מסמך - LocalScribe

**תאריך עיבוד:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
**קובץ מקור:** {Path(file_path).name}
**סוג מסמך:** {type_labels.get(doc_type, doc_type)}
**אורך מקורי:** {len(text):,} תווים

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

    print(f"⏱️  זמן עיבוד כולל: {total_elapsed:.1f} שניות")
    print()
    print(f"💾 תוצאות נשמרו:")
    print(f"   📄 Markdown: {md_file}")
    print(f"   📊 JSON:     {json_file}")
    print()

    # Display summary
    print("═" * 60)
    print(f"📋 סיכום המסמך ({type_labels.get(doc_type, doc_type)}):")
    print("═" * 60)
    print()
    print(summary)
    print()
    print("═" * 60)

    return md_file


def process_document_dir(dir_path: str):
    """Process all documents in a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        print(f"❌ התיקייה לא נמצאה: {dir_path}")
        return

    files = [f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_DOC_EXTENSIONS]
    if not files:
        print(f"❌ לא נמצאו מסמכים נתמכים בתיקייה: {dir_path}")
        return

    print(f"📂 נמצאו {len(files)} מסמכים בתיקייה")
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

    print(f"\n{'═' * 60}")
    print(f"✅ סוכמו {len(results)} מתוך {len(files)} מסמכים")
    print(f"📂 התוצאות נשמרו ב: {OUTPUT_DIR}")
    print(f"{'═' * 60}")


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
    print("║   🔒 100% מקומי | 🇮🇱 עברית | 🎭 זיהוי דוברים | 📄 מסמכים   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
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
            print("❌ נדרש נתיב לקובץ אחרי --document")
            return

    if "--document-dir" in sys.argv:
        idx = sys.argv.index("--document-dir")
        if idx + 1 < len(sys.argv):
            dir_path = sys.argv[idx + 1]
            ensure_dependencies(mode="document")
            process_document_dir(dir_path)
            return
        else:
            print("❌ נדרש נתיב לתיקייה אחרי --document-dir")
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
            print("שימוש:")
            print()
            print("  📢 מצב אודיו (תמלול + דוברים + סיכום):")
            print("  python3 localscribe.py <audio_file>          # עיבוד קובץ")
            print("  python3 localscribe.py --speakers 3 file.mp3 # ציון מספר דוברים")
            print("  python3 localscribe.py --record               # הקלטה ועיבוד")
            print()
            print("  📄 מצב מסמכים (סיכום חכם):")
            print("  python3 localscribe.py --document <file>      # סיכום מסמך בודד")
            print("  python3 localscribe.py --document-dir <folder> # סיכום כל המסמכים בתיקייה")
            print()
            print("  פורמטים נתמכים: .md .txt .pdf .docx .doc .rtf .html")
            print("  סוגי מסמכים: רפואי, משפטי, פרוטוקול פגישה, דוח, הצעת פרויקט, HR, כללי")
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
            print(f"❌ הקובץ לא נמצא: {audio_path}")
            sys.exit(1)

        hf_token = ensure_dependencies(mode="audio")
        process_audio(audio_path, hf_token, num_speakers)
        return

    # Interactive menu
    print("מה תרצה לעשות?")
    print()
    print("  1. 📂  לעבד קובץ אודיו (תמלול + דוברים + סיכום)")
    print("  2. 🎙️  להקליט פגישה חדשה ולעבד")
    print("  3. 📝  לתמלל בלבד (בלי זיהוי דוברים)")
    print("  4. 📄  לסכם מסמך טקסטואלי")
    print("  5. 📂  לסכם כל המסמכים בתיקייה")
    print()

    choice = input("בחר (1-5): ").strip()

    if choice == "1":
        hf_token = ensure_dependencies(mode="audio")
        print()
        audio_path = input("הכנס נתיב לקובץ אודיו: ").strip()
        if not os.path.exists(audio_path):
            print(f"❌ הקובץ לא נמצא: {audio_path}")
            return

        print()
        speakers_input = input("מספר דוברים (Enter לזיהוי אוטומטי): ").strip()
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

    elif choice == "4":
        ensure_dependencies(mode="document")
        print()
        doc_path = input("הכנס נתיב למסמך: ").strip()
        if not os.path.exists(doc_path):
            print(f"❌ הקובץ לא נמצא: {doc_path}")
            return
        process_document(doc_path)

    elif choice == "5":
        ensure_dependencies(mode="document")
        print()
        dir_path = input("הכנס נתיב לתיקייה: ").strip()
        process_document_dir(dir_path)

    else:
        print("❌ בחירה לא חוקית")


if __name__ == "__main__":
    main()
