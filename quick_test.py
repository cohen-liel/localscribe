#!/usr/bin/env python3
"""
Quick Test - בדיקה מהירה של כל הצינור
=======================================
בודק שכל הרכיבים עובדים: Ollama, pyannote, mlx-whisper.
מריץ סיכום על טקסט לדוגמה עם סימוני דוברים.

שימוש:
    source ~/.localscribe_env/bin/activate
    python3 quick_test.py
"""

import subprocess
import sys
import time
import os

OLLAMA_MODEL = "qwen3:1.7b"

# טקסט לדוגמה - סימולציה של תמלול פגישה עם זיהוי דוברים
SAMPLE_TRANSCRIPT_WITH_SPEAKERS = """
[00:00] **דובר 1:** שלום לכולם, תודה שהגעתם לפגישה. היום אנחנו צריכים לדבר על שלושה נושאים: ההשקה, התקציב, והגיוס.

[00:15] **דובר 1:** הנושא הראשון - ההשקה של המוצר החדש. אנחנו מתכננים להשיק בעוד שלושה שבועות, ב-25 למאי. יוסי, אתה אחראי על הצד הטכני - איפה אנחנו עומדים?

[00:32] **דובר 2:** אנחנו בשלבים אחרונים. הבאגים הקריטיים תוקנו, נשארו עוד שני באגים קטנים שנסגור עד סוף השבוע. אני צריך מדנה שתסיים את העיצוב של דף הנחיתה עד יום שלישי.

[00:51] **דובר 3:** בסדר, אני יכולה לסיים את זה. אני רק צריכה את הטקסטים הסופיים ממיכל.

[01:02] **דובר 4:** אני אשלח את הטקסטים מחר בבוקר.

[01:08] **דובר 1:** מצוין. הנושא השני הוא התקציב. אנחנו חורגים ב-15 אחוז מהתקציב המקורי. ההחלטה שלי היא שנקצץ את תקציב הפרסום בפייסבוק ונעביר את הכסף לגוגל, כי שם אנחנו רואים תשואה טובה יותר. אורי, אתה מטפל בזה?

[01:35] **דובר 5:** כן, אני אעדכן את הקמפיינים עד יום חמישי.

[01:42] **דובר 1:** הנושא השלישי - גיוס. אנחנו צריכים עוד מפתח פולסטאק. מיכל, את מתאמת את הראיונות. יש לנו שלושה מועמדים לשבוע הבא.

[01:58] **דובר 4:** נכון, הראיונות ביום שני ושלישי. אני אשלח לכולם את קורות החיים היום.

[02:10] **דובר 1:** אוקיי, אז לסיכום: השקה ב-25 למאי, דנה מסיימת עיצוב עד שלישי, מיכל שולחת טקסטים מחר, אורי מעדכן קמפיינים עד חמישי, וראיונות שבוע הבא. תודה לכולם!
"""


def check_component(name: str, check_fn) -> bool:
    """Check a single component."""
    try:
        result = check_fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False


def test_full_pipeline():
    """Test the full pipeline with sample data."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   LocalScribe v2.0 - בדיקה מהירה                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    all_ok = True

    # --- Component checks ---
    print("🔍 בודק רכיבים:")
    print()

    # Check Ollama
    def check_ollama():
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        assert result.returncode == 0, "Ollama לא רץ - הפעל: ollama serve"
        assert OLLAMA_MODEL.split(":")[0] in result.stdout, f"מודל {OLLAMA_MODEL} לא מותקן"
    all_ok &= check_component("Ollama + מודל סיכום", check_ollama)

    # Check mlx-whisper
    def check_whisper():
        import mlx_whisper  # noqa: F401
    all_ok &= check_component("mlx-whisper (תמלול)", check_whisper)

    # Check pyannote
    def check_pyannote():
        import pyannote.audio  # noqa: F401
    all_ok &= check_component("pyannote.audio (זיהוי דוברים)", check_pyannote)

    # Check torch
    def check_torch():
        import torch
        has_mps = torch.backends.mps.is_available()
        return has_mps
    all_ok &= check_component("PyTorch (Metal GPU)", check_torch)

    # Check pydub
    def check_pydub():
        from pydub import AudioSegment  # noqa: F401
    all_ok &= check_component("pydub (עיבוד אודיו)", check_pydub)

    # Check ffmpeg
    def check_ffmpeg():
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        assert result.returncode == 0
    all_ok &= check_component("ffmpeg", check_ffmpeg)

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
        raise Exception("לא נמצא - הריצי install.sh או הזיני ידנית")
    all_ok &= check_component("HuggingFace Token", check_hf_token)

    print()

    if not all_ok:
        print("⚠️  חלק מהרכיבים חסרים. הריצו: ./install.sh")
        print()
        return

    # --- Summarization test ---
    print("─" * 60)
    print("🧪 בדיקת סיכום עם טקסט לדוגמה (5 דוברים):")
    print("─" * 60)
    print()

    prompt = f"""/no_think
אתה עוזר מקצועי לסיכום פגישות בעברית. קיבלת תמלול של פגישה עם 5 משתתפים.
כל דובר מסומן (דובר 1, דובר 2, וכו').

עליך לספק:
## כותרת
## סיכום (3-5 משפטים)
## משימות לביצוע (מי, מה, עד מתי)
## החלטות שהתקבלו

התמלול:
---
{SAMPLE_TRANSCRIPT_WITH_SPEAKERS}
---

סכם בעברית:"""

    print(f"  🤖 מסכם עם {OLLAMA_MODEL}...")
    start_time = time.time()

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )

    elapsed = time.time() - start_time
    summary = result.stdout.strip()

    # Remove thinking tags if present
    if "<think>" in summary:
        import re
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

    print(f"  ✅ סיכום הושלם ב-{elapsed:.1f} שניות")
    print()
    print("═" * 60)
    print("📋 תוצאת הסיכום:")
    print("═" * 60)
    print()
    print(summary)
    print()
    print("═" * 60)
    print()
    print("🎉 הכל עובד! עכשיו אתה יכול:")
    print()
    print("   python3 localscribe.py recording.mp3      # עיבוד קובץ מלא")
    print("   python3 localscribe.py --record           # הקלטה ועיבוד")
    print()
    print("   הפלט יכלול: תמלול עברית + זיהוי מי אמר מה + סיכום חכם")
    print()


if __name__ == "__main__":
    test_full_pipeline()
