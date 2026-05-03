#!/usr/bin/env python3
"""
LocalScribe - תמלול וסיכום פגישות מקומי (100% אופליין)
=======================================================
סקריפט שמקליט/מתמלל פגישה בעברית ומסכם אותה - הכל על המאק שלך.

דרישות:
- Mac עם Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Ollama מותקן (להרצת מודל סיכום)
- mlx-whisper (לתמלול מהיר על Apple Silicon)

התקנה:
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
# הגדרות
# ============================================================
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # מודל תמלול - תומך עברית מעולה
OLLAMA_MODEL = "qwen3:1.7b"  # מודל סיכום - קטן ומהיר, עובד מעולה על M4
OUTPUT_DIR = Path.home() / "LocalScribe_Output"


def ensure_dependencies():
    """בדיקה שכל התלויות מותקנות"""
    print("🔍 בודק תלויות...")
    
    # בדיקת mlx-whisper
    try:
        import mlx_whisper
        print("  ✅ mlx-whisper מותקן")
    except ImportError:
        print("  ❌ mlx-whisper לא מותקן. מתקין...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx-whisper"], check=True)
        print("  ✅ mlx-whisper הותקן בהצלחה")
    
    # בדיקת Ollama
    result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
    if result.returncode != 0:
        print("  ❌ Ollama לא מותקן!")
        print("     התקן עם: brew install ollama")
        print("     או הורד מ: https://ollama.com/download")
        sys.exit(1)
    print("  ✅ Ollama מותקן")
    
    # בדיקה שהמודל קיים
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL.split(":")[0] not in result.stdout:
        print(f"  ⬇️  מוריד מודל {OLLAMA_MODEL}...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
    print(f"  ✅ מודל {OLLAMA_MODEL} מוכן")
    
    print()


def record_audio(duration_seconds=None):
    """
    הקלטת אודיו מהמיקרופון.
    אם duration_seconds=None, ההקלטה תמשיך עד שתלחץ Ctrl+C.
    """
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
        cmd = ["rec", "-r", "16000", "-c", "1", "-b", "16", str(output_file)]
        if duration_seconds:
            cmd.extend(["trim", "0", str(duration_seconds)])
        
        # ננסה עם sox/rec, אם לא קיים ננסה עם ffmpeg
        try:
            process = subprocess.run(cmd, check=True)
        except FileNotFoundError:
            # חלופה: שימוש ב-ffmpeg
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
        print("\n⏹️  הקלטה הופסקה")
    
    if output_file.exists():
        print(f"✅ הקלטה נשמרה: {output_file}")
        return str(output_file)
    return None


def transcribe_audio(audio_path):
    """
    תמלול קובץ אודיו עם MLX Whisper (מהיר מאוד על Apple Silicon).
    תומך בעברית!
    """
    import mlx_whisper
    
    print(f"📝 מתמלל את הקובץ: {Path(audio_path).name}")
    print(f"   (משתמש במודל: {WHISPER_MODEL})")
    print("   זה יכול לקחת כמה דקות בפעם הראשונה (הורדת המודל)...")
    print()
    
    start_time = time.time()
    
    # תמלול עם MLX Whisper - אופטימלי ל-Apple Silicon
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="he",  # עברית!
        task="transcribe",
        word_timestamps=True,
    )
    
    elapsed = time.time() - start_time
    transcript = result["text"]
    
    print(f"✅ תמלול הושלם ב-{elapsed:.1f} שניות!")
    print(f"   ({len(transcript.split())} מילים)")
    print()
    print("─" * 50)
    print("📄 התמלול:")
    print("─" * 50)
    print(transcript)
    print("─" * 50)
    print()
    
    return transcript


def summarize_text(transcript):
    """
    סיכום הטקסט עם מודל שפה מקומי (Ollama).
    """
    print(f"🤖 מסכם את הפגישה עם {OLLAMA_MODEL}...")
    print()
    
    prompt = f"""אתה עוזר שמסכם פגישות. קיבלת תמלול של פגישה בעברית.
עליך לספק:
1. **כותרת** - שם קצר לפגישה (שורה אחת)
2. **סיכום** - 3-5 משפטים שמסכמים את עיקרי הפגישה
3. **משימות לביצוע (Action Items)** - רשימה ממוספרת של דברים שצריך לעשות
4. **החלטות שהתקבלו** - אם יש

התמלול:
---
{transcript}
---

סכם בעברית:"""

    # קריאה ל-Ollama
    start_time = time.time()
    
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    elapsed = time.time() - start_time
    summary = result.stdout.strip()
    
    print(f"✅ סיכום הושלם ב-{elapsed:.1f} שניות!")
    print()
    print("═" * 50)
    print("📋 סיכום הפגישה:")
    print("═" * 50)
    print(summary)
    print("═" * 50)
    print()
    
    return summary


def save_results(audio_path, transcript, summary):
    """שמירת התוצאות לקובץ"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # שמירה כ-Markdown
    md_file = OUTPUT_DIR / f"meeting_{timestamp}.md"
    
    content = f"""# סיכום פגישה
**תאריך:** {datetime.now().strftime("%d/%m/%Y %H:%M")}  
**קובץ מקור:** {Path(audio_path).name}

---

## סיכום

{summary}

---

## תמלול מלא

{transcript}
"""
    
    md_file.write_text(content, encoding="utf-8")
    print(f"💾 התוצאות נשמרו ב: {md_file}")
    
    # שמירה גם כ-JSON (לשימוש עתידי באפליקציה)
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
    """עיבוד קובץ אודיו קיים"""
    if not os.path.exists(audio_path):
        print(f"❌ הקובץ לא נמצא: {audio_path}")
        sys.exit(1)
    
    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript)
    result_file = save_results(audio_path, transcript, summary)
    
    return result_file


def main():
    """תפריט ראשי"""
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   LocalScribe - תמלול וסיכום פגישות מקומי   ║")
    print("║   🔒 100% אופליין | 🇮🇱 תמיכה בעברית        ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    
    ensure_dependencies()
    
    # בדיקה אם קיבלנו קובץ כפרמטר
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"📂 מעבד קובץ: {audio_path}")
        process_existing_file(audio_path)
        return
    
    # תפריט אינטראקטיבי
    print("מה תרצה לעשות?")
    print("  1. 🎙️  להקליט פגישה חדשה ולסכם")
    print("  2. 📂  לתמלל ולסכם קובץ אודיו קיים")
    print("  3. 📝  לסכם טקסט שכבר תומלל")
    print()
    
    choice = input("בחר (1/2/3): ").strip()
    
    if choice == "1":
        print()
        audio_path = record_audio()
        if audio_path:
            transcript = transcribe_audio(audio_path)
            summary = summarize_text(transcript)
            save_results(audio_path, transcript, summary)
    
    elif choice == "2":
        print()
        audio_path = input("הכנס נתיב לקובץ אודיו: ").strip()
        process_existing_file(audio_path)
    
    elif choice == "3":
        print()
        print("הדבק את הטקסט (לחץ Enter פעמיים כשסיימת):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
            lines.append(line)
        transcript = "\n".join(lines[:-1])  # הסר שורה ריקה אחרונה
        summary = summarize_text(transcript)
        save_results("manual_input", transcript, summary)
    
    else:
        print("❌ בחירה לא חוקית")


if __name__ == "__main__":
    main()
