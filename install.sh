#!/bin/bash
# ============================================================
# LocalScribe - סקריפט התקנה למאק (Apple Silicon)
# ============================================================
# מתקין את כל מה שצריך כדי להריץ תמלול וסיכום מקומי
# ============================================================

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   LocalScribe - התקנה                        ║"
echo "║   תמלול וסיכום פגישות מקומי לMac             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# בדיקה שזה Mac עם Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ הסקריפט הזה מיועד ל-Mac עם Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi
echo "✅ Mac עם Apple Silicon מזוהה"

# בדיקה/התקנה של Homebrew
if ! command -v brew &> /dev/null; then
    echo "⬇️  מתקין Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo "✅ Homebrew מותקן"

# בדיקה/התקנה של Python
if ! command -v python3 &> /dev/null; then
    echo "⬇️  מתקין Python..."
    brew install python@3.11
fi
echo "✅ Python מותקן ($(python3 --version))"

# בדיקה/התקנה של ffmpeg (נדרש לעיבוד אודיו)
if ! command -v ffmpeg &> /dev/null; then
    echo "⬇️  מתקין ffmpeg..."
    brew install ffmpeg
fi
echo "✅ ffmpeg מותקן"

# בדיקה/התקנה של Ollama
if ! command -v ollama &> /dev/null; then
    echo "⬇️  מתקין Ollama..."
    brew install ollama
fi
echo "✅ Ollama מותקן"

# יצירת סביבה וירטואלית
echo ""
echo "📦 מתקין חבילות Python..."
python3 -m venv ~/.localscribe_env
source ~/.localscribe_env/bin/activate

pip install --upgrade pip
pip install mlx-whisper

echo "✅ mlx-whisper מותקן"

# הורדת מודל סיכום
echo ""
echo "⬇️  מוריד מודל סיכום (qwen3:1.7b)..."
echo "   (זה יכול לקחת כמה דקות בפעם הראשונה)"

# הפעלת Ollama ברקע אם הוא לא רץ
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 3
fi

ollama pull qwen3:1.7b

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ ההתקנה הושלמה בהצלחה!"
echo "═══════════════════════════════════════════════"
echo ""
echo "🚀 כדי להריץ:"
echo "   source ~/.localscribe_env/bin/activate"
echo "   python3 transcribe_and_summarize.py"
echo ""
echo "📂 או לתמלל קובץ ישירות:"
echo "   python3 transcribe_and_summarize.py /path/to/audio.mp3"
echo ""
echo "💡 טיפ: המודל Whisper יירד בפעם הראשונה שתריץ (~3GB)"
echo "   אחרי זה הכל עובד אופליין!"
echo ""
