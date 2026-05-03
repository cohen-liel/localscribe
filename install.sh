#!/bin/bash
# ============================================================
# LocalScribe v2.0 - סקריפט התקנה למאק (Apple Silicon)
# ============================================================
# מתקין את כל מה שצריך: תמלול עברית + זיהוי דוברים + סיכום
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   LocalScribe v2.0 - התקנה                                  ║"
echo "║   תמלול + זיהוי דוברים + סיכום פגישות (100% מקומי)          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# בדיקה שזה Mac עם Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ הסקריפט הזה מיועד ל-Mac עם Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi
echo "✅ Mac עם Apple Silicon מזוהה ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon'))"
echo ""

# ============================================================
# שלב 1: כלי מערכת (Homebrew, Python, ffmpeg)
# ============================================================
echo "📦 שלב 1: כלי מערכת"
echo "─────────────────────"

# Homebrew
if ! command -v brew &> /dev/null; then
    echo "⬇️  מתקין Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo "  ✅ Homebrew"

# Python
if ! command -v python3 &> /dev/null; then
    echo "  ⬇️  מתקין Python..."
    brew install python@3.11
fi
echo "  ✅ Python $(python3 --version 2>&1 | cut -d' ' -f2)"

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "  ⬇️  מתקין ffmpeg..."
    brew install ffmpeg
fi
echo "  ✅ ffmpeg"

# sox (for recording)
if ! command -v sox &> /dev/null; then
    echo "  ⬇️  מתקין sox (להקלטה)..."
    brew install sox
fi
echo "  ✅ sox"

echo ""

# ============================================================
# שלב 2: Ollama (מנוע AI לסיכום)
# ============================================================
echo "🤖 שלב 2: Ollama (מנוע סיכום)"
echo "─────────────────────────────────"

if ! command -v ollama &> /dev/null; then
    echo "  ⬇️  מתקין Ollama..."
    brew install ollama
fi
echo "  ✅ Ollama מותקן"

# הפעלת Ollama ברקע אם לא רץ
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  🔄 מפעיל Ollama ברקע..."
    ollama serve &>/dev/null &
    sleep 3
fi

# הורדת מודל סיכום
echo "  ⬇️  מוריד מודל סיכום (qwen3:1.7b, ~1.7GB)..."
ollama pull qwen3:1.7b
echo "  ✅ מודל סיכום מוכן"

echo ""

# ============================================================
# שלב 3: סביבת Python וחבילות
# ============================================================
echo "🐍 שלב 3: חבילות Python"
echo "──────────────────────────"

# יצירת סביבה וירטואלית
if [ ! -d "$HOME/.localscribe_env" ]; then
    echo "  📦 יוצר סביבה וירטואלית..."
    python3 -m venv ~/.localscribe_env
fi
source ~/.localscribe_env/bin/activate

echo "  ⬇️  מתקין חבילות (זה יכול לקחת כמה דקות)..."

# Upgrade pip
pip install --upgrade pip -q

# Core packages
pip install mlx-whisper -q
echo "  ✅ mlx-whisper (תמלול על Apple Silicon)"

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
echo "  ✅ PyTorch (מנוע חישוב)"

pip install pyannote.audio -q
echo "  ✅ pyannote.audio (זיהוי דוברים)"

pip install pydub -q
echo "  ✅ pydub (עיבוד אודיו)"

echo ""

# ============================================================
# שלב 4: HuggingFace Token
# ============================================================
echo "🔑 שלב 4: HuggingFace Token"
echo "─────────────────────────────"

HF_TOKEN_FILE="$HOME/.localscribe_hf_token"

if [ -f "$HF_TOKEN_FILE" ]; then
    echo "  ✅ Token כבר קיים"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "  ✅ Token נמצא (huggingface-cli login)"
else
    echo ""
    echo "  ⚠️  נדרש HuggingFace Token (חינמי) עבור מודל זיהוי הדוברים."
    echo ""
    echo "  📋 הוראות:"
    echo "  1. היכנס ל: https://huggingface.co/settings/tokens"
    echo "  2. צור token חדש (Read access מספיק)"
    echo "  3. קבל את תנאי השימוש:"
    echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "     https://huggingface.co/pyannote/segmentation-3.0"
    echo ""
    read -p "  הדבק את ה-Token כאן (או Enter לדלג): " HF_TOKEN
    
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" > "$HF_TOKEN_FILE"
        chmod 600 "$HF_TOKEN_FILE"
        echo "  ✅ Token נשמר!"
    else
        echo "  ⚠️  תצטרך להזין Token בהפעלה הראשונה"
    fi
fi

echo ""

# ============================================================
# סיום
# ============================================================
echo "═══════════════════════════════════════════════════════════════"
echo "✅ ההתקנה הושלמה בהצלחה!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🚀 כדי להריץ:"
echo ""
echo "   source ~/.localscribe_env/bin/activate"
echo "   python3 localscribe.py recording.mp3"
echo ""
echo "📋 אפשרויות:"
echo "   python3 localscribe.py <file>          # עיבוד קובץ"
echo "   python3 localscribe.py --record        # הקלטה ועיבוד"
echo "   python3 localscribe.py --speakers 3 f  # ציון מספר דוברים"
echo "   python3 localscribe.py                 # תפריט אינטראקטיבי"
echo ""
echo "💡 בפעם הראשונה, המודלים יירדו אוטומטית (~5GB סה\"כ)."
echo "   אחרי זה הכל עובד אופליין!"
echo ""
echo "📂 התוצאות יישמרו ב: ~/LocalScribe_Output/"
echo ""
