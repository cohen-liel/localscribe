#!/bin/bash
# ============================================================
# LocalScribe v2.0 — Installation Script for Mac (Apple Silicon)
# ============================================================
# Installs everything needed: Hebrew transcription + speaker
# diarization + summarization — all running 100% locally.
# ============================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   LocalScribe v2.0 — Installation                          ║"
echo "║   Transcription + Speaker Diarization + Summarization       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Verify Apple Silicon Mac
if [[ $(uname -m) != "arm64" ]]; then
    echo "ERROR: This script requires a Mac with Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi
echo "[OK] Apple Silicon Mac detected ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon'))"
echo ""

# ============================================================
# Step 1: System Tools (Homebrew, Python, ffmpeg)
# ============================================================
echo "Step 1: System Tools"
echo "─────────────────────"

# Homebrew
if ! command -v brew &> /dev/null; then
    echo "  Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo "  [OK] Homebrew"

# Python
if ! command -v python3 &> /dev/null; then
    echo "  Installing Python..."
    brew install python@3.11
fi
echo "  [OK] Python $(python3 --version 2>&1 | cut -d' ' -f2)"

# ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "  Installing ffmpeg..."
    brew install ffmpeg
fi
echo "  [OK] ffmpeg"

# sox (for recording)
if ! command -v sox &> /dev/null; then
    echo "  Installing sox (for audio recording)..."
    brew install sox
fi
echo "  [OK] sox"

echo ""

# ============================================================
# Step 2: Ollama (Local LLM Engine for Summarization)
# ============================================================
echo "Step 2: Ollama (Summarization Engine)"
echo "──────────────────────────────────────"

if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    brew install ollama
fi
echo "  [OK] Ollama installed"

# Start Ollama in the background if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  Starting Ollama in the background..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Download summarization model
echo "  Downloading summarization model (qwen3:1.7b, ~1.7GB)..."
ollama pull qwen3:1.7b
echo "  [OK] Summarization model ready"

echo ""

# ============================================================
# Step 3: Python Environment & Packages
# ============================================================
echo "Step 3: Python Packages"
echo "────────────────────────"

# Create virtual environment
if [ ! -d "$HOME/.localscribe_env" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv ~/.localscribe_env
fi
source ~/.localscribe_env/bin/activate

echo "  Installing packages (this may take a few minutes)..."

# Upgrade pip
pip install --upgrade pip -q

# Core packages
pip install mlx-whisper -q
echo "  [OK] mlx-whisper (transcription on Apple Silicon)"

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q
echo "  [OK] PyTorch (computation engine)"

pip install pyannote.audio -q
echo "  [OK] pyannote.audio (speaker diarization)"

pip install pydub -q
echo "  [OK] pydub (audio processing)"

pip install pdfplumber python-docx -q
echo "  [OK] pdfplumber + python-docx (document parsing)"

echo ""

# ============================================================
# Step 4: HuggingFace Token
# ============================================================
echo "Step 4: HuggingFace Token"
echo "──────────────────────────"

HF_TOKEN_FILE="$HOME/.localscribe_hf_token"

if [ -f "$HF_TOKEN_FILE" ]; then
    echo "  [OK] Token already exists"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "  [OK] Token found (huggingface-cli login)"
else
    echo ""
    echo "  A free HuggingFace Token is required for the speaker diarization model."
    echo ""
    echo "  Instructions:"
    echo "  1. Go to: https://huggingface.co/settings/tokens"
    echo "  2. Create a new token (Read access is sufficient)"
    echo "  3. Accept the model license terms:"
    echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "     https://huggingface.co/pyannote/segmentation-3.0"
    echo ""
    read -p "  Paste your token here (or press Enter to skip): " HF_TOKEN
    
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" > "$HF_TOKEN_FILE"
        chmod 600 "$HF_TOKEN_FILE"
        echo "  [OK] Token saved!"
    else
        echo "  [WARN] You will need to enter the token on first run"
    fi
fi

echo ""

# ============================================================
# Done
# ============================================================
echo "═══════════════════════════════════════════════════════════════"
echo "  Installation complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  To get started:"
echo ""
echo "   source ~/.localscribe_env/bin/activate"
echo "   python3 localscribe.py recording.mp3"
echo ""
echo "  Options:"
echo "   python3 localscribe.py <file>              # Process audio file"
echo "   python3 localscribe.py --record            # Record and process"
echo "   python3 localscribe.py --speakers 3 f.mp3  # Specify speaker count"
echo "   python3 localscribe.py --document f.pdf    # Summarize a document"
echo "   python3 localscribe.py                     # Interactive menu"
echo ""
echo "  On first run, models will be downloaded automatically (~5GB total)."
echo "  After that, everything works fully offline!"
echo ""
echo "  Output is saved to: ~/LocalScribe_Output/"
echo ""
