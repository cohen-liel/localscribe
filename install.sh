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

# Detect SSL MITM proxy (corp Macs sometimes intercept TLS — pip won't trust it)
PIP_SSL_FLAGS=""
if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
    echo "[INFO] HTTPS proxy detected — using --trusted-host for pip"
    PIP_SSL_FLAGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org"
fi
echo ""

# ============================================================
# Step 1: System Tools (Homebrew, Python, ffmpeg)
# ============================================================
echo "Step 1: System Tools"
echo "─────────────────────"

# Homebrew (optional — we have a pip fallback for ffmpeg below)
HAS_BREW=0
if command -v brew &> /dev/null && [ -w "$(brew --prefix)/Cellar" 2>/dev/null ]; then
    HAS_BREW=1
    echo "  [OK] Homebrew (writable)"
else
    echo "  [INFO] Homebrew unavailable or not writable — will use pip fallbacks"
fi

# Python
if ! command -v python3 &> /dev/null; then
    if [ "$HAS_BREW" = "1" ]; then
        echo "  Installing Python..."
        brew install python@3.11
    else
        echo "  ERROR: python3 not found and Homebrew is not available."
        echo "         Install Python 3.11+ manually: https://www.python.org/downloads/macos/"
        exit 1
    fi
fi
echo "  [OK] Python $(python3 --version 2>&1 | cut -d' ' -f2)"

# ffmpeg — try brew, fall back to pip static-ffmpeg later (after venv exists)
NEED_STATIC_FFMPEG=0
if ! command -v ffmpeg &> /dev/null; then
    if [ "$HAS_BREW" = "1" ]; then
        echo "  Installing ffmpeg via Homebrew..."
        brew install ffmpeg
    else
        echo "  [INFO] Will install ffmpeg via pip (static-ffmpeg) after venv setup"
        NEED_STATIC_FFMPEG=1
    fi
else
    echo "  [OK] ffmpeg"
fi

# sox — optional, only used by the legacy --record code path on some setups
if ! command -v sox &> /dev/null && [ "$HAS_BREW" = "1" ]; then
    echo "  Installing sox..."
    brew install sox || echo "  [WARN] sox install failed — recording may be limited"
fi

echo ""

# ============================================================
# Step 2: Ollama (Local LLM Engine for Summarization)
# ============================================================
echo "Step 2: Ollama (Summarization Engine)"
echo "──────────────────────────────────────"

if ! command -v ollama &> /dev/null; then
    if [ "$HAS_BREW" = "1" ]; then
        echo "  Installing Ollama..."
        brew install ollama
    else
        echo "  ERROR: ollama not found. Install from https://ollama.com/download"
        exit 1
    fi
fi
echo "  [OK] Ollama installed"

# Start Ollama in the background if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  Starting Ollama in the background..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Default summarization model — change OLLAMA_MODEL in localscribe.py if you prefer another
DEFAULT_OLLAMA_MODEL="${LOCALSCRIBE_OLLAMA_MODEL:-gemma4:e4b}"
echo "  Downloading summarization model ($DEFAULT_OLLAMA_MODEL)..."
ollama pull "$DEFAULT_OLLAMA_MODEL"
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
pip install $PIP_SSL_FLAGS --upgrade pip -q

# Persist trusted-hosts to pip config so subsequent installs don't need flags
if [ -n "$PIP_SSL_FLAGS" ]; then
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" >/dev/null 2>&1 || true
fi

# Install pinned dependencies (see requirements.txt for why versions are pinned)
pip install $PIP_SSL_FLAGS -r requirements.txt -q
echo "  [OK] Core Python packages installed"

# ffmpeg fallback: install static-ffmpeg via pip and symlink into ~/.local/bin
if [ "$NEED_STATIC_FFMPEG" = "1" ] || ! command -v ffmpeg &> /dev/null; then
    echo "  Installing static-ffmpeg via pip (Homebrew not available)..."
    pip install $PIP_SSL_FLAGS static-ffmpeg -q
    python3 -c "import static_ffmpeg; static_ffmpeg.add_paths()" >/dev/null 2>&1 || true
    mkdir -p "$HOME/.local/bin"
    STATIC_FF_DIR="$(python3 -c 'import os, static_ffmpeg; print(os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin", "darwin_arm64"))')"
    if [ -x "$STATIC_FF_DIR/ffmpeg" ]; then
        ln -sf "$STATIC_FF_DIR/ffmpeg"  "$HOME/.local/bin/ffmpeg"
        ln -sf "$STATIC_FF_DIR/ffprobe" "$HOME/.local/bin/ffprobe"
        echo "  [OK] ffmpeg + ffprobe symlinked into ~/.local/bin"
        case ":$PATH:" in
            *":$HOME/.local/bin:"*) ;;
            *) echo "  [INFO] Add ~/.local/bin to your PATH (e.g. in ~/.zshrc)";;
        esac
    fi
fi

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
echo "  Output is saved to: ./output/ (relative to localscribe.py)"
echo ""
