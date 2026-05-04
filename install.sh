#!/bin/bash
# ============================================================
# LocalScribe v2.0 — Installation Script for Mac (Apple Silicon)
# ============================================================
# Installs everything needed: Hebrew transcription + speaker
# diarization + summarization — all running 100% locally.
# ============================================================

set -euo pipefail

fail() {
    echo ""
    echo "  [ERROR] $1"
    shift || true
    for line in "$@"; do
        echo "          $line"
    done
    echo ""
    exit 1
}

require_command() {
    local command_name="$1"
    local install_hint="$2"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        fail "$command_name is required but was not found." "$install_hint"
    fi
}

brew_install_if_missing() {
    local command_name="$1"
    local package_name="$2"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "  Installing $package_name..."
        if ! brew install "$package_name"; then
            fail "Homebrew failed to install $package_name." \
                "Fix Homebrew, then rerun this script." \
                "Run: brew doctor" \
                "If Homebrew reports unwritable directories, run:" \
                "  sudo chown -R \"$(whoami)\" \"$(brew --prefix)\"" \
                "  sudo chmod -R u+w \"$(brew --prefix)\""
        fi
        hash -r 2>/dev/null || true
    fi
}

reject_static_ffmpeg_tool() {
    local command_name="$1"
    local command_path
    local real_path

    command_path="$(command -v "$command_name" 2>/dev/null || true)"
    if [ -z "$command_path" ]; then
        return
    fi

    real_path="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$command_path")"
    case "$real_path" in
        *".localscribe_env"*|*"static_ffmpeg"*)
            fail "$command_name points to an old static-ffmpeg install." \
                "$command_path -> $real_path" \
                "Remove it with: rm -f \"$HOME/.local/bin/$command_name\"" \
                "Then install the system package with: brew install ffmpeg"
            ;;
    esac
}

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

# Detect SSL MITM proxy, but do not bypass TLS verification.
if [ -n "${HTTPS_PROXY:-}" ] || [ -n "${https_proxy:-}" ]; then
    echo "[INFO] HTTPS proxy detected — pip will still use normal certificate verification"
fi

# Some managed Python builds point OpenSSL at a stale or private CA bundle.
# Prefer the macOS system CA bundle when the user has not explicitly selected one.
if [ -z "${PIP_CERT:-}" ] && [ -z "${SSL_CERT_FILE:-}" ] && [ -r "/etc/ssl/cert.pem" ]; then
    export PIP_CERT="/etc/ssl/cert.pem"
    echo "[INFO] Using system CA bundle for pip: $PIP_CERT"
fi
echo ""

# ============================================================
# Step 1: System Tools (Homebrew, Python, ffmpeg)
# ============================================================
echo "Step 1: System Tools"
echo "─────────────────────"

# Homebrew is required. The installer should not hide a broken system package manager
# by installing private binary copies inside the Python environment.
require_command "brew" "Install Homebrew: https://brew.sh"

BREW_PREFIX="$(brew --prefix 2>/dev/null || true)"
if [ -z "$BREW_PREFIX" ]; then
    fail "Homebrew is installed but brew --prefix failed." "Run: brew doctor"
fi
echo "  [OK] Homebrew ($BREW_PREFIX)"

# Python
if ! command -v python3 &> /dev/null; then
    echo "  Installing Python..."
    brew install python@3.12 || brew install python@3.11
    hash -r 2>/dev/null || true
fi
echo "  [OK] Python $(python3 --version 2>&1 | cut -d' ' -f2)"

reject_static_ffmpeg_tool "ffmpeg"
reject_static_ffmpeg_tool "ffprobe"

brew_install_if_missing "ffmpeg" "ffmpeg"
brew_install_if_missing "ffprobe" "ffmpeg"

reject_static_ffmpeg_tool "ffmpeg"
reject_static_ffmpeg_tool "ffprobe"
require_command "ffmpeg" "Install with: brew install ffmpeg"
require_command "ffprobe" "Install with: brew install ffmpeg"
echo "  [OK] ffmpeg + ffprobe"

echo ""

# ============================================================
# Step 2: Ollama (Local LLM Engine for Summarization)
# ============================================================
echo "Step 2: Ollama (Summarization Engine)"
echo "──────────────────────────────────────"

if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    brew install ollama
    hash -r 2>/dev/null || true
fi
require_command "ollama" "Install with: brew install ollama"
echo "  [OK] Ollama installed"

# Start Ollama in the background if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "  Starting Ollama in the background..."
    ollama serve &>/dev/null &
    sleep 3
fi

# Default summarization model — change OLLAMA_MODEL in localscribe.py if you prefer another
DEFAULT_OLLAMA_MODEL="${LOCALSCRIBE_OLLAMA_MODEL:-gemma4:e4b}"
if ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$DEFAULT_OLLAMA_MODEL"; then
    echo "  [OK] Summarization model already installed ($DEFAULT_OLLAMA_MODEL)"
else
    echo "  Downloading summarization model ($DEFAULT_OLLAMA_MODEL)..."
    ollama pull "$DEFAULT_OLLAMA_MODEL"
    echo "  [OK] Summarization model ready"
fi

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

if [ -n "${PIP_CERT:-}" ]; then
    pip config --site set global.cert "$PIP_CERT" >/dev/null 2>&1 || true
fi

# Install pinned dependencies (see requirements.txt for why versions are pinned)
if ! pip install -r requirements.txt -q; then
    fail "Python dependency installation failed." \
        "If this is a TLS/certificate issue, install your organization's CA certificate." \
        "Then configure pip with PIP_CERT or: pip config set global.cert /path/to/ca.pem" \
        "Do not bypass certificate verification with --trusted-host."
fi
echo "  [OK] Core Python packages installed"

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
