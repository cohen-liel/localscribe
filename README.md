# LocalScribe v2.0 — Local Meeting Transcription + Speaker Diarization + Smart Summarization

**Transcribe Hebrew meetings with "who said what" speaker identification + intelligent document summarization — 100% local on your Mac.**

No cloud. No API keys for transcription. No data leaves your machine. Ever.

---

## What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Hebrew Transcription | Whisper Large V3 | **ivrit.ai Turbo (MLX)** (94–95% accuracy) |
| Speaker Diarization | — | **pyannote 3.x** (who said what) |
| Summarization | Qwen3 1.7B | **Gemma 4 e4b** (speaker-aware, configurable) |
| Action Items | Basic | **Assigned to specific speakers** |
| Apple Metal GPU | — | **Accelerated diarization** |
| Document Summarization | — | **Medical, legal, business, HR, and more** |
| Test Data | — | **Audio samples + sample documents included** |

---

## How It Works

### Audio Pipeline

```
🎙️  Recording / Audio File
        │
        ▼
┌──────────────────────────────┐
│  Stage 1: Speaker Diarization │  pyannote.audio 3.1 (Apple Metal GPU)
│  "Who spoke and when?"        │  → Speaker 1: 00:00–00:15, Speaker 2: 00:15–00:32 ...
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 2: Hebrew Transcription│  ivrit.ai Turbo (mlx-whisper, Apple ANE)
│  "What was said?"             │  → Accurate Hebrew text per segment
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 3: Smart Summarization │  Qwen3 1.7B (Ollama, fully local)
│  "What matters?"              │  → Summary + Decisions + Action Items
└──────────────────────────────┘
        │
        ▼
📄  Structured Markdown + JSON output
```

### Document Pipeline

```
📄  Document (PDF / DOCX / Markdown / TXT / HTML)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1: Read & Parse        │  Supports PDF, DOCX, MD, TXT, RTF, HTML
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 2: Auto-Detect Type    │  Medical | Legal | Meeting | Report | Proposal | HR | General
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 3: Smart Summarization │  Type-specific prompt → tailored summary
└──────────────────────────────┘
        │
        ▼
📄  Structured Markdown + JSON output
```

---

## Installation (10 minutes)

### Prerequisites

- Mac with Apple Silicon (M1 / M2 / M3 / M4)
- macOS 13+ (Ventura or later)
- ~8 GB free disk space (for models)
- Free HuggingFace account (for speaker diarization model)

### Automatic Installation

```bash
git clone https://github.com/cohen-liel/localscribe.git
cd localscribe
chmod +x install.sh
./install.sh
```

### Manual Installation

```bash
# 1. System tools
brew install ffmpeg sox ollama
# If you can't use Homebrew, install python+ollama manually and rely on the
# pip-based static-ffmpeg fallback (install.sh handles this automatically).

# 2. Summarization model — pick any Ollama model you like
ollama serve &
ollama pull gemma4:e4b      # default; alternatives: qwen3:4b, gemma3:4b, qwen3:1.7b

# 3. Python environment
python3 -m venv ~/.localscribe_env
source ~/.localscribe_env/bin/activate
pip install -r requirements.txt
# On Macs with a corp SSL MITM proxy add:
#   --trusted-host pypi.org --trusted-host files.pythonhosted.org

# 4. HuggingFace Token (free, required for speaker diarization)
#    Create token: https://huggingface.co/settings/tokens
#    Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
#                  https://huggingface.co/pyannote/segmentation-3.0
#    Save:        echo "hf_xxx" > ~/.localscribe_hf_token && chmod 600 ~/.localscribe_hf_token
```

---

## Usage

### Audio Mode (Transcription + Diarization + Summarization)

```bash
source ~/.localscribe_env/bin/activate

# Process an audio file (full pipeline)
python3 localscribe.py meeting.mp3

# Record a meeting and process it
python3 localscribe.py --record

# Specify a known number of speakers (improves accuracy)
python3 localscribe.py meeting.mp3 --speakers 3

# Interactive menu
python3 localscribe.py
```

### Document Mode (Smart Summarization)

```bash
# Summarize a single document
python3 localscribe.py --document report.pdf

# Summarize all documents in a folder
python3 localscribe.py --document-dir ./documents/
```

### Quick Test (no recording needed)

```bash
python3 quick_test.py
```

### Supported Audio Formats

mp3, wav, m4a, mp4, webm, ogg, flac, aac

### Supported Document Formats

md, txt, pdf, docx, doc, rtf, html

### Auto-Detected Document Types

| Icon | Type | Tailored Summary Includes |
|------|------|--------------------------|
| 🏥 | Medical | Diagnoses, medications, follow-up instructions |
| ⚖️ | Legal | Parties, key clauses, financial obligations, deadlines |
| 📋 | Meeting | Decisions, action items, open issues |
| 📊 | Report | Key metrics, trends, risks, recommendations |
| 💡 | Proposal | Objectives, timeline, budget, ROI |
| 👥 | HR / Policy | Principles, rules, employee impact |
| 📄 | General | Key points, important details, conclusions |

---

## Example Output

Given a 30-minute team meeting with 4 participants:

```markdown
# Meeting Summary — LocalScribe

**Date:** 2026-05-03 14:30
**Speakers:** 4
**Duration:** 32:15

---

## Title
Status Meeting — New Product Launch

## Summary
The team discussed the product launch scheduled for May 25. The technical side
is nearly complete with two minor bugs remaining. There is a 15% budget overrun
that will be addressed by reallocating spend from Facebook to Google Ads.
Hiring for a full-stack developer position begins next week.

## Action Items
1. **Speaker 2** (Yossi) — Fix the 2 remaining bugs by end of week
2. **Speaker 3** (Dana) — Finish landing page design by Tuesday
3. **Speaker 4** (Michal) — Send final copy tomorrow morning
4. **Speaker 5** (Ori) — Update ad campaigns by Thursday
5. **Speaker 4** (Michal) — Share candidate resumes today

## Decisions
- Launch on May 25 as planned
- Shift ad budget from Facebook to Google
- Full-stack developer interviews next week
```

---

## Performance on Apple Silicon

| Operation | Estimated Time | Notes |
|-----------|---------------|-------|
| Speaker Diarization (30 min audio) | ~1–2 min | Metal GPU accelerated |
| Hebrew Transcription (30 min audio) | ~3–4 min | Apple ANE accelerated |
| Summarization | 10–30 sec | Ollama (local) |
| **Total for a 30-min meeting** | **~5–7 min** | |
| Model download (first run only) | ~10 min | ~8 GB total |

---

## Project Structure

```
localscribe/
├── localscribe.py              # Main script (v2.0 — full pipeline)
├── transcribe_and_summarize.py # Legacy script (v1.0 — transcription + summary only)
├── quick_test.py               # Quick smoke test for all components
├── install.sh                  # Automated installation script
├── requirements.txt            # Python dependencies
├── architecture.md             # Technical architecture document
├── README.md                   # This file
└── test_data/                  # Test files
    ├── README.md               # Guide to test data
    ├── download_test_audio.sh  # Script to download additional audio samples
    ├── audio/                  # Hebrew audio samples
    │   ├── hebrew_social_conversation.mp3  (2 speakers, 4:30)
    │   ├── hebrew_personal_matters.mp3     (2 speakers, 3:00)
    │   ├── hebrew_making_understood.mp3    (2 speakers, 2:30)
    │   └── hebrew_bible_genesis_ch*.mp3    (single speaker, ~4:00 each)
    └── documents/              # Sample documents for summarization testing
        ├── meeting_summary_startup.md      (Startup team meeting)
        ├── meeting_summary_board.md        (Board of directors meeting)
        ├── medical_discharge_letter.md     (Hospital discharge letter)
        ├── medical_referral.md             (Medical referral)
        ├── legal_contract_summary.md       (Rental contract)
        ├── quarterly_report.md             (Quarterly business report)
        ├── project_proposal.md             (IT project proposal)
        └── hr_policy_update.md             (Hybrid work policy)
```

---

## Tips

### Better Transcription
- Record in a quiet environment
- Speak clearly at a normal pace
- An external microphone significantly improves quality

### Better Summarization
- The default is `gemma4:e4b` (set in `localscribe.py`). To switch models:
  ```bash
  ollama pull qwen3:4b    # Smaller, faster
  ollama pull gemma3:4b   # Lightweight alternative
  ollama pull qwen3:1.7b  # Smallest, fastest
  ```
- Edit `OLLAMA_MODEL` at the top of `localscribe.py` accordingly

### Better Speaker Diarization
- Specify the known number of speakers: `--speakers 3`
- Use a stereo microphone that separates channels if possible
- Minimize background noise

---

## Troubleshooting

**Pip SSL errors (`CERTIFICATE_VERIFY_FAILED`)** — your network is intercepting TLS (common on corporate Macs). Use:
```bash
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

**`brew install` fails with "not writable"** — fix permissions, or skip brew and let `install.sh` use its `static-ffmpeg` pip fallback. To repair brew: `sudo chown -R $(whoami) /opt/homebrew`.

**ffprobe is killed (`exit -9`) when called from Python** — endpoint security tools (e.g. Santa) block unsigned binaries spawned from Python. LocalScribe v2.0 already avoids `pydub`/`ffprobe` and reads audio with `ffmpeg + soundfile` instead, so this only affects custom scripts.

**`PermissionError` writing to `~/LocalScribe_Output`** — macOS TCC blocks Python from creating folders in protected locations. Output now defaults to `./output/` next to the script.

**`Cannot access gated repo: pyannote/speaker-diarization-community-1`** — you installed `pyannote.audio>=4.0`. Pin to 3.x: `pip install "pyannote.audio>=3.1,<4.0"`.

**`UnpicklingError: Weights only load failed`** — `torch>=2.6` changed `torch.load` defaults. Pin to 2.5: `pip install "torch>=2.2,<2.6" "torchaudio>=2.2,<2.6"`.

**`AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'`** — `torchaudio>=2.7` removed it. Same pin as above.

**`hf_hub_download() got an unexpected keyword argument 'use_auth_token'`** — `huggingface_hub>=1.0` removed it. Pin: `pip install "huggingface_hub<0.30"`.

---

## FAQ

**Q: Why do I need a HuggingFace Token?**
A: The speaker diarization model (pyannote) requires you to accept its license terms. The token is free and only needed for the initial model download.

**Q: How much disk space does it use?**
A: ~8 GB total (3 GB Whisper + 3 GB pyannote + 1.7 GB Qwen3). One-time download.

**Q: Does it work offline?**
A: Yes! After the initial setup, everything runs completely offline. No internet required.

**Q: How accurate is speaker diarization?**
A: pyannote 3.1 achieves a Diarization Error Rate (DER) of approximately 10–15% on typical conversations. Specifying the number of speakers in advance (`--speakers N`) improves accuracy.

**Q: How is this different from Otter.ai / Fireflies / Granola?**
A: (1) Fully local — no data ever leaves your machine. (2) Free forever. (3) Works offline. (4) Superior Hebrew transcription accuracy via ivrit.ai.

**Q: Can I run this on iPhone?**
A: Not yet. The current version is a Mac PoC. An iOS app would use FluidAudio (CoreML) instead of pyannote. See `architecture.md` for the planned iOS architecture.

---

## Roadmap: iOS App

The iOS architecture is already planned (see `architecture.md`):
- **FluidAudio** (Swift, CoreML) — Speaker diarization on Apple Neural Engine
- **ivrit.ai Turbo** (CoreML) — Hebrew transcription
- **Qwen3** (MLX-Swift) — Local summarization

---

## License

MIT License — free to use, modify, and distribute.

**Models used:**
- ivrit.ai Whisper Turbo: MIT License
- pyannote.audio: MIT License (model requires license acceptance)
- Qwen3: Apache 2.0

---

*Built with privacy in mind. Not a single byte leaves your machine.*
