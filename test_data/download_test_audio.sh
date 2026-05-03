#!/bin/bash
# ============================================================
# LocalScribe — Download Test Audio Files
# ============================================================
# Downloads Hebrew audio files from freely available sources
# for testing the transcription + diarization pipeline.
# ============================================================

set -e

AUDIO_DIR="$(dirname "$0")/audio"
mkdir -p "$AUDIO_DIR"

echo ""
echo "Downloading Hebrew audio test files..."
echo ""

# ─── Source 1: Listen & Learn Hebrew (Archive.org) ───
# Conversations in Hebrew with 2 speakers (male + female)
# Ideal for testing speaker diarization

echo "Source 1: Listen & Learn Hebrew (2-speaker conversations)"
echo "   Origin: archive.org | License: Public Domain"

ARCHIVE_BASE="https://archive.org/download/lp_listen-learn-hebrew_yaakov-israel-ben-david-dr-paul-holtzman"

declare -A CONVERSATIONS=(
    ["hebrew_social_conversation.mp3"]="disc1/01.01.%20Band%201.%20Social%20Conversation.mp3"
    ["hebrew_personal_matters.mp3"]="disc1/01.02.%20Band%202.%20Personal%20Matters.mp3"
    ["hebrew_making_understood.mp3"]="disc1/01.03.%20Band%203.%20Making%20Yourself%20Understood.mp3"
)

for fname in "${!CONVERSATIONS[@]}"; do
    if [ ! -f "$AUDIO_DIR/$fname" ] || [ ! -s "$AUDIO_DIR/$fname" ]; then
        echo "   Downloading $fname..."
        wget -q --timeout=60 "${ARCHIVE_BASE}/${CONVERSATIONS[$fname]}" -O "$AUDIO_DIR/$fname" || echo "   [WARN] Failed: $fname"
    else
        echo "   [OK] $fname (already exists)"
    fi
done

echo ""

# ─── Source 2: Mechon Mamre Hebrew Bible (clear Hebrew reading) ───
# Single speaker, clear Hebrew — ideal for testing transcription accuracy

echo "Source 2: Mechon Mamre — Hebrew Bible (single speaker, clear Hebrew)"
echo "   Origin: mechon-mamre.org | License: Public Domain"

for ch in 01 02 03 04 05; do
    fname="hebrew_bible_genesis_ch${ch}.mp3"
    if [ ! -f "$AUDIO_DIR/$fname" ] || [ ! -s "$AUDIO_DIR/$fname" ]; then
        echo "   Downloading $fname..."
        wget -q --timeout=30 "https://www.mechon-mamre.org/mp3/t01${ch}.mp3" -O "$AUDIO_DIR/$fname" || echo "   [WARN] Failed: $fname"
    else
        echo "   [OK] $fname (already exists)"
    fi
done

echo ""

# ─── Summary ───
echo "═══════════════════════════════════════════════════════════"
echo "Downloaded audio files:"
echo ""

total=0
for f in "$AUDIO_DIR"/*.mp3; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || echo "?")
        fname=$(basename "$f")
        printf "   %-40s %6s  %ss\n" "$fname" "$size" "${dur%.*}"
        total=$((total + 1))
    fi
done

echo ""
echo "   Total: $total files"
echo ""
echo "To test:"
echo "   python3 localscribe.py test_data/audio/hebrew_social_conversation.mp3"
echo ""
echo "Tip: The Listen & Learn Hebrew files contain 2 speakers"
echo "   (male + female) — perfect for testing speaker diarization!"
echo ""
