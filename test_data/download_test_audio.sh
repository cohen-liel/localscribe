#!/bin/bash
# ============================================================
# LocalScribe - הורדת קבצי אודיו לבדיקה
# ============================================================
# מוריד קבצי אודיו בעברית ממקורות חופשיים לבדיקת המערכת
# ============================================================

set -e

AUDIO_DIR="$(dirname "$0")/audio"
mkdir -p "$AUDIO_DIR"

echo ""
echo "📥 מוריד קבצי אודיו בעברית לבדיקה..."
echo ""

# ─── מקור 1: Listen & Learn Hebrew (Archive.org) ───
# שיחות בעברית עם 2 דוברים (גבר ואישה) - מצוין לבדיקת זיהוי דוברים
echo "📦 מקור 1: Listen & Learn Hebrew (שיחות עם 2 דוברים)"
echo "   מקור: archive.org | רישיון: Public Domain"

ARCHIVE_BASE="https://archive.org/download/lp_listen-learn-hebrew_yaakov-israel-ben-david-dr-paul-holtzman"

declare -A CONVERSATIONS=(
    ["hebrew_social_conversation.mp3"]="disc1/01.01.%20Band%201.%20Social%20Conversation.mp3"
    ["hebrew_personal_matters.mp3"]="disc1/01.02.%20Band%202.%20Personal%20Matters.mp3"
    ["hebrew_making_understood.mp3"]="disc1/01.03.%20Band%203.%20Making%20Yourself%20Understood.mp3"
)

for fname in "${!CONVERSATIONS[@]}"; do
    if [ ! -f "$AUDIO_DIR/$fname" ] || [ ! -s "$AUDIO_DIR/$fname" ]; then
        echo "   ⬇️  $fname..."
        wget -q --timeout=60 "${ARCHIVE_BASE}/${CONVERSATIONS[$fname]}" -O "$AUDIO_DIR/$fname" || echo "   ⚠️  נכשל: $fname"
    else
        echo "   ✅ $fname (כבר קיים)"
    fi
done

echo ""

# ─── מקור 2: Mechon Mamre Hebrew Bible (קריאה ברורה בעברית) ───
# דובר יחיד, עברית ברורה - מצוין לבדיקת תמלול
echo "📦 מקור 2: Mechon Mamre - תנ\"ך בעברית (דובר יחיד, עברית ברורה)"
echo "   מקור: mechon-mamre.org | רישיון: Public Domain"

for ch in 01 02 03 04 05; do
    fname="hebrew_bible_genesis_ch${ch}.mp3"
    if [ ! -f "$AUDIO_DIR/$fname" ] || [ ! -s "$AUDIO_DIR/$fname" ]; then
        echo "   ⬇️  $fname..."
        wget -q --timeout=30 "https://www.mechon-mamre.org/mp3/t01${ch}.mp3" -O "$AUDIO_DIR/$fname" || echo "   ⚠️  נכשל: $fname"
    else
        echo "   ✅ $fname (כבר קיים)"
    fi
done

echo ""

# ─── סיכום ───
echo "═══════════════════════════════════════════════════════════"
echo "📋 קבצי אודיו שהורדו:"
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
echo "   סה\"כ: $total קבצים"
echo ""
echo "🧪 לבדיקה:"
echo "   python3 localscribe.py test_data/audio/hebrew_social_conversation.mp3"
echo ""
echo "💡 טיפ: הקבצים מ-Listen & Learn Hebrew מכילים 2 דוברים"
echo "   (גבר ואישה) - מושלם לבדיקת זיהוי דוברים!"
echo ""
