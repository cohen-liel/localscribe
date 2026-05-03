# Test Data - קבצי בדיקה

תיקייה זו מכילה קבצי בדיקה לבדיקת LocalScribe - אודיו ומסמכים.

---

## אודיו (`audio/`)

קבצי אודיו בעברית ממקורות חופשיים לבדיקת הצינור המלא (תמלול + זיהוי דוברים + סיכום).

### הורדה אוטומטית
```bash
chmod +x test_data/download_test_audio.sh
./test_data/download_test_audio.sh
```

### קבצים זמינים

| קובץ | מקור | תיאור | דוברים | אורך |
|-------|------|--------|--------|------|
| `hebrew_social_conversation.mp3` | Archive.org | שיחה חברתית בעברית | 2 (גבר + אישה) | ~4:30 |
| `hebrew_personal_matters.mp3` | Archive.org | שיחה על עניינים אישיים | 2 (גבר + אישה) | ~3:00 |
| `hebrew_making_understood.mp3` | Archive.org | שיחה על הבנה הדדית | 2 (גבר + אישה) | ~2:30 |
| `hebrew_bible_genesis_ch01.mp3` | Mechon Mamre | בראשית פרק א' | 1 (דובר יחיד) | ~6:00 |
| `hebrew_bible_genesis_ch02.mp3` | Mechon Mamre | בראשית פרק ב' | 1 (דובר יחיד) | ~4:25 |
| `hebrew_bible_genesis_ch03-05.mp3` | Mechon Mamre | בראשית פרקים ג'-ה' | 1 (דובר יחיד) | ~4:20 כ"א |

### מקורות נוספים מומלצים

לבדיקה מתקדמת יותר, ניתן להוריד מהמקורות הבאים:

| מקור | תיאור | קישור |
|------|--------|-------|
| **Verbit Hebrew Medical Audio** | 1000+ הקלטות רפואיות, 41 דוברים | [HuggingFace](https://huggingface.co/datasets/verbit/hebrew_medical_audio) |
| **ivrit.ai Knesset Plenums** | הקלטות ישיבות כנסת עם תמלולים | [HuggingFace](https://huggingface.co/datasets/ivrit-ai/knesset-plenums) |
| **HebDB** | 2500 שעות דיבור ספונטני בעברית | [HuggingFace](https://huggingface.co/datasets/SLPRL-HUJI/HebDB) |
| **Robo-Shaul** | 30 שעות פודקאסט כלכלי ישראלי | [HuggingFace](https://huggingface.co/datasets/Roboshaul/Roboshaul) |

---

## מסמכים (`documents/`)

מסמכים גנריים בעברית לבדיקת יכולת סיכום המסמכים.

| קובץ | סוג | תיאור |
|-------|-----|--------|
| `meeting_summary_startup.md` | סיכום פגישה | פגישת סטטוס סטארטאפ טכנולוגי |
| `meeting_summary_board.md` | סיכום פגישה | ישיבת דירקטוריון חברה |
| `medical_discharge_letter.md` | מסמך רפואי | מכתב שחרור מבית חולים (בדיוני) |
| `medical_referral.md` | מסמך רפואי | הפניה רפואית (בדיונית) |
| `legal_contract_summary.md` | מסמך משפטי | סיכום חוזה שכירות |
| `quarterly_report.md` | דוח עסקי | דוח רבעוני של חברה |
| `project_proposal.md` | הצעת פרויקט | הצעה לפרויקט טכנולוגי |
| `hr_policy_update.md` | מדיניות | עדכון מדיניות משאבי אנוש |

### שימוש
```bash
# סיכום מסמך בודד
python3 localscribe.py --document test_data/documents/meeting_summary_startup.md

# סיכום כל המסמכים בתיקייה
python3 localscribe.py --document-dir test_data/documents/
```

---

**הערה:** כל המסמכים בתיקייה זו הם בדיוניים ונוצרו לצורכי בדיקה בלבד. אין בהם מידע אישי או רגיש אמיתי.
