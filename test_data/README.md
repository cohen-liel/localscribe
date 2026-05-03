# Test Data

This directory contains sample files for testing LocalScribe — both audio and documents.

---

## Audio (`audio/`)

Hebrew audio files from freely available sources for testing the full pipeline (transcription + speaker diarization + summarization).

### Automatic Download
```bash
chmod +x test_data/download_test_audio.sh
./test_data/download_test_audio.sh
```

### Available Files

| File | Source | Description | Speakers | Duration |
|------|--------|-------------|----------|----------|
| `hebrew_social_conversation.mp3` | Archive.org | Social conversation in Hebrew | 2 (male + female) | ~4:30 |
| `hebrew_personal_matters.mp3` | Archive.org | Conversation about personal matters | 2 (male + female) | ~3:00 |
| `hebrew_making_understood.mp3` | Archive.org | Conversation about mutual understanding | 2 (male + female) | ~2:30 |
| `hebrew_bible_genesis_ch01.mp3` | Mechon Mamre | Genesis Chapter 1 | 1 (single speaker) | ~6:00 |
| `hebrew_bible_genesis_ch02.mp3` | Mechon Mamre | Genesis Chapter 2 | 1 (single speaker) | ~4:25 |
| `hebrew_bible_genesis_ch03-05.mp3` | Mechon Mamre | Genesis Chapters 3–5 | 1 (single speaker) | ~4:20 each |

### Additional Recommended Sources

For more advanced testing, you can download from the following datasets:

| Source | Description | Link |
|--------|-------------|------|
| **Verbit Hebrew Medical Audio** | 1,000+ medical recordings, 41 speakers | [HuggingFace](https://huggingface.co/datasets/verbit/hebrew_medical_audio) |
| **ivrit.ai Knesset Plenums** | Israeli parliament session recordings with transcripts | [HuggingFace](https://huggingface.co/datasets/ivrit-ai/knesset-plenums) |
| **HebDB** | 2,500 hours of spontaneous Hebrew speech | [HuggingFace](https://huggingface.co/datasets/SLPRL-HUJI/HebDB) |
| **Robo-Shaul** | 30 hours of Israeli economics podcast | [HuggingFace](https://huggingface.co/datasets/Roboshaul/Roboshaul) |

---

## Documents (`documents/`)

Sample documents in Hebrew for testing the document summarization feature. The system auto-detects the document type and applies a tailored summarization prompt.

| File | Type | Description |
|------|------|-------------|
| `meeting_summary_startup.md` | Meeting | Startup team status meeting |
| `meeting_summary_board.md` | Meeting | Board of directors meeting |
| `medical_discharge_letter.md` | Medical | Hospital discharge letter (fictional) |
| `medical_referral.md` | Medical | Medical referral (fictional) |
| `legal_contract_summary.md` | Legal | Rental contract summary |
| `quarterly_report.md` | Report | Company quarterly business report |
| `project_proposal.md` | Proposal | IT project proposal |
| `hr_policy_update.md` | HR / Policy | Hybrid work policy update |

### Usage
```bash
# Summarize a single document
python3 localscribe.py --document test_data/documents/meeting_summary_startup.md

# Summarize all documents in the folder
python3 localscribe.py --document-dir test_data/documents/
```

---

**Note:** All documents in this directory are fictional and were created solely for testing purposes. They do not contain any real personal or sensitive information.
