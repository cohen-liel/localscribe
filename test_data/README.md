# Test Data

This directory contains sample files for testing LocalScribe — audio recordings, sample documents, and real-world documents from public sources.

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
| `knesset_economics_committee_16apr26.mp3` | Knesset TV | Economics Committee meeting (16.04.26) | Multi-speaker | 10:00 |
| `knesset_constitution_committee_29apr26.mp3` | Knesset TV | Constitution Committee meeting (29.04.26) | Multi-speaker | 10:00 |
| `knesset_education_committee_19apr26.mp3` | Knesset TV | Education Committee meeting (19.04.26) | Multi-speaker | 7:22 |

### Additional Recommended Sources

For more advanced testing, you can download from the following datasets:

| Source | Description | Link |
|--------|-------------|------|
| **Verbit Hebrew Medical Audio** | 1,000+ medical recordings, 41 speakers | [HuggingFace](https://huggingface.co/datasets/verbit/hebrew_medical_audio) |
| **ivrit.ai Knesset Plenums** | Israeli parliament session recordings with transcripts | [HuggingFace](https://huggingface.co/datasets/ivrit-ai/knesset-plenums) |
| **HebDB** | 2,500 hours of spontaneous Hebrew speech | [HuggingFace](https://huggingface.co/datasets/SLPRL-HUJI/HebDB) |
| **Robo-Shaul** | 30 hours of Israeli economics podcast | [HuggingFace](https://huggingface.co/datasets/Roboshaul/Roboshaul) |

---

## Sample Documents (`documents/`)

Fictional documents in Hebrew/English for testing the document summarization feature. The system auto-detects the document type and applies a tailored summarization prompt.

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

---

## Real-World Documents (`documents/real_docs/`)

Authentic documents downloaded from public Israeli sources — government reports, legal documents, financial forms, and more. These are ideal for stress-testing the summarization engine on real content.

| File | Type | Pages | Source | Language |
|------|------|-------|--------|----------|
| `bank_of_israel_annual_report_2024.pdf` | Financial Report | 234 | Bank of Israel | English |
| `israel_governance_report_2024.pdf` | Governance Report | 66 | SGI Network | English |
| `hebrew_supreme_court_decision.pdf` | Court Decision | 14 | Supreme Court of Israel | Hebrew |
| `hebrew_rental_contract.pdf` | Legal Contract | 5 | Public template | Hebrew |
| `hebrew_nda_agreement.pdf` | Legal / NDA | 2 | Public template | Hebrew |
| `hebrew_legal_orders_of_protection.pdf` | Legal Guide | 4 | Family Legal Care | Hebrew |
| `knesset_committee_participation_guide.pdf` | Government Guide | 3 | Knesset / NGO | Hebrew |
| `company_annual_report_form.pdf` | Business Form | 3 | Companies Registrar | Hebrew |
| `private_company_annual_report.pdf` | Business Form | 3 | Companies Registrar | Hebrew |

### Usage
```bash
# Summarize a single document
python3 localscribe.py --document test_data/documents/meeting_summary_startup.md

# Summarize a real PDF document
python3 localscribe.py --document test_data/documents/real_docs/hebrew_supreme_court_decision.pdf

# Summarize all sample documents
python3 localscribe.py --document-dir test_data/documents/

# Summarize all real-world documents
python3 localscribe.py --document-dir test_data/documents/real_docs/
```

---

**Note:** Sample documents in `documents/` are fictional and were created solely for testing purposes. Real-world documents in `documents/real_docs/` are publicly available government and legal documents downloaded from official sources.
