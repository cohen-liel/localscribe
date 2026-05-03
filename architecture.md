# LocalScribe - Architecture Document

## Vision

LocalScribe is a fully on-device meeting transcription and summarization product for Hebrew, targeting iPhone 17 (A19 Pro) and Mac M4. The core value proposition is **absolute privacy** — no audio or text ever leaves the user's device.

---

## System Architecture

### Mac PoC (Current - Python)

```
┌─────────────────────────────────────────────────────────────┐
│                    LocalScribe Mac PoC                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │   │
│  │  Diarization │───▶│  Hebrew ASR  │───▶│ Summarization│   │
│  │              │    │              │    │              │   │
│  │ pyannote 3.1 │    │ ivrit.ai     │    │ Qwen3 1.7B   │   │
│  │ (Metal GPU)  │    │ Turbo (MLX)  │    │ (Ollama)     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  Input: Audio file (.mp3/.wav/.m4a)                          │
│  Output: Markdown + JSON with speakers, transcript, summary  │
└─────────────────────────────────────────────────────────────┘
```

### iOS Production App (Future - Swift)

```
┌─────────────────────────────────────────────────────────────┐
│                  LocalScribe iOS App                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │   │
│  │  Diarization │───▶│  Hebrew ASR  │───▶│ Summarization│   │
│  │              │    │              │    │              │   │
│  │ FluidAudio   │    │ ivrit.ai     │    │ Qwen3 0.5B   │   │
│  │ (CoreML/ANE) │    │ Turbo(CoreML)│    │ (MLX-Swift)  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
│  All inference on Apple Neural Engine (ANE)                   │
│  RAM budget: ~4GB total (of 8-12GB available on iPhone 17)   │
│  Battery impact: Minimal (ANE is power-efficient)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Stage 1: Speaker Diarization

| Property | Mac PoC | iOS App |
|----------|---------|---------|
| Library | pyannote.audio 3.1 | FluidAudio SDK |
| Model | pyannote/speaker-diarization-3.1 | CoreML (converted) |
| Hardware | Metal GPU (MPS) | Apple Neural Engine |
| Latency | Batch (offline) | Real-time streaming (<150ms) |
| Output | List of (start, end, speaker_id) | Same |

**Key capability:** Identifies WHO spoke WHEN, without knowing speakers in advance.

### Stage 2: Hebrew ASR (Automatic Speech Recognition)

| Property | Mac PoC | iOS App |
|----------|---------|---------|
| Model | ivrit-ai/whisper-large-v3-turbo-d4 | Same (CoreML converted) |
| Engine | mlx-whisper (Apple Silicon optimized) | CoreML / WhisperKit |
| Accuracy | 94-95% on Hebrew | Same |
| Speed | ~8x real-time on M4 | ~4-6x real-time on A19 |
| Languages | Hebrew (primary), English (secondary) | Same |

**Why ivrit.ai?** Fine-tuned on 22,000 hours of Hebrew audio. Outperforms Google, Amazon, and vanilla Whisper on Hebrew by 5-10% WER.

### Stage 3: Summarization

| Property | Mac PoC | iOS App |
|----------|---------|---------|
| Model | Qwen3 1.7B | Qwen3 0.5B or 1.5B |
| Engine | Ollama | MLX-Swift / CoreML |
| RAM | ~1.7GB | ~0.5-1.5GB |
| Speed | ~30 tok/s on M4 | ~15-20 tok/s on A19 |
| Prompt | Hebrew-optimized with speaker context | Same |

**Output format:**
- Meeting title
- Summary (3-5 sentences)
- Action items (assigned to specific speakers)
- Decisions made
- Open questions

---

## Data Flow

```
Audio Input (mic or file)
    │
    ├── [VAD] Voice Activity Detection (Silero)
    │         → Remove silence, identify speech regions
    │
    ├── [Diarization] Speaker Segmentation
    │         → Assign speaker IDs to time regions
    │         → Merge adjacent segments from same speaker
    │
    ├── [ASR] Per-segment transcription
    │         → Hebrew text for each speaker segment
    │         → Post-processing: custom vocabulary corrections
    │
    ├── [Formatting] Combine into structured transcript
    │         → "[00:00] Speaker 1: ..."
    │         → "[00:15] Speaker 2: ..."
    │
    └── [LLM] Summarization with speaker context
              → Summary, action items, decisions
```

---

## Memory Budget (iPhone 17, 8GB RAM)

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| iOS + System | ~3GB | Always reserved |
| Diarization model | ~0.5GB | CoreML, ANE-optimized |
| Whisper model | ~1.5GB | Turbo variant (smaller) |
| LLM (Qwen3 0.5B) | ~0.5GB | Quantized (Q4) |
| Audio buffer | ~0.2GB | 60 min @ 16kHz mono |
| App overhead | ~0.3GB | UI, processing buffers |
| **Total** | **~6GB** | **Fits in 8GB** |

---

## Development Roadmap

### Phase 1: Mac PoC (Current)
- [x] Basic transcription (mlx-whisper)
- [x] Basic summarization (Ollama + Qwen3)
- [x] Speaker diarization (pyannote.audio)
- [x] Full pipeline integration
- [ ] Testing with real Hebrew meetings
- [ ] Custom vocabulary support
- [ ] Speaker name assignment (learn names)

### Phase 2: Mac App (Next)
- [ ] SwiftUI desktop app
- [ ] FluidAudio integration (replace pyannote)
- [ ] Real-time streaming mode
- [ ] Meeting history and search
- [ ] Export to various formats

### Phase 3: iOS App (Target)
- [ ] CoreML model conversion (ivrit.ai)
- [ ] FluidAudio iOS integration
- [ ] MLX-Swift for LLM inference
- [ ] Background recording (Audio Session)
- [ ] Share Extension (receive audio from other apps)
- [ ] iCloud sync (encrypted, optional)

---

## Competitive Landscape

| Feature | LocalScribe | Otter.ai | Fireflies | Apple Notes |
|---------|------------|----------|-----------|-------------|
| Hebrew support | ✅ Excellent | ⚠️ Basic | ⚠️ Basic | ❌ None |
| On-device | ✅ 100% | ❌ Cloud | ❌ Cloud | ✅ Partial |
| Speaker ID | ✅ | ✅ | ✅ | ❌ |
| Privacy | ✅ Zero-trust | ❌ | ❌ | ✅ |
| Offline | ✅ | ❌ | ❌ | ⚠️ Limited |
| Cost | Free/One-time | $100+/yr | $120+/yr | Free |
| Custom vocab | ✅ (planned) | ❌ | ❌ | ❌ |

---

## Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Diarization accuracy on Hebrew | Medium | FluidAudio is language-agnostic (works on audio features) |
| iPhone RAM constraints | High | Use quantized models (Q4), load/unload sequentially |
| Battery drain during long meetings | Medium | ANE is power-efficient; can process in background |
| CoreML conversion of ivrit.ai | Medium | WhisperKit already supports custom Whisper models |
| LLM quality for Hebrew summaries | Low | Qwen3 has excellent Hebrew; can upgrade to 1.5B |

---

## References

- [FluidAudio](https://github.com/FluidInference/FluidAudio) - CoreML audio models for Apple
- [ivrit.ai](https://huggingface.co/ivrit-ai) - Hebrew ASR models
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Ollama](https://ollama.com) - Local LLM runner
