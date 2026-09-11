# HARP: Hallucination-Aware Routing to a Predictive Anchor

Training-free, reference-free **span-level repair of hallucination in
generative speech enhancement (SE)**.

Generative SE can replace heavily masked speech with fluent content that
was never spoken. HARP runs a discriminative model in parallel as an
*anchor*, transcribes both outputs with a lightweight ASR
(Whisper base.en), flags spans where the transcripts disagree, and
replaces only those spans in the generative waveform with the anchor
waveform (25 ms raised-cosine crossfades).

Paper: *HARP: Hallucination-Aware Routing to a Predictive Anchor for
Training-Free Repair of Generative Speech Enhancement*, submitted to
ICASSP 2027 (Kyuwon Ahn, Taehoon Kim — Sogang University).

## Install

```bash
pip install -r requirements.txt   # numpy, soundfile, jiwer, openai-whisper
```

## Use

Enhance the same 16 kHz noisy input with your generative model and your
discriminative anchor, then:

```bash
python harp.py --gen x_g.wav --anchor x_d.wav --out harp.wav \
               --delta 0.2 --crossfade 0.025
python score.py --wav harp.wav --ref "reference transcript"
```

`--delta` dilates each risk span (paper setting: 0.2 s, tuned on 0 dB
only). Deletions (words present in the anchor but missing in the
generative output) get a ±0.10 s proxy window at the expected position.

## Models and data (not redistributed here)

- Generative bases: [SGMSE+](https://github.com/sp-uhh/sgmse),
  [StoRM](https://github.com/sp-uhh/storm), PGUSE (paper's recipe)
- Discriminative anchors:
  [PrimeK-Net](https://github.com/huaidanquede/PrimeK-Net),
  [MP-SENet](https://github.com/yxlu-0102/MP-SENet)
- ASR: [openai/whisper](https://github.com/openai/whisper)
  (signal: base.en, evaluation: small.en)
- Data: [VoiceBank-DEMAND](https://datashare.ed.ac.uk/handle/10283/2791),
  [MUSAN](https://www.openslr.org/17/)

Use each from its original source under its own license. This
repository contains code only (MIT).

## Pre-specified decision criteria

Fixed before the main evaluation (fixed before the main evaluation and hard-coded in the experiment
notebooks): significant WER reduction vs. the generative baseline
(paired Wilcoxon, p<0.01); DNSMOS drop ≤0.05 vs. the best
generative-family baseline; localization AUROC ≥0.70. Hyperparameters
tuned on the 0 dB condition only.

## Citation

Citation entry will be added upon publication.
