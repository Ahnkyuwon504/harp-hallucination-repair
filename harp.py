"""HARP: Hallucination-Aware Routing to a Predictive anchor.

Training-free, reference-free span repair for generative speech
enhancement. Given the same noisy input enhanced by (1) a generative
model and (2) a discriminative anchor, HARP transcribes both with a
lightweight ASR, finds spans where the transcripts disagree, and
replaces only those spans in the generative waveform with the anchor
waveform (raised-cosine crossfade at the boundaries).

Paper: "HARP: Hallucination-Aware Routing to a Predictive Anchor for
Training-Free Repair of Generative Speech Enhancement" (submitted to
ICASSP 2027). Core functions are taken verbatim from the experiment
notebooks; only I/O plumbing was added.

Usage:
    python harp.py --gen x_g.wav --anchor x_d.wav --out harp.wav \
                   [--delta 0.2] [--crossfade 0.025]

Audio is expected at 16 kHz mono.
"""
import argparse, difflib
import numpy as np
import soundfile as sf
import jiwer
import whisper

_norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                       jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

_W = None
def _model():
    global _W
    if _W is None:
        _W = whisper.load_model('base.en')
    return _W

def words_of(path):
    """Transcribe and return [(normalized_word, start_s, end_s), ...]."""
    res = _model().transcribe(path, language='en', fp16=False,
                              word_timestamps=True)
    out = []
    for seg in res.get('segments', []):
        for w in seg.get('words', []):
            tok = _norm(w.get('word', '')).strip()
            if tok:
                out.append((tok, float(w['start']), float(w['end'])))
    return out

def s4_intervals(gw, pw, dil):
    """Risk spans on the generative timeline.

    gw: words of the generative output, pw: words of the anchor.
    Insert/replace ops take the generative word's span; a deletion
    (word present in the anchor but missing in the generative output)
    takes a +/-0.10 s proxy window at the expected position. All spans
    are dilated by `dil` seconds and merged.
    """
    a = [x[0] for x in pw]; b = [x[0] for x in gw]
    ivs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=a, b=b, autojunk=False).get_opcodes():
        if tag in ('insert', 'replace'):
            for k in range(j1, j2):
                ivs.append((gw[k][1]-dil, gw[k][2]+dil))
        elif tag == 'delete' and gw:
            c0 = gw[j1][1] if j1 < len(gw) else gw[-1][2]
            ivs.append((c0-0.10-dil, c0+0.10+dil))
    ivs = sorted((max(0.0, a0), b0) for a0, b0 in ivs)
    merged = []
    for a0, b0 in ivs:
        if merged and a0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b0))
        else:
            merged.append((a0, b0))
    return merged

def fuse(y_gen, y_anchor, ivs, cf, sr=16000):
    """Replace `ivs` spans of y_gen with y_anchor; raised-cosine edges."""
    m = min(len(y_gen), len(y_anchor))
    g = y_gen[:m].copy(); k = y_anchor[:m]
    L = max(1, int(cf*sr))
    mask = np.zeros(m, dtype=np.float32)
    for a0, b0 in ivs:
        i0, i1 = int(a0*sr), min(m, int(b0*sr))
        if i1 <= i0:
            continue
        mask[i0:i1] = 1.0
    if L > 1 and mask.any():
        ramp = 0.5-0.5*np.cos(np.linspace(0, np.pi, L, dtype=np.float32))
        d = np.diff(np.concatenate([[0.], mask, [0.]]))
        for i in np.where(d > 0)[0]:
            j0 = max(0, i-L//2); seg = min(L, m-j0)
            mask[j0:j0+seg] = np.maximum(mask[j0:j0+seg], ramp[:seg])
        for i in np.where(d < 0)[0]:
            j0 = max(0, i-1-L//2); seg = min(L, m-j0)
            mask[j0:j0+seg] = np.maximum(mask[j0:j0+seg], ramp[::-1][:seg])
    return g*(1-mask)+k*mask

def harp(gen_path, anchor_path, out_path, delta=0.2, crossfade=0.025):
    gw = words_of(gen_path)
    pw = words_of(anchor_path)
    ivs = s4_intervals(gw, pw, delta)
    y_g, sr1 = sf.read(gen_path, dtype='float32')
    y_d, sr2 = sf.read(anchor_path, dtype='float32')
    assert sr1 == sr2 == 16000, 'expected 16 kHz mono audio'
    y = fuse(y_g, y_d, ivs, crossfade)
    sf.write(out_path, y, 16000)
    return ivs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True, help='generative output wav')
    ap.add_argument('--anchor', required=True, help='discriminative output wav')
    ap.add_argument('--out', required=True, help='repaired output wav')
    ap.add_argument('--delta', type=float, default=0.2)
    ap.add_argument('--crossfade', type=float, default=0.025)
    a = ap.parse_args()
    spans = harp(a.gen, a.anchor, a.out, a.delta, a.crossfade)
    print('risk spans (s):', [(round(x, 2), round(y, 2)) for x, y in spans])
