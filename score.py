"""WER scoring used in the paper (evaluation ASR: Whisper small.en).

    python score.py --wav harp.wav --ref "reference transcript"

WER is computed on normalized text (lower-cased, punctuation removed),
matching the paper's protocol. Quality metrics (PESQ, STOI, DNSMOS,
NISQA) are computed with their reference implementations; see README.
"""
import argparse
import jiwer
import whisper

_norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                       jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

def wer(wav_path, ref_text, model_name='small.en'):
    m = whisper.load_model(model_name)
    hyp = m.transcribe(wav_path, language='en', fp16=False)['text']
    return jiwer.wer(_norm(ref_text), _norm(hyp))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--ref', required=True, help='reference transcript text')
    a = ap.parse_args()
    print(f'WER: {wer(a.wav, a.ref):.4f}')
