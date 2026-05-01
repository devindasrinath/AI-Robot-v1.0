"""
Wake word data pipeline for "Nexu".

Step 1: Generate TTS samples of "Nexu" with augmentation (positives)
Step 2: Download Google Speech Commands for negatives
Step 3: Extract log-mel spectrograms → save .npz for training + calibration

Run: python wake_word_model/model/data_pipeline.py
"""

import os
import sys
import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
N_MELS        = 40
N_FFT         = 512       # 32ms window
HOP_LENGTH    = 160       # 10ms hop
TARGET_FRAMES = 98        # ~1 second
TARGET_LEN    = SAMPLE_RATE  # 1 second of audio

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MEL_DIR  = ROOT / "calibration_data" / "samples"
DATA_DIR.mkdir(exist_ok=True)
MEL_DIR.mkdir(parents=True, exist_ok=True)

POS_DIR = DATA_DIR / "nexu"
NEG_DIR = DATA_DIR / "negatives"
POS_DIR.mkdir(exist_ok=True)
NEG_DIR.mkdir(exist_ok=True)

# ── Mel spectrogram extraction ────────────────────────────────────────────────
def wav_to_mel(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert waveform → log-mel spectrogram [1, N_MELS, TARGET_FRAMES]."""
    # Resample if needed
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad or trim to exactly 1 second
    if len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))
    else:
        wav = wav[:TARGET_LEN]

    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=20, fmax=8000,
    )
    log_mel = librosa.power_to_db(mel + 1e-9, ref=np.max)

    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)

    # Pad or trim time axis to TARGET_FRAMES
    if log_mel.shape[1] < TARGET_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, TARGET_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    return log_mel[np.newaxis, :, :]   # [1, 40, 98]


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(wav: np.ndarray, noise_wavs: list = None) -> list:
    """Return list of augmented versions of wav."""
    results = [wav.copy()]

    # Time stretch (speed change)
    for rate in [0.85, 0.92, 1.08, 1.15]:
        try:
            stretched = librosa.effects.time_stretch(wav, rate=rate)
            results.append(stretched)
        except Exception:
            pass

    # Pitch shift
    for steps in [-2, -1, 1, 2]:
        try:
            shifted = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=steps)
            results.append(shifted)
        except Exception:
            pass

    # Add background noise
    if noise_wavs:
        for _ in range(3):
            noise = random.choice(noise_wavs)
            if len(noise) < len(wav):
                noise = np.tile(noise, int(np.ceil(len(wav) / len(noise))))
            noise = noise[:len(wav)]
            snr_db = random.uniform(10, 25)
            signal_power = np.mean(wav ** 2)
            noise_power = np.mean(noise ** 2) + 1e-9
            scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            results.append(wav + scale * noise)

    # Volume jitter
    for gain in [0.7, 0.85, 1.15, 1.3]:
        results.append(np.clip(wav * gain, -1.0, 1.0))

    return results


# ── Step 1: Generate "Nexu" TTS samples ──────────────────────────────────────
def generate_nexu_samples():
    print("\n=== Step 1: Generating 'Nexu' TTS samples ===")

    phrases = [
        "Nexu", "nexu", "Hey Nexu", "hey nexu",
        "Nexu!", "NEXU", "Nexu?",
    ]

    generated = []

    # Try gTTS (online)
    try:
        from gtts import gTTS
        import io
        print("Using gTTS (Google TTS)...")
        for i, phrase in enumerate(phrases):
            for lang_tld in [('en', 'com'), ('en', 'co.uk'), ('en', 'com.au')]:
                try:
                    tts = gTTS(text=phrase, lang=lang_tld[0], tld=lang_tld[1], slow=False)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    wav, sr = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
                    out = POS_DIR / f"nexu_gtts_{i}_{lang_tld[1].replace('.','')}.wav"
                    sf.write(str(out), wav, SAMPLE_RATE)
                    generated.append(wav)
                except Exception as e:
                    print(f"  gTTS variant failed: {e}")
        print(f"  Generated {len(generated)} base samples via gTTS")
    except ImportError:
        print("  gTTS not installed, trying espeak...")

    # Try espeak-ng (offline fallback)
    if len(generated) < 3:
        import subprocess
        print("Using espeak-ng (offline)...")
        voices = ['en', 'en-us', 'en-gb', 'en+m1', 'en+m2', 'en+f1', 'en+f2']
        speeds = [130, 150, 170]
        pitches = [40, 50, 60]
        for voice in voices:
            for speed in speeds:
                for pitch in pitches:
                    out = POS_DIR / f"nexu_espeak_{voice.replace('+','_')}_{speed}_{pitch}.wav"
                    try:
                        subprocess.run([
                            'espeak-ng', '-v', voice, '-s', str(speed), '-p', str(pitch),
                            '-w', str(out), 'Nexu'
                        ], capture_output=True, check=True, timeout=5)
                        wav, sr = librosa.load(str(out), sr=SAMPLE_RATE, mono=True)
                        generated.append(wav)
                    except Exception:
                        pass
        print(f"  Generated {len(generated)} base samples via espeak-ng")

    if not generated:
        print("ERROR: No TTS engine available. Install gtts: pip install gtts")
        print("       Or install espeak-ng: sudo dnf install espeak-ng")
        sys.exit(1)

    # Load background noise from GSC if available
    noise_dir = DATA_DIR / "speech_commands" / "_background_noise_"
    noise_wavs = []
    if noise_dir.exists():
        for f in noise_dir.glob("*.wav"):
            try:
                wav, _ = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
                noise_wavs.append(wav)
            except Exception:
                pass
        print(f"  Loaded {len(noise_wavs)} background noise tracks for augmentation")

    # Augment all base samples
    print("  Augmenting...")
    all_wavs = []
    for base_wav in generated:
        augmented = augment(base_wav, noise_wavs)
        all_wavs.extend(augmented)

    # Save augmented samples
    for i, wav in enumerate(all_wavs):
        out = POS_DIR / f"nexu_aug_{i:04d}.wav"
        if not out.exists():
            sf.write(str(out), wav, SAMPLE_RATE)

    total = len(list(POS_DIR.glob("*.wav")))
    print(f"  Total positive samples: {total}")
    return total


# ── Step 2: Download Google Speech Commands (negatives) ───────────────────────
def download_negatives():
    print("\n=== Step 2: Loading Google Speech Commands (negatives) ===")

    # GSC is already extracted — load wav files directly with librosa
    # (avoids torchaudio codec dependency issues)
    gsc_root = DATA_DIR / "speech_commands" / "SpeechCommands" / "speech_commands_v0.02"

    if not gsc_root.exists():
        # Try to download via torchaudio if not yet extracted
        try:
            import torchaudio
            print("  Downloading GSC dataset...")
            torchaudio.datasets.SPEECHCOMMANDS(
                str(DATA_DIR / "speech_commands"), download=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            return 0

    if not gsc_root.exists():
        print(f"  GSC not found at {gsc_root}")
        return 0

    neg_words = ['yes', 'no', 'up', 'down', 'left', 'right',
                 'on', 'off', 'stop', 'go', 'zero', 'one',
                 'two', 'three', 'four', 'five', 'six',
                 'seven', 'eight', 'nine']

    per_word = 80
    count = 0

    for word in neg_words:
        word_dir = gsc_root / word
        if not word_dir.exists():
            continue
        wav_files = list(word_dir.glob("*.wav"))
        random.shuffle(wav_files)
        saved = 0
        for wav_file in wav_files[:per_word]:
            try:
                wav, sr = librosa.load(str(wav_file), sr=SAMPLE_RATE, mono=True)
                out = NEG_DIR / f"{word}_{saved:04d}.wav"
                sf.write(str(out), wav, SAMPLE_RATE)
                saved += 1
                count += 1
            except Exception:
                pass

    print(f"  Saved {count} negative samples from GSC")
    return count


# ── Step 3: Extract mel spectrograms ─────────────────────────────────────────
def extract_mels():
    print("\n=== Step 3: Extracting log-mel spectrograms ===")

    pos_files = list(POS_DIR.glob("*.wav"))
    neg_files = list(NEG_DIR.glob("*.wav"))

    print(f"  Positives: {len(pos_files)}  Negatives: {len(neg_files)}")

    X, y = [], []

    for f in tqdm(pos_files, desc="  Positive (Nexu)"):
        try:
            wav, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            mel = wav_to_mel(wav)
            X.append(mel)
            y.append(1)
        except Exception:
            pass

    for f in tqdm(neg_files, desc="  Negative"):
        try:
            wav, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            mel = wav_to_mel(wav)
            X.append(mel)
            y.append(0)
        except Exception:
            pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Train/val split (90/10)
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Save dataset
    np.save(str(DATA_DIR / "X_train.npy"), X_train)
    np.save(str(DATA_DIR / "y_train.npy"), y_train)
    np.save(str(DATA_DIR / "X_val.npy"),   X_val)
    np.save(str(DATA_DIR / "y_val.npy"),   y_val)

    print(f"  Train: {len(X_train)} ({y_train.sum()} pos)  "
          f"Val: {len(X_val)} ({y_val.sum()} pos)")

    # Save calibration samples (100 random from training set)
    cal_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    for i, idx in enumerate(cal_indices):
        np.savez(str(MEL_DIR / f"sample_{i:04d}.npz"),
                 mel_input=X_train[idx:idx+1])

    print(f"  Saved 100 calibration samples to {MEL_DIR}")
    print(f"  Class balance — 0 (no_wake): {(y==0).sum()}  1 (wake): {(y==1).sum()}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    n_pos = generate_nexu_samples()
    n_neg = download_negatives()

    if n_pos == 0:
        print("No positive samples — aborting.")
        sys.exit(1)

    extract_mels()
    print("\nData pipeline complete. Run train.py next.")
