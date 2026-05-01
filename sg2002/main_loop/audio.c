#include "audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>

/*
 * Audio capture via arecord (no libasound dependency).
 * Mel spectrogram computed in C matching the Python training pipeline:
 *   SAMPLE_RATE=16000, N_FFT=512, HOP=160, N_MELS=40, N_FRAMES=98
 */

#define SAMPLE_RATE  16000
#define N_FFT        512
#define HOP_LENGTH   160
#define N_MELS       40
#define N_FRAMES     98
#define N_BINS       (N_FFT / 2 + 1)   /* 257 */

/* Need N_FFT + (N_FRAMES-1)*HOP samples to produce N_FRAMES output frames */
#define NEEDED_SAMPLES  (N_FFT + (N_FRAMES - 1) * HOP_LENGTH)  /* 16192 */
#define RING_SIZE       17000   /* slightly larger than NEEDED_SAMPLES */

/* ── State ───────────────────────────────────────────────────────────────── */
static int16_t  s_ring[RING_SIZE];
static int      s_ring_pos  = 0;
static bool     s_available = false;
static pthread_mutex_t s_ring_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Precomputed tables */
static float s_hann[N_FFT];
static float s_mel_fb[N_MELS][N_BINS];

/* ── FFT (radix-2 Cooley-Tukey, in-place, N must be power of 2) ─────────── */

typedef struct { float r, i; } Cx;

static void fft(Cx *x, int n)
{
    /* Bit-reversal permutation */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { Cx t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    /* Butterfly stages */
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        Cx wlen = { cosf(ang), sinf(ang) };
        for (int i = 0; i < n; i += len) {
            Cx w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                Cx u = x[i + j];
                Cx v = { x[i+j+len/2].r*w.r - x[i+j+len/2].i*w.i,
                          x[i+j+len/2].r*w.i + x[i+j+len/2].i*w.r };
                x[i + j]         = (Cx){ u.r + v.r, u.i + v.i };
                x[i + j + len/2] = (Cx){ u.r - v.r, u.i - v.i };
                float wr = w.r*wlen.r - w.i*wlen.i;
                w.i = w.r*wlen.i + w.i*wlen.r;
                w.r = wr;
            }
        }
    }
}

/* ── Mel filterbank precomputation ───────────────────────────────────────── */

static float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void precompute(void)
{
    /* Hann window */
    for (int i = 0; i < N_FFT; i++)
        s_hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (N_FFT - 1)));

    /* Mel filterbank — triangular filters, matches librosa defaults */
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel((float)SAMPLE_RATE / 2.0f);

    float mel_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++)
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (N_MELS + 1);

    float bin_pts[N_MELS + 2];
    for (int i = 0; i < N_MELS + 2; i++) {
        float hz = mel_to_hz(mel_pts[i]);
        bin_pts[i] = floorf((N_FFT + 1) * hz / SAMPLE_RATE);
    }

    memset(s_mel_fb, 0, sizeof(s_mel_fb));
    for (int m = 0; m < N_MELS; m++) {
        int f0 = (int)bin_pts[m];
        int f1 = (int)bin_pts[m + 1];
        int f2 = (int)bin_pts[m + 2];
        for (int k = f0; k < f1 && k < N_BINS; k++)
            if (f1 > f0) s_mel_fb[m][k] = (float)(k - f0) / (f1 - f0);
        for (int k = f1; k <= f2 && k < N_BINS; k++)
            if (f2 > f1) s_mel_fb[m][k] = (float)(f2 - k) / (f2 - f1);
    }
}

/* ── Mel spectrogram computation ─────────────────────────────────────────── */

static void compute_mel(const int16_t *samples, float *mel_out)
{
    static Cx frame[N_FFT];
    float power[N_BINS];

    for (int fi = 0; fi < N_FRAMES; fi++) {
        int offset = fi * HOP_LENGTH;

        /* Apply Hann window */
        for (int i = 0; i < N_FFT; i++) {
            float s = (offset + i < NEEDED_SAMPLES)
                      ? (samples[offset + i] / 32768.0f) : 0.0f;
            frame[i].r = s * s_hann[i];
            frame[i].i = 0.0f;
        }

        fft(frame, N_FFT);

        /* Power spectrum */
        for (int k = 0; k < N_BINS; k++)
            power[k] = frame[k].r * frame[k].r + frame[k].i * frame[k].i;

        /* Mel filterbank + log — output layout: [mel_bin][frame] */
        for (int m = 0; m < N_MELS; m++) {
            float energy = 0.0f;
            for (int k = 0; k < N_BINS; k++)
                energy += s_mel_fb[m][k] * power[k];
            mel_out[m * N_FRAMES + fi] = logf(energy + 1e-9f);
        }
    }

    /* Normalize: zero mean, unit variance (matches training pipeline) */
    float sum = 0.0f;
    for (int i = 0; i < MEL_ELEMS; i++) sum += mel_out[i];
    float mean = sum / MEL_ELEMS;

    float sq = 0.0f;
    for (int i = 0; i < MEL_ELEMS; i++) {
        float d = mel_out[i] - mean;
        sq += d * d;
    }
    float std = sqrtf(sq / MEL_ELEMS) + 1e-9f;
    for (int i = 0; i < MEL_ELEMS; i++)
        mel_out[i] = (mel_out[i] - mean) / std;
}

/* ── Capture thread ──────────────────────────────────────────────────────── */

static void *capture_thread(void *arg)
{
    (void)arg;

    /* Launch arecord — streams raw PCM to stdout */
    FILE *pipe = popen(
        "arecord -D hw:1,0 -r 16000 -c 1 -f S16_LE 2>/dev/null", "r");
    if (!pipe) {
        fprintf(stderr, "[audio] arecord failed — mic unavailable\n");
        return NULL;
    }

    printf("[audio] Mic started (hw:1,0, 16kHz mono)\n");
    s_available = true;

    int16_t hop[HOP_LENGTH];
    while (1) {
        if (fread(hop, sizeof(int16_t), HOP_LENGTH, pipe) != HOP_LENGTH)
            break;

        pthread_mutex_lock(&s_ring_mutex);
        /* Shift ring left to make room if needed */
        if (s_ring_pos + HOP_LENGTH > RING_SIZE) {
            int shift = s_ring_pos + HOP_LENGTH - RING_SIZE;
            memmove(s_ring, s_ring + shift,
                    (s_ring_pos - shift) * sizeof(int16_t));
            s_ring_pos -= shift;
        }
        memcpy(s_ring + s_ring_pos, hop, HOP_LENGTH * sizeof(int16_t));
        s_ring_pos += HOP_LENGTH;
        pthread_mutex_unlock(&s_ring_mutex);
    }

    pclose(pipe);
    s_available = false;
    printf("[audio] Mic disconnected\n");
    return NULL;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int audio_init(void)
{
    precompute();
    memset(s_ring, 0, sizeof(s_ring));

    pthread_t tid;
    pthread_create(&tid, NULL, capture_thread, NULL);
    pthread_detach(tid);

    /* Give arecord a moment to start; if mic absent the thread exits quietly */
    printf("[audio] Waiting for mic...\n");
    return 0;
}

bool audio_get_mel(float *mel_out)
{
    pthread_mutex_lock(&s_ring_mutex);

    if (!s_available || s_ring_pos < NEEDED_SAMPLES) {
        pthread_mutex_unlock(&s_ring_mutex);
        memset(mel_out, 0, MEL_ELEMS * sizeof(float));
        return false;
    }

    /* Copy the most recent NEEDED_SAMPLES into a local buffer */
    int16_t tmp[NEEDED_SAMPLES];
    memcpy(tmp, s_ring + s_ring_pos - NEEDED_SAMPLES,
           NEEDED_SAMPLES * sizeof(int16_t));

    pthread_mutex_unlock(&s_ring_mutex);

    compute_mel(tmp, mel_out);
    return true;
}
