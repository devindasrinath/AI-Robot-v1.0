/**
 * wake_word_test.c — Verify wake_word DS-CNN cvimodel on SG2002 NPU
 *
 * Build: make -C sg2002/wake_word/
 * Run:   ./wake_word_test /root/wake_word.cvimodel
 *
 * Inputs : mel_input  [1, 1, 40, 98]  FP32  (log-mel spectrogram)
 * Outputs: logits     [1, 2]           FP32  [no_wake, wake]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cviruntime.h>

#define MODEL_DEFAULT  "/root/wake_word.cvimodel"
#define MEL_ELEMS      (1 * 40 * 98)   /* 3920 floats */
#define N_RUNS         20               /* timing iterations */

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

int main(int argc, char *argv[])
{
    const char *model_path = (argc > 1) ? argv[1] : MODEL_DEFAULT;

    printf("=== Wake Word NPU Verification ===\n");
    printf("Model: %s\n\n", model_path);

    /* Load model */
    CVI_MODEL_HANDLE model;
    CVI_RC rc = CVI_NN_RegisterModel(model_path, &model);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "RegisterModel failed: %d\n", rc);
        return EXIT_FAILURE;
    }

    /* Get tensors */
    CVI_TENSOR *inputs, *outputs;
    int32_t n_in, n_out;
    rc = CVI_NN_GetInputOutputTensors(model, &inputs, &n_in, &outputs, &n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "GetInputOutputTensors failed: %d\n", rc);
        CVI_NN_CleanupModel(model);
        return EXIT_FAILURE;
    }

    printf("Inputs: %d  Outputs: %d\n", n_in, n_out);
    for (int i = 0; i < n_in;  i++)
        printf("  in[%d]  '%s'\n", i, CVI_NN_TensorName(&inputs[i]));
    for (int i = 0; i < n_out; i++)
        printf("  out[%d] '%s'\n", i, CVI_NN_TensorName(&outputs[i]));
    printf("\n");

    /* Resolve tensors by name */
    CVI_TENSOR *t_mel    = CVI_NN_GetTensorByName("mel_input", inputs,  n_in);
    CVI_TENSOR *t_logits = CVI_NN_GetTensorByName("logits*",   outputs, n_out);

    if (!t_mel || !t_logits) {
        /* Fall back to index 0 if names changed after quantization */
        printf("WARNING: name lookup failed, using index 0\n");
        t_mel    = &inputs[0];
        t_logits = &outputs[0];
    }

    float *mel_ptr    = (float *)CVI_NN_TensorPtr(t_mel);
    float *logits_ptr = (float *)CVI_NN_TensorPtr(t_logits);

    /* ── Test 1: Zeros (silence → no_wake expected) ─────────────────────── */
    memset(mel_ptr, 0, MEL_ELEMS * sizeof(float));
    rc = CVI_NN_Forward(model, inputs, n_in, outputs, n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "Forward (zeros) failed: %d\n", rc);
        CVI_NN_CleanupModel(model);
        return EXIT_FAILURE;
    }
    printf("Test 1 — zeros (silence):\n");
    printf("  logits: [no_wake=%.4f, wake=%.4f]  → %s\n",
           logits_ptr[0], logits_ptr[1],
           logits_ptr[1] > logits_ptr[0] ? "WAKE" : "no_wake");

    /* ── Test 2: Ones (arbitrary mel → check no crash) ──────────────────── */
    for (int i = 0; i < MEL_ELEMS; i++) mel_ptr[i] = 1.0f;
    rc = CVI_NN_Forward(model, inputs, n_in, outputs, n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "Forward (ones) failed: %d\n", rc);
        CVI_NN_CleanupModel(model);
        return EXIT_FAILURE;
    }
    printf("Test 2 — ones:\n");
    printf("  logits: [no_wake=%.4f, wake=%.4f]  → %s\n",
           logits_ptr[0], logits_ptr[1],
           logits_ptr[1] > logits_ptr[0] ? "WAKE" : "no_wake");

    /* ── Test 3: Latency benchmark ───────────────────────────────────────── */
    printf("\nLatency benchmark (%d runs):\n", N_RUNS);
    long total_us = 0;
    long min_us   = 1000000L;
    long max_us   = 0;

    for (int r = 0; r < N_RUNS; r++) {
        /* Slightly vary input to avoid any caching */
        mel_ptr[r % MEL_ELEMS] = (float)r / N_RUNS;

        long t0 = now_us();
        rc = CVI_NN_Forward(model, inputs, n_in, outputs, n_out);
        long elapsed = now_us() - t0;

        if (rc != CVI_RC_SUCCESS) {
            fprintf(stderr, "  run %d failed: %d\n", r, rc);
            break;
        }
        total_us += elapsed;
        if (elapsed < min_us) min_us = elapsed;
        if (elapsed > max_us) max_us = elapsed;
    }

    printf("  min=%ldus  max=%ldus  avg=%ldus\n",
           min_us, max_us, total_us / N_RUNS);
    printf("\n");

    /* ── Gate: avg latency < 10ms means NPU (not CPU fallback) ────────────── */
    long avg_us = total_us / N_RUNS;
    if (avg_us < 10000) {
        printf("PASS: avg latency %ldus < 10ms → running on NPU\n", avg_us);
    } else {
        printf("WARN: avg latency %ldus ≥ 10ms — may be on CPU\n", avg_us);
    }

    CVI_NN_CleanupModel(model);
    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
