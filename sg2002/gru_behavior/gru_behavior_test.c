/**
 * gru_behavior_test.c — Verify NexuGRUPolicy cvimodel on SG2002 NPU
 *
 * Build: make -C sg2002/gru_behavior/
 * Run:   ./gru_behavior_test /root/gru_behavior.cvimodel
 *
 * Inputs:
 *   obs       [1, 1, 16]  FP32  observation vector
 *   hidden_in [1, 64]     FP32  GRU hidden state
 * Outputs:
 *   action_logits [1, 12] FP32
 *   hidden_out    [1, 64] FP32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cviruntime.h>

#define MODEL_DEFAULT  "/root/gru_behavior.cvimodel"
#define OBS_SIZE       16
#define HIDDEN_SIZE    64
#define NUM_ACTIONS    12
#define N_STEPS        30

static const char *ACTION_NAMES[NUM_ACTIONS] = {
    "idle_sit", "look_around", "approach", "follow",
    "play_gesture", "vocal_happy", "vocal_curious", "vocal_alert",
    "sleep", "avoid_obstacle", "self_groom", "wag_tail"
};

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

static int argmax(const float *arr, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[best]) best = i;
    return best;
}

int main(int argc, char *argv[])
{
    const char *model_path = (argc > 1) ? argv[1] : MODEL_DEFAULT;

    printf("=== Nexu GRU Policy NPU Verification ===\n");
    printf("Model: %s\n\n", model_path);

    CVI_MODEL_HANDLE model;
    CVI_RC rc = CVI_NN_RegisterModel(model_path, &model);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "RegisterModel failed: %d\n", rc);
        return EXIT_FAILURE;
    }

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

    /* Resolve tensors */
    CVI_TENSOR *t_obs    = CVI_NN_GetTensorByName("obs",            inputs,  n_in);
    CVI_TENSOR *t_hidden = CVI_NN_GetTensorByName("hidden_in",      inputs,  n_in);
    CVI_TENSOR *t_logits = CVI_NN_GetTensorByName("action_logits*", outputs, n_out);
    CVI_TENSOR *t_hout   = CVI_NN_GetTensorByName("hidden_out*",    outputs, n_out);

    if (!t_obs || !t_hidden) {
        printf("WARNING: name lookup failed, using indices\n");
        t_obs    = &inputs[0];
        t_hidden = &inputs[1];
    }
    if (!t_logits || !t_hout) {
        t_logits = &outputs[0];
        t_hout   = (n_out > 1) ? &outputs[1] : NULL;
    }

    float *obs_ptr    = (float *)CVI_NN_TensorPtr(t_obs);
    float *hidden_ptr = (float *)CVI_NN_TensorPtr(t_hidden);
    float *logits_ptr = (float *)CVI_NN_TensorPtr(t_logits);
    float *hout_ptr   = t_hout ? (float *)CVI_NN_TensorPtr(t_hout) : NULL;

    /* Zero hidden state */
    memset(hidden_ptr, 0, HIDDEN_SIZE * sizeof(float));

    /* ── Simulated step sequence ─────────────────────────────────────────── */
    printf("Running %d inference steps:\n", N_STEPS);
    long total_us = 0, min_us = 1000000L, max_us = 0;

    for (int step = 0; step < N_STEPS; step++) {
        /* Build a simple obs: idle at first, then person appears at step 10 */
        memset(obs_ptr, 0, OBS_SIZE * sizeof(float));
        obs_ptr[9]  = 0.9f;   /* battery_level */
        obs_ptr[14] = 0.7f;   /* personality_curiosity */
        obs_ptr[15] = 0.6f;   /* personality_playful */

        if (step >= 10 && step < 20) {
            obs_ptr[6] = 1.0f;                        /* person_visible */
            obs_ptr[7] = (step - 10) / 10.0f;        /* person_dist increasing */
        }
        if (step == 5) obs_ptr[8] = 1.0f;   /* wake word */
        if (step == 12) obs_ptr[0] = 1.0f;  /* touch_head */

        long t0 = now_us();
        rc = CVI_NN_Forward(model, inputs, n_in, outputs, n_out);
        long elapsed = now_us() - t0;

        if (rc != CVI_RC_SUCCESS) {
            fprintf(stderr, "  step %d Forward failed: %d\n", step, rc);
            break;
        }

        total_us += elapsed;
        if (elapsed < min_us) min_us = elapsed;
        if (elapsed > max_us) max_us = elapsed;

        int action = argmax(logits_ptr, NUM_ACTIONS);
        printf("  step %2d  %-15s  (%ldus)\n", step, ACTION_NAMES[action], elapsed);

        /* Feed hidden state back */
        if (hout_ptr)
            memcpy(hidden_ptr, hout_ptr, HIDDEN_SIZE * sizeof(float));
    }

    printf("\nLatency: min=%ldus  max=%ldus  avg=%ldus\n",
           min_us, max_us, total_us / N_STEPS);

    long avg_us = total_us / N_STEPS;
    if (avg_us < 10000)
        printf("PASS: avg %ldus < 10ms → NPU execution confirmed\n", avg_us);
    else
        printf("WARN: avg %ldus ≥ 10ms — check for CPU fallback\n", avg_us);

    CVI_NN_CleanupModel(model);
    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
