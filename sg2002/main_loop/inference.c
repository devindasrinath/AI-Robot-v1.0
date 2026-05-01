#include "inference.h"

#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <cviruntime.h>

/* ── NPU mutex — CV181x is single-context: only one Forward at a time ─────── */
static pthread_mutex_t s_npu_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ── Wake word model ─────────────────────────────────────────────────────── */
static CVI_MODEL_HANDLE s_ww_model;
static CVI_TENSOR *s_ww_inputs,  *s_ww_outputs;
static int32_t     s_ww_n_in,     s_ww_n_out;
static float      *s_ww_mel_ptr;
static float      *s_ww_logits_ptr;

/* ── GRU model ───────────────────────────────────────────────────────────── */
static CVI_MODEL_HANDLE s_gru_model;
static CVI_TENSOR *s_gru_inputs,  *s_gru_outputs;
static int32_t     s_gru_n_in,     s_gru_n_out;
static float      *s_gru_obs_ptr;
static float      *s_gru_hidden_ptr;
static float      *s_gru_logits_ptr;
static float      *s_gru_hout_ptr;

/* ── Init ────────────────────────────────────────────────────────────────── */

int inference_init(const char *wake_word_path, const char *gru_path)
{
    CVI_RC rc;

    /* Wake word */
    rc = CVI_NN_RegisterModel(wake_word_path, &s_ww_model);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] wake word RegisterModel failed: %d\n", rc);
        return -1;
    }
    rc = CVI_NN_GetInputOutputTensors(s_ww_model,
                                       &s_ww_inputs,  &s_ww_n_in,
                                       &s_ww_outputs, &s_ww_n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] wake word GetTensors failed: %d\n", rc);
        return -1;
    }

    CVI_TENSOR *t_mel    = CVI_NN_GetTensorByName("mel_input", s_ww_inputs,  s_ww_n_in);
    CVI_TENSOR *t_logits = CVI_NN_GetTensorByName("logits*",   s_ww_outputs, s_ww_n_out);
    if (!t_mel)    t_mel    = &s_ww_inputs[0];
    if (!t_logits) t_logits = &s_ww_outputs[0];

    s_ww_mel_ptr    = (float *)CVI_NN_TensorPtr(t_mel);
    s_ww_logits_ptr = (float *)CVI_NN_TensorPtr(t_logits);
    printf("[inference] Wake word model ready\n");

    /* GRU */
    rc = CVI_NN_RegisterModel(gru_path, &s_gru_model);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] GRU RegisterModel failed: %d\n", rc);
        return -1;
    }
    rc = CVI_NN_GetInputOutputTensors(s_gru_model,
                                       &s_gru_inputs,  &s_gru_n_in,
                                       &s_gru_outputs, &s_gru_n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] GRU GetTensors failed: %d\n", rc);
        return -1;
    }

    CVI_TENSOR *t_obs  = CVI_NN_GetTensorByName("obs",            s_gru_inputs,  s_gru_n_in);
    CVI_TENSOR *t_hin  = CVI_NN_GetTensorByName("hidden_in",      s_gru_inputs,  s_gru_n_in);
    CVI_TENSOR *t_act  = CVI_NN_GetTensorByName("action_logits*", s_gru_outputs, s_gru_n_out);
    CVI_TENSOR *t_hout = CVI_NN_GetTensorByName("hidden_out*",    s_gru_outputs, s_gru_n_out);
    if (!t_obs)  t_obs  = &s_gru_inputs[0];
    if (!t_hin)  t_hin  = &s_gru_inputs[1];
    if (!t_act)  t_act  = &s_gru_outputs[0];
    if (!t_hout) t_hout = (s_gru_n_out > 1) ? &s_gru_outputs[1] : NULL;

    s_gru_obs_ptr    = (float *)CVI_NN_TensorPtr(t_obs);
    s_gru_hidden_ptr = (float *)CVI_NN_TensorPtr(t_hin);
    s_gru_logits_ptr = (float *)CVI_NN_TensorPtr(t_act);
    s_gru_hout_ptr   = t_hout ? (float *)CVI_NN_TensorPtr(t_hout) : NULL;
    printf("[inference] GRU model ready\n");

    return 0;
}

void inference_destroy(void)
{
    CVI_NN_CleanupModel(s_ww_model);
    CVI_NN_CleanupModel(s_gru_model);
}

/* ── Wake word ───────────────────────────────────────────────────────────── */

bool wake_word_run(const float *mel)
{
    memcpy(s_ww_mel_ptr, mel, MEL_ELEMS * sizeof(float));

    pthread_mutex_lock(&s_npu_mutex);
    CVI_RC rc = CVI_NN_Forward(s_ww_model, s_ww_inputs, s_ww_n_in,
                                s_ww_outputs, s_ww_n_out);
    pthread_mutex_unlock(&s_npu_mutex);

    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] wake word Forward failed: %d\n", rc);
        return false;
    }
    return s_ww_logits_ptr[1] > s_ww_logits_ptr[0];
}

/* ── GRU ─────────────────────────────────────────────────────────────────── */

int gru_run(const float *obs_arr, float *hidden, float *logits)
{
    memcpy(s_gru_obs_ptr,    obs_arr, OBS_SIZE    * sizeof(float));
    memcpy(s_gru_hidden_ptr, hidden,  HIDDEN_SIZE * sizeof(float));

    pthread_mutex_lock(&s_npu_mutex);
    CVI_RC rc = CVI_NN_Forward(s_gru_model, s_gru_inputs, s_gru_n_in,
                                s_gru_outputs, s_gru_n_out);
    pthread_mutex_unlock(&s_npu_mutex);

    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[inference] GRU Forward failed: %d\n", rc);
        return -1;
    }

    memcpy(logits, s_gru_logits_ptr, NUM_ACTIONS * sizeof(float));
    if (s_gru_hout_ptr)
        memcpy(hidden, s_gru_hout_ptr, HIDDEN_SIZE * sizeof(float));
    return 0;
}
