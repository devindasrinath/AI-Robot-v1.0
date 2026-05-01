#pragma once
#include <stdbool.h>
#include "obs.h"

/* Load both models. Call once at startup.
   Returns 0 on success, -1 on failure. */
int  inference_init(const char *wake_word_path, const char *gru_path);

void inference_destroy(void);

/* Run wake word model on mel spectrogram (MEL_ELEMS floats).
   Returns true if "Nexu" detected. */
bool wake_word_run(const float *mel);

/* Run GRU policy.
   obs_arr : float[OBS_SIZE]   — current observation
   hidden  : float[HIDDEN_SIZE] — GRU hidden state (in/out, update in-place)
   logits  : float[NUM_ACTIONS] — raw action logits output
   Returns 0 on success. */
int  gru_run(const float *obs_arr, float *hidden, float *logits);
