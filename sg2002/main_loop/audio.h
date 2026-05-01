#pragma once
#include <stdbool.h>
#include "obs.h"

/* Initialize audio capture (ALSA USB mic at hw:1,0).
   Returns 0 if mic found, -1 if unavailable (stub mode). */
int  audio_init(void);

/* Fill mel[MEL_ELEMS] with the latest 1-second mel spectrogram.
   Returns true if new audio was captured, false if stub/unavailable.
   Non-blocking — returns immediately with last captured window. */
bool audio_get_mel(float *mel);
