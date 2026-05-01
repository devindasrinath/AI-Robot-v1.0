#pragma once
#include "obs.h"

extern const char *ACTION_NAMES[NUM_ACTIONS];

/* Temperature sampling — picks action index probabilistically from logits.
   temperature: 0.6 = focused, 1.0 = balanced, 1.4 = random */
int action_sample(const float *logits, float temperature);

/* Dispatch action → send UART commands to ESP32-C3 */
void action_dispatch(int action);
