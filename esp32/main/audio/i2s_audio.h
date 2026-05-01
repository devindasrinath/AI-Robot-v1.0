#pragma once
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* I2S pins for PCM5102A DAC */
#define AUDIO_BCLK    0   /* GPIO0  — bit clock  */
#define AUDIO_LRCLK   1   /* GPIO1  — word select */
#define AUDIO_DOUT   10   /* GPIO10 — serial data to PCM5102 DIN */

#define AUDIO_SAMPLE_RATE  44100

/* audio_init — set up I2S peripheral and launch audio task.
 * play_queue: QueueHandle holding char[32] sound names (e.g. "vocal_happy"). */
void audio_init(QueueHandle_t play_queue);
