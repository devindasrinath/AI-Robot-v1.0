#include "i2s_audio.h"

#include <string.h>
#include <math.h>
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "audio";
static i2s_chan_handle_t s_tx_chan;

/* ── I2S init ────────────────────────────────────────────────────────────── */

static void i2s_setup(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &s_tx_chan, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                     I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = AUDIO_BCLK,
            .ws   = AUDIO_LRCLK,
            .dout = AUDIO_DOUT,
            .din  = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx_chan));

    ESP_LOGI(TAG, "I2S ready — PCM5102A @ %d Hz", AUDIO_SAMPLE_RATE);
}

/* ── Tone primitives ─────────────────────────────────────────────────────── */

#define CHUNK 256   /* samples per write batch */

/* Linear frequency sweep from freq_start → freq_end over duration_s seconds.
 * amplitude ∈ [0,1]. Writes stereo 16-bit samples to I2S. */
static void play_sweep(float freq_start, float freq_end,
                        float duration_s, float amplitude)
{
    static int16_t buf[CHUNK * 2];   /* stereo interleaved: L,R,L,R... */
    int total = (int)(AUDIO_SAMPLE_RATE * duration_s);
    int done  = 0;
    float phase = 0.0f;
    const float two_pi = 2.0f * (float)M_PI;

    while (done < total) {
        int n = (total - done < CHUNK) ? (total - done) : CHUNK;
        for (int i = 0; i < n; i++) {
            float t    = (float)(done + i) / (float)total;
            float freq = freq_start + (freq_end - freq_start) * t;
            float s    = amplitude * sinf(phase) * 32767.0f;
            int16_t v  = (int16_t)(s < -32768.0f ? -32768.0f : (s > 32767.0f ? 32767.0f : s));
            buf[i * 2]     = v;
            buf[i * 2 + 1] = v;
            phase += two_pi * freq / AUDIO_SAMPLE_RATE;
            if (phase >= two_pi) phase -= two_pi;
        }
        size_t written;
        i2s_channel_write(s_tx_chan, buf, (size_t)(n * 4), &written, pdMS_TO_TICKS(200));
        done += n;
    }
}

/* Envelope-shaped tone — fade in/out to avoid clicks */
static void play_note(float freq, float duration_s, float amplitude)
{
    static int16_t buf[CHUNK * 2];
    int total     = (int)(AUDIO_SAMPLE_RATE * duration_s);
    int fade_samp = (int)(AUDIO_SAMPLE_RATE * 0.01f);   /* 10ms fade */
    int done      = 0;
    float phase   = 0.0f;
    const float two_pi = 2.0f * (float)M_PI;

    while (done < total) {
        int n = (total - done < CHUNK) ? (total - done) : CHUNK;
        for (int i = 0; i < n; i++) {
            int pos = done + i;
            float env = 1.0f;
            if (pos < fade_samp)            env = (float)pos / fade_samp;
            else if (pos > total - fade_samp) env = (float)(total - pos) / fade_samp;
            float s   = env * amplitude * sinf(phase) * 32767.0f;
            int16_t v = (int16_t)(s < -32768.0f ? -32768.0f : (s > 32767.0f ? 32767.0f : s));
            buf[i * 2]     = v;
            buf[i * 2 + 1] = v;
            phase += two_pi * freq / AUDIO_SAMPLE_RATE;
            if (phase >= two_pi) phase -= two_pi;
        }
        size_t written;
        i2s_channel_write(s_tx_chan, buf, (size_t)(n * 4), &written, pdMS_TO_TICKS(200));
        done += n;
    }
}

static void play_silence(float duration_s)
{
    static int16_t buf[CHUNK * 2];
    memset(buf, 0, sizeof(buf));
    int total = (int)(AUDIO_SAMPLE_RATE * duration_s);
    int done  = 0;
    while (done < total) {
        int n = (total - done < CHUNK) ? (total - done) : CHUNK;
        size_t written;
        i2s_channel_write(s_tx_chan, buf, (size_t)(n * 4), &written, pdMS_TO_TICKS(200));
        done += n;
    }
}

/* ── Vocal sound synthesizers ────────────────────────────────────────────── */

/* Bright ascending arpeggio — cheerful */
static void synth_happy(void)
{
    play_note(261.6f, 0.08f, 0.65f);   /* C4 */
    play_note(329.6f, 0.08f, 0.65f);   /* E4 */
    play_note(392.0f, 0.08f, 0.65f);   /* G4 */
    play_note(523.3f, 0.14f, 0.70f);   /* C5 — held a bit longer */
    play_silence(0.04f);
}

/* Rising glide — inquisitive */
static void synth_curious(void)
{
    play_sweep(280.0f, 480.0f, 0.30f, 0.55f);
    play_silence(0.04f);
}

/* Two sharp beeps — attention/warning */
static void synth_alert(void)
{
    play_note(880.0f, 0.10f, 0.75f);
    play_silence(0.06f);
    play_note(880.0f, 0.10f, 0.75f);
    play_silence(0.04f);
}

/* Playful up-down wobble */
static void synth_play(void)
{
    play_sweep(380.0f, 580.0f, 0.14f, 0.60f);
    play_sweep(580.0f, 380.0f, 0.14f, 0.60f);
    play_sweep(380.0f, 580.0f, 0.14f, 0.60f);
    play_silence(0.05f);
}

/* ── Audio task ──────────────────────────────────────────────────────────── */

static void audio_task(void *arg)
{
    QueueHandle_t q = (QueueHandle_t)arg;
    char name[32];

    while (1) {
        if (xQueueReceive(q, name, portMAX_DELAY) == pdTRUE) {
            /* Strip optional .wav extension sent by SG2002 action dispatcher */
            char *dot = strrchr(name, '.');
            if (dot) *dot = '\0';

            ESP_LOGI(TAG, "Play: %s", name);
            if      (strcmp(name, "vocal_happy")   == 0) synth_happy();
            else if (strcmp(name, "vocal_curious") == 0) synth_curious();
            else if (strcmp(name, "vocal_alert")   == 0) synth_alert();
            else if (strcmp(name, "vocal_play")    == 0) synth_play();
            else ESP_LOGW(TAG, "Unknown sound: %s", name);
        }
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void audio_init(QueueHandle_t play_queue)
{
    i2s_setup();
    xTaskCreate(audio_task, "audio", 4096, play_queue, 3, NULL);
}
