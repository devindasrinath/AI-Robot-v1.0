/*
 * Nexu main loop — SG2002 brain
 *
 * Cycle (100ms / 10Hz):
 *   1. Collect observations (touch/collision from UART, face from stub)
 *   2. Every 10 cycles (1s): run wake word on mel window
 *   3. Run GRU behavior policy on NPU
 *   4. Temperature-sample action
 *   5. Dispatch action → UART → ESP32-C3
 *
 * Usage:
 *   ./nexu_main [wake_word.cvimodel] [gru_behavior.cvimodel]
 *
 * Run on board:
 *   export LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib
 *   ./nexu_main /root/wake_word.cvimodel /root/gru_behavior.cvimodel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <signal.h>

#include "obs.h"
#include "inference.h"
#include "uart_sg.h"
#include "audio.h"
#include "action.h"

#define LOOP_MS          100    /* main loop period */
#define WAKE_WORD_EVERY  10     /* run wake word every N ticks (1s) */
#define TOUCH_DECAY_TICKS 600   /* 60s before time_since_touch reaches 1.0 */
#define IDLE_DECAY_TICKS  300   /* 30s before idle_ticks_norm reaches 1.0 */
#define TEMPERATURE       0.8f

#define WW_MODEL_DEFAULT  "/root/wake_word.cvimodel"
#define GRU_MODEL_DEFAULT "/root/gru_behavior.cvimodel"

static volatile int s_running = 1;

static void on_signal(int sig) { (void)sig; s_running = 0; }

static long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

int main(int argc, char *argv[])
{
    const char *ww_path  = (argc > 1) ? argv[1] : WW_MODEL_DEFAULT;
    const char *gru_path = (argc > 2) ? argv[2] : GRU_MODEL_DEFAULT;

    printf("=== Nexu Main Loop ===\n");
    printf("Wake word model : %s\n", ww_path);
    printf("GRU model       : %s\n\n", gru_path);

    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    srand((unsigned)time(NULL));

    /* ── Init subsystems ─────────────────────────────────────────────────── */
    if (inference_init(ww_path, gru_path) != 0) return EXIT_FAILURE;
    uart_sg_init();
    audio_init();

    /* ── Personality — randomised once per boot, fixed for session ─────────── */
    float personality_curiosity = 0.4f + ((float)rand() / RAND_MAX) * 0.5f;
    float personality_playful   = 0.4f + ((float)rand() / RAND_MAX) * 0.5f;
    printf("[main] Personality: curiosity=%.2f  playful=%.2f\n\n",
           personality_curiosity, personality_playful);

    /* ── State ───────────────────────────────────────────────────────────── */
    float hidden[HIDDEN_SIZE];
    float logits[NUM_ACTIONS];
    float mel[MEL_ELEMS];
    memset(hidden, 0, sizeof(hidden));

    ObsVector obs = {0};
    obs.battery_level          = 0.95f;
    obs.personality_curiosity  = personality_curiosity;
    obs.personality_playful    = personality_playful;

    int   tick            = 0;
    int   ticks_since_touch = 0;
    int   idle_ticks      = 0;
    int   last_action     = 0;

    /* ── Main loop ───────────────────────────────────────────────────────── */
    printf("[main] Running at %dHz. Ctrl+C to stop.\n\n", 1000 / LOOP_MS);

    while (s_running) {
        long t_start = now_ms();
        tick++;

        /* ── 1. Consume UART events from ESP32-C3 ─────────────────────── */
        obs.touch_head      = g_uart_events.touch_head      ? 1.0f : 0.0f;
        obs.touch_body      = g_uart_events.touch_body      ? 1.0f : 0.0f;
        obs.touch_tail      = g_uart_events.touch_tail      ? 1.0f : 0.0f;
        obs.collision_front = g_uart_events.collision_front ? 1.0f : 0.0f;
        obs.collision_left  = g_uart_events.collision_left  ? 1.0f : 0.0f;
        obs.collision_right = g_uart_events.collision_right ? 1.0f : 0.0f;

        /* Track time since last touch */
        if (obs.touch_head > 0.5f || obs.touch_body > 0.5f || obs.touch_tail > 0.5f)
            ticks_since_touch = 0;
        else
            ticks_since_touch++;
        obs.time_since_touch = clampf((float)ticks_since_touch / TOUCH_DECAY_TICKS, 0, 1);

        /* Clear consumed events */
        g_uart_events.touch_head      = false;
        g_uart_events.touch_body      = false;
        g_uart_events.touch_tail      = false;
        g_uart_events.collision_front = false;
        g_uart_events.collision_left  = false;
        g_uart_events.collision_right = false;

        /* ── 2. Face detection — read result from face_watcher.sh ───────── */
        {
            FILE *fface = fopen("/tmp/nexu_face", "r");
            if (fface) {
                int vis = 0; float dist = 0.0f;
                if (fscanf(fface, "%d %f", &vis, &dist) == 2) {
                    obs.person_visible  = vis  ? 1.0f : 0.0f;
                    obs.person_distance = dist > 1.0f ? 1.0f : dist;
                }
                fclose(fface);
            } else {
                obs.person_visible  = 0.0f;
                obs.person_distance = 0.0f;
            }
        }

        /* ── 3. Wake word (every 1s) ──────────────────────────────────── */
        obs.wake_word = 0.0f;
        if (tick % WAKE_WORD_EVERY == 0) {
            audio_get_mel(mel);
            if (wake_word_run(mel)) {
                obs.wake_word = 1.0f;
                printf("[main] Wake word detected!\n");
            }
        }

        /* ── 4. Idle tracking ─────────────────────────────────────────── */
        if (last_action == 0 || last_action == 10)  /* idle_sit or self_groom */
            idle_ticks++;
        else
            idle_ticks = 0;
        obs.idle_ticks_norm = clampf((float)idle_ticks / IDLE_DECAY_TICKS, 0, 1);

        /* ── 5. GRU inference ─────────────────────────────────────────── */
        float obs_arr[OBS_SIZE];
        obs_to_array(&obs, obs_arr);

        if (gru_run(obs_arr, hidden, logits) != 0) {
            fprintf(stderr, "[main] GRU inference failed\n");
            usleep(LOOP_MS * 1000);
            continue;
        }

        /* ── 6. Sample and dispatch action ───────────────────────────── */
        int action = action_sample(logits, TEMPERATURE);
        action_dispatch(action);
        last_action = action;

        /* ── 7. Timing — sleep remainder of 100ms ────────────────────── */
        long elapsed = now_ms() - t_start;
        long sleep_ms = LOOP_MS - elapsed;
        if (sleep_ms > 0)
            usleep((useconds_t)(sleep_ms * 1000));
    }

    printf("\n[main] Shutting down.\n");
    uart_sg_send("DISPLAY:sleep\n");
    inference_destroy();
    return EXIT_SUCCESS;
}
